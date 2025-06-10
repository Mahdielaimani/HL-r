"""
PCAP Preprocessing for Pre-filtered TCP Flows
-----------------------------
- Reads a PCAP file containing pre-filtered TCP flows.
- Samples a fixed number of packets.
- Segments or creates n-grams from packets.
- Encodes the data into PyTorch tensors.
- Pads all samples to the same length.
- Yields batches of data.

By Rima Daqch
"""
# ----------------------------------------------------------------------------------------------------
# Importing necessary libraries 
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from scapy.all import PcapReader, raw
import collections
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ----------------------------------------------------------------------------------------------------
def segment_packet(packet, segment_size=2):
    chunks = [packet[i:i+segment_size] for i in range(0, len(packet), segment_size)]
    if chunks and len(chunks[-1]) < segment_size:
        chunks[-1] += ['<PAD>'] * (segment_size - len(chunks[-1]))
    return chunks

# ----------------------------------------------------------------------------------------------------
def n_gram(info_hex, n_gram_size=2, overlap=1):
    ngram_list = []
    ngram_list_conc = []
    overlap_size = int(n_gram_size * overlap)
    for i in range(0, len(info_hex), overlap_size):
        if i + n_gram_size <= len(info_hex):
            gram = info_hex[i:i+n_gram_size]
            ngram_list.append(gram)
            ngram_list_conc.append(' '.join(gram))
    return np.asarray(ngram_list_conc), np.asarray(ngram_list)

# ----------------------------------------------------------------------------------------------------
def batch_hex_dict(packets, batchsize, segment_size, overlap, mode, k=1):
    pkt_dict = {}
    for i, p in enumerate(packets):
        raw_packet = raw(p)
        info = [raw_packet[j:j+k] for j in range(0, len(raw_packet), k)]
        info_hex = [byte.hex() for byte in info]
        
        if mode == 'ngram':
            _, ngrammed_pkt = n_gram(info_hex, segment_size, overlap)
            pkt_dict[f'packet{i}'] = ngrammed_pkt
        elif mode == 'segment':
            segmented_pkt = segment_packet(info_hex, segment_size=segment_size)
            pkt_dict[f'packet{i}'] = segmented_pkt
        
        if (i+1) % batchsize == 0:
            yield pkt_dict
            pkt_dict = {}
    if pkt_dict:
        yield pkt_dict

# ----------------------------------------------------------------------------------------------------
def pad_tensors(tensor_list):
    if not tensor_list:
        return None
    max_size = max(tensor.size(0) for tensor in tensor_list)
    padded_list = []
    for tensor in tensor_list:
        pad_size = max_size - tensor.size(0)
        padding = (0, 0, 0, pad_size)
        padded_tensor = F.pad(tensor, padding, "constant", 256)
        padded_list.append(padded_tensor)
    return torch.stack(padded_list, dim=0)

# ----------------------------------------------------------------------------------------------------
def encode_from_hex(x):
    return 256 if x == '<PAD>' else int.from_bytes(bytes.fromhex(x), byteorder='big')

# ----------------------------------------------------------------------------------------------------
def encode_ngram(ngrammed_array):
    vectorized_encode = np.vectorize(lambda x: encode_from_hex(x))
    encoded = torch.tensor(np.array([vectorized_encode(packet) for packet in ngrammed_array]))
    return encoded

# ----------------------------------------------------------------------------------------------------
class PCAPTCPFlowDataset(IterableDataset):
    def __init__(self, pcap_file, batch_size=32, max_packets=100, 
                 segment_size=32, overlap=1, mode='segment', bytes_per_unit=1):
        self.pcap_file = pcap_file
        self.batch_size = batch_size
        self.max_packets = max_packets
        self.segment_size = segment_size
        self.overlap = overlap
        self.mode = mode
        self.bytes_per_unit = bytes_per_unit

    def __iter__(self):
        print(f"Reading PCAP file: {self.pcap_file}")
        with PcapReader(self.pcap_file) as pcap_reader:
            while True:
                current_batch = []
                for _ in range(self.batch_size):
                    try:
                        packet = next(pcap_reader)
                        packet_data = self.process_packet(packet)
                        current_batch.append(packet_data)
                    except StopIteration:
                        break

                if not current_batch:
                    break

                processed_batch = self.process_batch(current_batch)
                yield processed_batch
    
    def process_packet(self, packet):
        raw_packet = raw(packet)
        info = [raw_packet[j:j+self.bytes_per_unit] for j in range(0, len(raw_packet), self.bytes_per_unit)]
        info_hex = [byte.hex() for byte in info]
        
        if self.mode == 'ngram':
            _, processed_pkt = n_gram(info_hex, self.segment_size, self.overlap)
        else:
            processed_pkt = segment_packet(info_hex, segment_size=self.segment_size)

        vectorized_encode = np.vectorize(lambda x: encode_from_hex(x))
        encoded = torch.tensor(np.array([vectorized_encode(segment) for segment in processed_pkt]))
        return encoded
    
    def process_batch(self, packet_tensors):
        max_rows = max(tensor.size(0) for tensor in packet_tensors)
        max_cols = 0
        for tensor in packet_tensors:
            if tensor.dim() > 1 and tensor.size(1) > max_cols:
                max_cols = tensor.size(1)
        if max_cols == 0 and packet_tensors:
            max_cols = self.segment_size
        
        padded_batch = []
        for tensor in packet_tensors:
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(1)

            if tensor.size(0) < max_rows:
                row_padding = torch.full((max_rows - tensor.size(0), tensor.size(1)), 256, dtype=torch.long)
                tensor = torch.cat([tensor, row_padding], dim=0)
            if tensor.size(1) < max_cols:
                col_padding = torch.full((tensor.size(0), max_cols - tensor.size(1)), 256, dtype=torch.long)
                tensor = torch.cat([tensor, col_padding], dim=1)

            padded_batch.append(tensor)
        return torch.stack(padded_batch)
0
# ----------------------------------------------------------------------------------------------------
def process_pcap_file(pcap_file, batch_size=32, max_iterations=1000):
    dataset = PCAPTCPFlowDataset(
        pcap_file=pcap_file,
        batch_size=batch_size,
        max_packets=1000,
        segment_size=32,
        mode='segment',
        bytes_per_unit=1
    )
    dataloader = DataLoader(dataset, batch_size=None)

    for i, data_batch in enumerate(dataloader):
        if i >= max_iterations:
            break
        print(f"Iteration {i}")
        print(f"Data batch shape: {data_batch.shape}")
        print("###############################################")
        print(f"Data batch: {data_batch}")
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR = os.getenv('DATA_DIR')
    pcap_files = [
        f"{DATA_DIR}/Monday-filtred.pcap"
    ]

    dataset = PCAPTCPFlowDataset(
        pcap_file=pcap_files[0], 
        batch_size=32,
        max_packets=1000,
        segment_size=32,
        mode='segment',
        bytes_per_unit=1
    )

    for data_batch in dataset:  
        print("Data batch shape:", data_batch.shape)
        print("###############################################")
        print("Data batch:", data_batch)
        print("===============================================")

# ----------------------------------------------------------------------------------------------------
# End of script
# ----------------------------------------------------------------------------------------------------

  
               