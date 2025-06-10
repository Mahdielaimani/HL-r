import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bytes_level import ByteHierarchyEncoder, ByteHierarchyDecoder
from segments_level import SegmentHierarchyEncoder, SegmentHierarchyDecoder


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")


class nvae(nn.Module):
    """
    Architecture:
    Input -> Byte Encoder -> Segment Encoder -> Segment Decoder -> Byte Decoder -> Output
    """
    
    def __init__(self, 
                 embedding_size=32, 
                 num_residual_levels=3,
                 beta=1.0,
                 free_bits=0.0):
        super(nvae, self).__init__()
        
        self.embedding_size = embedding_size
        self.num_residual_levels = num_residual_levels
        self.beta = beta 
        self.free_bits = free_bits  
        
        print_memory_usage()

        # Level 0: Input embedding
        self.input_embedding = nn.Embedding(257, embedding_size)
        self.input_embedding.weight.requires_grad = False
        
        # Level 1: Byte-level hierarchy
        self.byte_encoder = ByteHierarchyEncoder(
            embedding_size=embedding_size,
            num_residual_levels=num_residual_levels
        )
        
        self.byte_decoder = ByteHierarchyDecoder(
            embedding_size=embedding_size,
            num_residual_levels=num_residual_levels
        )
        
        print("Byte level components initialized")
        print_memory_usage()
        
        # Level 2: Segment-level hierarchy
        self.segment_encoder = SegmentHierarchyEncoder(
            embedding_size=embedding_size
        )
        
        self.segment_decoder = SegmentHierarchyDecoder(
            embedding_size=embedding_size
        )
        
        print("Segment level components initialized")
        print_memory_usage()
        
        # connect byte-level to segment-level 
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        print("Two-Level NVAE initialization complete")
        print_memory_usage()
    
    def forward(self, x, mode='train'):
        """

        Args:
            x: Input tensor of shape (batch_size, num_segments, segment_size, embedding_size)
            mode: 'train' or 'generate'
        
        Returns:
            Dictionary containing outputs from both levels and loss components
        """
        print("x before embedding: ", x.shape)
        x = x.long() #ent
        x = self.input_embedding(x)
        
        print("Shape of x:", x.shape)

        batch_size, num_segments, segment_size, embedding_size = x.shape
        
        # ===========================================
        # ENCODING PHASE
        # ===========================================
        
        # Level 1: Byte-level encoding
        byte_outputs = self.byte_encoder(x)
        
        # Extract byte-level latent variables
        byte_z1 = byte_outputs['z1']  # (batch_size*num_segments, embedding_size, segment_size)
        byte_z2 = byte_outputs['z2']
        byte_z3 = byte_outputs['z3']
        byte_res3_out = byte_outputs['res3_out']
        
        byte_pooled = self.global_pool(byte_res3_out).squeeze(-1)  # (batch_size*num_segments, embedding_size)
        
        # Reshape back to segment structure
        byte_pooled = byte_pooled.view(batch_size, num_segments, self.embedding_size)
        
        segment_input = byte_pooled  # (batch_size, num_segments, embedding_size)
        
        # Level 2: Segment-level encoding
        segment_outputs = self.segment_encoder(segment_input)
        
        # Extract segment-level latent variables
        segment_z1 = segment_outputs['z1']  # (batch_size, embedding_size, num_segments)
        segment_z2 = segment_outputs['z2']
        segment_z3 = segment_outputs['z3']
        
        # ===========================================
        # DECODING PHASE
        # ===========================================
        
        # Level 2: Segment-level decoding
        segment_decoded = self.segment_decoder(segment_outputs, mode=mode)
        segment_reconstruction = segment_decoded['output']  # (batch_size, num_segments, embedding_size)
        
        segment_expanded = segment_reconstruction.unsqueeze(2).expand(-1, -1, segment_size, -1)
        segment_expanded = segment_expanded.reshape(batch_size * num_segments, embedding_size, segment_size)
        
        combined_byte_outputs = {
            'z1': byte_z1,
            'z2': byte_z2,
            'z3': byte_z3,
            'mu_z1': byte_outputs['mu_z1'],
            'logvar_z1': byte_outputs['logvar_z1'],
            'mu_z2': byte_outputs['mu_z2'],
            'logvar_z2': byte_outputs['logvar_z2'],
            'mu_z3': byte_outputs['mu_z3'],
            'logvar_z3': byte_outputs['logvar_z3']
        }
        
        # Level 1: Byte-level decoding
        byte_decoded = self.byte_decoder(combined_byte_outputs, mode=mode)
        byte_reconstruction = byte_decoded['output']  # (batch_size*num_segments, embedding_size, segment_size)
        
        final_reconstruction = byte_reconstruction.view(batch_size, num_segments, embedding_size, segment_size)
        final_reconstruction = final_reconstruction.permute(0, 1, 3, 2)  # (batch_size, num_segments, segment_size, embedding_size)
        
        # ===========================================
        # LOSS COMPUTATION
        # ===========================================
        
        # Reconstruction loss
        recon_loss = F.mse_loss(final_reconstruction, x, reduction='mean')
        
        # KL divergence losses for byte level
        byte_kl_loss = (
            self.kl_divergence(byte_outputs['mu_z1'], byte_outputs['logvar_z1']) +
            self.kl_divergence(byte_outputs['mu_z2'], byte_outputs['logvar_z2']) +
            self.kl_divergence(byte_outputs['mu_z3'], byte_outputs['logvar_z3'])
        )
        
        # KL divergence losses for segment level
        segment_kl_loss = (
            self.kl_divergence(segment_outputs['mu_z1'], segment_outputs['logvar_z1']) +
            self.kl_divergence(segment_outputs['mu_z2'], segment_outputs['logvar_z2']) +
            self.kl_divergence(segment_outputs['mu_z3'], segment_outputs['logvar_z3'])
        )
        
        total_kl_loss = byte_kl_loss + segment_kl_loss
        
        # Total loss with beta 
        total_loss = recon_loss + self.beta * total_kl_loss
        
        return {
            'reconstruction': final_reconstruction,
            'segment_reconstruction': segment_reconstruction,
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': total_kl_loss,
            'byte_kl_loss': byte_kl_loss,
            'segment_kl_loss': segment_kl_loss,
            'byte_outputs': byte_outputs,
            'segment_outputs': segment_outputs,
            'latent_vars': {
                'byte_z1': byte_z1, 'byte_z2': byte_z2, 'byte_z3': byte_z3,
                'segment_z1': segment_z1, 'segment_z2': segment_z2, 'segment_z3': segment_z3
            }
        }
    
    def generate(self, batch_size=1, num_segments=10, segment_size=32, temp=1.0, device='cpu'):
        """
        Generate new samples from the learned distribution
        
        Args:
            batch_size: Number of samples to generate
            num_segments: Number of segments per sample
            segment_size: Size of each segment
            temp: Temperature for sampling (higher = more random)
            device: Device to generate on
        
        Returns:
            Generated samples
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior distributions
            # Byte-level priors
            byte_mu_z1 = torch.zeros(batch_size * num_segments, self.embedding_size, segment_size).to(device)
            byte_logvar_z1 = torch.zeros(batch_size * num_segments, self.embedding_size, segment_size).to(device)
            byte_mu_z2 = torch.zeros(batch_size * num_segments, self.embedding_size, segment_size).to(device)
            byte_logvar_z2 = torch.zeros(batch_size * num_segments, self.embedding_size, segment_size).to(device)
            byte_mu_z3 = torch.zeros(batch_size * num_segments, self.embedding_size, segment_size).to(device)
            byte_logvar_z3 = torch.zeros(batch_size * num_segments, self.embedding_size, segment_size).to(device)
            
            # Segment-level priors
            segment_mu_z1 = torch.zeros(batch_size, self.embedding_size, num_segments).to(device)
            segment_logvar_z1 = torch.zeros(batch_size, self.embedding_size, num_segments).to(device)
            segment_mu_z2 = torch.zeros(batch_size, self.embedding_size, num_segments).to(device)
            segment_logvar_z2 = torch.zeros(batch_size, self.embedding_size, num_segments).to(device)
            segment_mu_z3 = torch.zeros(batch_size, self.embedding_size, num_segments).to(device)
            segment_logvar_z3 = torch.zeros(batch_size, self.embedding_size, num_segments).to(device)
            
            # Create output dictionaries
            byte_outputs = {
                'mu_z1': byte_mu_z1, 'logvar_z1': byte_logvar_z1,
                'mu_z2': byte_mu_z2, 'logvar_z2': byte_logvar_z2,
                'mu_z3': byte_mu_z3, 'logvar_z3': byte_logvar_z3
            }
            
            segment_outputs = {
                'mu_z1': segment_mu_z1, 'logvar_z1': segment_logvar_z1,
                'mu_z2': segment_mu_z2, 'logvar_z2': segment_logvar_z2,
                'mu_z3': segment_mu_z3, 'logvar_z3': segment_logvar_z3
            }
            
            # Generate from segment level first
            segment_generated = self.segment_decoder.sample(segment_outputs, temp=temp)
            
            # Generate from byte level
            byte_generated = self.byte_decoder.sample(byte_outputs, temp=temp)
            
            # Reshape and combine
            final_generated = byte_generated['output'].view(batch_size, num_segments, self.embedding_size, segment_size)
            final_generated = final_generated.permute(0, 1, 3, 2)
            
            return final_generated
    
    def encode(self, x):
        """
        Encode input to latent space (inference mode)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, mode='train')
    
    def decode(self, latent_vars):
        """
        Decode from latent variables
        """
        self.eval()
        with torch.no_grad():
            pass
    
    def kl_divergence(self, mu, logvar):
        """
        Compute KL divergence with free bits
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2])
        
        if self.free_bits > 0:
            kl = torch.clamp(kl, min=self.free_bits)
        
        return kl.mean()
    
    def set_beta(self, beta):
        """Update beta parameter for KL weighting"""
        self.beta = beta
    
    def get_model_size(self):
        """Return model parameter count and memory usage"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'memory_mb': param_size / (1024 ** 2)
        }