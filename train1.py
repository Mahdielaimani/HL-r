import os
import gc
import psutil
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import Normalize
import torch.amp
from torch.utils.tensorboard import SummaryWriter
import wandb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch.nn.functional as F
from sklearn.manifold import TSNE

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.preprocessing import PCAPTCPFlowDataset
from nvae import nvae  

#------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------Memory Optimization Functions------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

def aggressive_memory_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def set_memory_efficient_settings():
    """Set PyTorch memory efficient settings"""
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        # Reduce memory fragmentation
        torch.cuda.set_per_process_memory_fraction(0.8)  # only 80% 

def check_memory_usage(stage=""):
    """Check and print memory usage with threshold warning"""
    process = psutil.Process(os.getpid())
    rss_memory = process.memory_info().rss / (1024 ** 3)  # GB
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        print(f"{stage} Memory - CPU: {rss_memory:.2f}GB | GPU Allocated: {allocated:.2f}GB | GPU Reserved: {reserved:.2f}GB")
        
        # Warning thresholds
        if rss_memory > 12:  # More than 12GB CPU
            print(f"WARNING: High CPU memory usage: {rss_memory:.2f}GB")
        if allocated > 6:    # More than 6GB GPU
            print(f"WARNING: High GPU memory usage: {allocated:.2f}GB")
            
        return rss_memory, allocated, reserved
    else:
        print(f"{stage} Memory - CPU: {rss_memory:.2f}GB")
        return rss_memory, 0, 0

#------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------Argument parsing-------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Train HierarchicalVAE on PCAP data with memory optimization')
parser.add_argument('--data_dir', type=str, default=os.getenv('DATA_DIR', '/home/cyberai/packetGen/data/filtred-tcp'),
                    help='Directory containing PCAP data')
parser.add_argument('--pcap_file', type=str, default='Monday-filtred.pcap',
                    help='PCAP filename within data directory')
parser.add_argument('--output_dir', type=str, default='./models',
                    help='Directory to save model checkpoints')
parser.add_argument('--log_dir', type=str, default='./logs',
                    help='Directory for tensorboard logs')
parser.add_argument('--batch_size', type=int, default=64,  
                    help='Training batch size (reduced for memory efficiency)')
parser.add_argument('--val_size', type=float, default=0.1,
                    help='Validation set size (fraction of dataset)')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, 
                    help='Learning rate')
parser.add_argument('--beta_byte', type=float, default=0.3,  
                    help='KL divergence weight for byte level (beta-VAE parameter)')
parser.add_argument('--beta_segment', type=float, default=0.2,  
                    help='KL divergence weight for segment level (beta-VAE parameter)')
parser.add_argument('--segment_size', type=int, default=32,  
                    help='Segment size for processing')
parser.add_argument('--use_amp', action='store_true', default=True,  
                    help='Use Automatic Mixed Precision (enabled by default for memory efficiency)')
parser.add_argument('--use_checkpointing', action='store_true', default=True,
                    help='Use gradient checkpointing (enabled by default for memory efficiency)')
parser.add_argument('--save_interval', type=int, default=10,  
                    help='Save model every N epochs')
parser.add_argument('--resume', type=str, default=None,
                    help='Resume training from checkpoint file')
parser.add_argument('--use_wandb', action='store_true', default=True,
                    help='Use Weights & Biases for logging')
parser.add_argument('--wandb_project', type=str, default='CyberAI_PacketGen_VAE',
                    help='Weights & Biases project name')
parser.add_argument('--wandb_entity', type=str, default='cyberai-aim',
                    help='Weights & Biases entity name')
parser.add_argument('--run_name', type=str, default='test1',
                    help='Name for this training run')
parser.add_argument('--max_batches', type=int, default=500, 
                    help='Maximum number of batches per epoch')
parser.add_argument('--patience', type=int, default=15,     
                    help='Early stopping patience')
parser.add_argument('--min_delta', type=float, default=1e-4,
                    help='Minimum change for early stopping')
parser.add_argument('--accumulation_steps', type=int, default=4,  
                    help='Gradient accumulation steps to simulate larger batch size')
parser.add_argument('--memory_efficient', action='store_true', default=True,
                    help='Enable aggressive memory optimization')
parser.add_argument('--embedding_size', type=int, default=128, 
                   help='Embedding size for the model ')
parser.add_argument('--num_residual_levels', type=int, default=3,
                   help='Number of residual levels (default: 3)')
parser.add_argument('--free_bits', type=float, default=0.0,
                   help='Free bits for KL regularization (default: 0.0)')

args = parser.parse_args()

#------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------Helper Classes/Functions-----------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
def preprocess_packet_data(x):
    """
    Preprocess packet data for gradient computation
    """
    if not x.dtype.is_floating_point:
        x = x.float()
    
    if x.max() > 1.0:
        x = x / 256.0  
    
    if x.requires_grad is False:
        x = x.float()
        x.requires_grad_()
    
    return x
#------------------------------------------------------------------

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                device_weights = {k: v.to(next(model.parameters()).device) for k, v in self.best_weights.items()}
                model.load_state_dict(device_weights)
            return True
        return False

#------------------------------------------------------------------------------------------------------------------------

def beta_annealing(epoch, max_epochs=100, final_beta=0.3, level='byte'):
    """Conservative beta annealing for memory efficiency"""
    if level == 'segment':
        final_beta *= 0.6  
    
    if epoch < 10:  
        return 0.001 
    elif epoch < 40:
        progress = (epoch - 10) / 30
        return 0.001 + (0.03 - 0.001) * progress  
    elif epoch < 80:
        progress = (epoch - 40) / 40
        return 0.03 + (final_beta - 0.03) * progress
    else:
        return final_beta

#------------------------------------------------------------------------------------------------------------------------

def cosine_lr_schedule(epoch, initial_lr=1e-4, min_lr=1e-6, warmup_epochs=10, max_epochs=100):
    """Cosine annealing with warmup for learning rate"""
    if epoch < warmup_epochs:
        return initial_lr * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
#------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------Training Functions-----------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, device, scaler, epoch, writer, args):
    model.train()
    epoch_losses = {'total': 0.0, 'recon': 0.0, 'kl_byte': 0.0, 'kl_segment': 0.0}

    beta_byte = beta_annealing(epoch, args.epochs, args.beta_byte, level='byte')
    beta_segment = beta_annealing(epoch, args.epochs, args.beta_segment, level='segment')
    lr = cosine_lr_schedule(epoch, args.lr, args.lr * 0.01, 10, args.epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    batch_count = 0
    accumulation_count = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training (MemOpt)")

    effective_batch_size = args.batch_size * args.accumulation_steps

    for batch_idx, batch in enumerate(progress_bar):
        if batch_idx % 10 == 0:
            aggressive_memory_cleanup()

        x = batch.to(device, non_blocking=True)
        x = preprocess_packet_data(x)

        # tensor shape
        print(f"Input tensor shape: {x.shape}")
        
        target_embedding_size = args.embedding_size  
        segment_size = args.segment_size
        
        if x.dim() == 2:  # (batch_size, features)
            batch_size = x.shape[0]
            total_features = x.shape[1]
            
            features_per_segment = segment_size * target_embedding_size
            num_segments = total_features // features_per_segment
            
            if num_segments == 0:
                num_segments = 1
                target_total_features = features_per_segment
                if total_features < target_total_features:
                    padding = torch.zeros(batch_size, target_total_features - total_features, device=device)
                    x = torch.cat([x, padding], dim=1)
                else:
                    x = x[:, :target_total_features]
            else:
                used_features = num_segments * features_per_segment
                x = x[:, :used_features]
            
            # Reshape to target format
            x = x.view(batch_size, num_segments, segment_size, target_embedding_size)
            
        elif x.dim() == 3:  # (batch_size, seq_len, original_features)
            batch_size, seq_len, original_features = x.shape
            
            if original_features != target_embedding_size:
                if original_features < target_embedding_size:
                    padding = torch.zeros(batch_size, seq_len, target_embedding_size - original_features, device=device)
                    x = torch.cat([x, padding], dim=2)
                else:
                    x = x[:, :, :target_embedding_size]
            
            num_segments = seq_len // segment_size
            
            if num_segments == 0:
                num_segments = 1
                target_seq_len = segment_size
                if seq_len < target_seq_len:
                    padding = torch.zeros(batch_size, target_seq_len - seq_len, target_embedding_size, device=device)
                    x = torch.cat([x, padding], dim=1)
                else:
                    x = x[:, :target_seq_len, :]
            else:
                used_seq_len = num_segments * segment_size
                x = x[:, :used_seq_len, :]
            
            x = x.view(batch_size, num_segments, segment_size, target_embedding_size)
            
        elif x.dim() == 4:  # (batch_size, num_segments, segment_size, original_embedding)
            batch_size, num_segments, current_segment_size, original_embedding = x.shape
            
            if original_embedding != target_embedding_size:
                if original_embedding < target_embedding_size:
                    padding = torch.zeros(batch_size, num_segments, current_segment_size, 
                                        target_embedding_size - original_embedding, device=device)
                    x = torch.cat([x, padding], dim=3)
                else:
                    x = x[:, :, :, :target_embedding_size]
            
            if current_segment_size != segment_size:
                if current_segment_size < segment_size:
                    padding = torch.zeros(batch_size, num_segments, 
                                        segment_size - current_segment_size, target_embedding_size, device=device)
                    x = torch.cat([x, padding], dim=2)
                else:
                    x = x[:, :, :segment_size, :]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        print(f"Reshaped tensor shape: {x.shape}")
        
        expected_shape = (x.shape[0], x.shape[1], segment_size, target_embedding_size)
        if x.shape[2:] != (segment_size, target_embedding_size):
            print(f"Warning: Final shape {x.shape} doesn't match expected {expected_shape}")

        with torch.amp.autocast(device_type=device.type, enabled=args.use_amp):
            outputs = model(x)

            recon_loss = outputs['recon_loss']
            byte_kl_loss = outputs['byte_kl_loss']
            segment_kl_loss = outputs['segment_kl_loss']

            weighted_kl_byte = beta_byte * byte_kl_loss
            weighted_kl_segment = beta_segment * segment_kl_loss

            total_loss = (recon_loss + weighted_kl_byte + weighted_kl_segment) / args.accumulation_steps

        if args.use_amp:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        epoch_losses['total'] += total_loss.item() * args.accumulation_steps
        epoch_losses['recon'] += recon_loss.item()
        epoch_losses['kl_byte'] += weighted_kl_byte.item()
        epoch_losses['kl_segment'] += weighted_kl_segment.item()

        accumulation_count += 1

        if accumulation_count >= args.accumulation_steps:
            if args.use_amp:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            accumulation_count = 0
            batch_count += 1

        if batch_idx % 5 == 0:
            avg_total = epoch_losses['total'] / max(batch_count, 1)
            avg_recon = epoch_losses['recon'] / max(batch_idx + 1, 1)
            current_num_segments = x.shape[1] if x.dim() == 4 else "N/A"

            progress_bar.set_postfix({
                'loss': f"{avg_total:.4f}",
                'recon': f"{avg_recon:.4f}",
                'β_b': f"{beta_byte:.3f}",
                'β_s': f"{beta_segment:.3f}",
                'lr': f"{lr:.1e}",
                'eff_bs': effective_batch_size,
                'segs': current_num_segments,
                'emb': target_embedding_size
            })

        del x, outputs, recon_loss, byte_kl_loss, segment_kl_loss

        if batch_count >= args.max_batches // args.accumulation_steps:
            break

    if accumulation_count > 0:
        if args.use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    total_samples = max(batch_count, 1)
    for key in epoch_losses:
        epoch_losses[key] /= total_samples

    return epoch_losses, beta_byte, beta_segment, lr
#------------------------------------------------------------------------------------------------------------------------
def validate_memory_efficient(model, dataloader, device, beta_byte, beta_segment, writer, epoch, max_val_batches=50):
    """Memory-efficient validation function adapted for NVAE with adaptive embedding size"""
    model.eval()
    val_losses = {'total': 0.0, 'recon': 0.0, 'kl_byte': 0.0, 'kl_segment': 0.0}

    batch_count = 0
    progress_bar = tqdm(dataloader, desc="Validation (MemOpt)")

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            if batch_idx >= max_val_batches:
                break

            x = batch.to(device, non_blocking=True)
            x = preprocess_packet_data(x)

            target_embedding_size = args.embedding_size
            segment_size = args.segment_size
            
            if x.dim() == 2:  # (batch_size, features)
                batch_size = x.shape[0]
                total_features = x.shape[1]
                
                features_per_segment = segment_size * target_embedding_size
                num_segments = total_features // features_per_segment
                
                if num_segments == 0:
                    num_segments = 1
                    target_total_features = features_per_segment
                    if total_features < target_total_features:
                        padding = torch.zeros(batch_size, target_total_features - total_features, device=device)
                        x = torch.cat([x, padding], dim=1)
                    else:
                        x = x[:, :target_total_features]
                else:
                    used_features = num_segments * features_per_segment
                    x = x[:, :used_features]
                
                x = x.view(batch_size, num_segments, segment_size, target_embedding_size)
                
            elif x.dim() == 3:  # (batch_size, seq_len, original_features)
                batch_size, seq_len, original_features = x.shape
                
                if original_features != target_embedding_size:
                    if original_features < target_embedding_size:
                        padding = torch.zeros(batch_size, seq_len, target_embedding_size - original_features, device=device)
                        x = torch.cat([x, padding], dim=2)
                    else:
                        x = x[:, :, :target_embedding_size]
                
                num_segments = seq_len // segment_size
                
                if num_segments == 0:
                    num_segments = 1
                    target_seq_len = segment_size
                    if seq_len < target_seq_len:
                        padding = torch.zeros(batch_size, target_seq_len - seq_len, target_embedding_size, device=device)
                        x = torch.cat([x, padding], dim=1)
                    else:
                        x = x[:, :target_seq_len, :]
                else:
                    used_seq_len = num_segments * segment_size
                    x = x[:, :used_seq_len, :]
                
                x = x.view(batch_size, num_segments, segment_size, target_embedding_size)
                
            elif x.dim() == 4:  
                batch_size, num_segments, current_segment_size, original_embedding = x.shape
                
                if original_embedding != target_embedding_size:
                    if original_embedding < target_embedding_size:
                        padding = torch.zeros(batch_size, num_segments, current_segment_size, 
                                            target_embedding_size - original_embedding, device=device)
                        x = torch.cat([x, padding], dim=3)
                    else:
                        x = x[:, :, :, :target_embedding_size]
                
                if current_segment_size != segment_size:
                    if current_segment_size < segment_size:
                        padding = torch.zeros(batch_size, num_segments, 
                                            segment_size - current_segment_size, target_embedding_size, device=device)
                        x = torch.cat([x, padding], dim=2)
                    else:
                        x = x[:, :, :segment_size, :]

            with torch.amp.autocast(device_type=device.type, enabled=args.use_amp):
                outputs = model(x, mode='train')

                recon_loss = outputs['recon_loss']
                byte_kl_loss = outputs['byte_kl_loss']
                segment_kl_loss = outputs['segment_kl_loss']

                weighted_kl_byte = beta_byte * byte_kl_loss
                weighted_kl_segment = beta_segment * segment_kl_loss
                total_loss = recon_loss + weighted_kl_byte + weighted_kl_segment
                
            val_losses['total'] += total_loss.item()
            val_losses['recon'] += recon_loss.item()
            val_losses['kl_byte'] += weighted_kl_byte.item()
            val_losses['kl_segment'] += weighted_kl_segment.item()

            batch_count += 1

            del x, outputs, recon_loss, byte_kl_loss, segment_kl_loss

            if batch_idx % 10 == 9:
                aggressive_memory_cleanup()

    for key in val_losses:
        val_losses[key] /= max(batch_count, 1)

    return val_losses
#------------------------------------------------------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scaler, epoch, losses, args, filename=None, extra_info=None):
    """Enhanced checkpoint saving with memory optimization"""
    if filename is None:
        filename = f"hierarchical_vae_epoch_{epoch+1}.pt"
    
    checkpoint_path = os.path.join(args.output_dir, filename)
    
    # Move model to CPU for saving to reduce GPU memory usage
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_cpu,
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'args': vars(args),
        'model_params': sum(p.numel() for p in model.parameters()),
        'timestamp': time.time()
    }
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    if extra_info:
        checkpoint.update(extra_info)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")
    
    # Clean up CPU tensors
    del model_cpu
    aggressive_memory_cleanup()

#------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------Main Training Function-------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------

def main():
    print("======  HierarchicalVAE Training ======")
    
    # Set memory efficient settings
    if args.memory_efficient:
        set_memory_efficient_settings()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    check_memory_usage("Initial")

    # Initialize logging
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    if args.use_wandb:
        
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config={
                "architecture": "HierarchicalVAE_MemoryOptimized",
                "dataset": args.pcap_file,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "effective_batch_size": args.batch_size * args.accumulation_steps,
                "learning_rate": args.lr,
                "beta_byte": args.beta_byte,
                "beta_segment": args.beta_segment,
                "segment_size": args.segment_size,
                "use_amp": args.use_amp,
                "use_checkpointing": args.use_checkpointing,
                "accumulation_steps": args.accumulation_steps,
                "memory_efficient": args.memory_efficient
            }
        )

    # Dataset setup 
    pcap_path = os.path.join(args.data_dir, args.pcap_file)
    print(f"PCAP file: {pcap_path}")

    # Reduced dataset estimation for memory efficiency
    print("Estimating dataset size (reduced for memory efficiency)...")
    
    # Model setup with reduced parameters
    model = nvae(
        embedding_size =args.embedding_size,
        num_residual_levels=args.num_residual_levels,
        free_bits=args.free_bits
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    check_memory_usage("After model creation")
    
    
    optimizer = optim.Adamax(  
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.999), 
        weight_decay=1e-4,
        eps=1e-8
    )
    
    scaler = torch.amp.GradScaler() if args.use_amp else None
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    
    # Training loop with memory optimization
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        epoch_start_time = time.time()
        
        # Memory check before epoch
        check_memory_usage(f"Before epoch {epoch+1}")
        
        class MemoryEfficientDataset(PCAPTCPFlowDataset):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
            def __iter__(self):
                count = 0
                for batch in super().__iter__():
                    yield batch
                    count += 1
                    if count >= args.max_batches:
                        break
        
        train_dataset = MemoryEfficientDataset(
            pcap_file=pcap_path,
            batch_size=args.batch_size,
            segment_size=args.segment_size,
            mode='segment',
            bytes_per_unit=1
        )
        
        val_dataset = MemoryEfficientDataset(
            pcap_file=pcap_path,
            batch_size=args.batch_size,
            segment_size=args.segment_size,
            mode='segment',
            bytes_per_unit=1
        )
        
        # Training phase with memory optimization
        train_losses, beta_byte, beta_segment, lr = train_epoch(
            model, train_dataset, optimizer, device, scaler, epoch, writer, args
        )
        
        check_memory_usage(f"After training epoch {epoch+1}")
        
        # Validation phase with memory optimization
        val_losses = validate_memory_efficient(
            model, val_dataset, device, beta_byte, beta_segment, writer, epoch
        )
        
        # Log metrics
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/total_loss": train_losses['total'],
                "train/recon_loss": train_losses['recon'],
                "train/kl_byte_loss": train_losses['kl_byte'],
                "train/kl_segment_loss": train_losses['kl_segment'],
                "val/total_loss": val_losses['total'],
                "val/recon_loss": val_losses['recon'],
                "val/kl_byte_loss": val_losses['kl_byte'],
                "val/kl_segment_loss": val_losses['kl_segment'],
                "beta_byte": beta_byte,
                "beta_segment": beta_segment,
                "learning_rate": lr
            })
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Training   - Total: {train_losses['total']:.4f}, Recon: {train_losses['recon']:.4f}")
        print(f"  Validation - Total: {val_losses['total']:.4f}, Recon: {val_losses['recon']:.4f}")
        print(f"  β_byte: {beta_byte:.4f}, β_segment: {beta_segment:.4f}, LR: {lr:.2e}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            save_checkpoint(
                model, optimizer, scaler, epoch, val_losses, args,
                filename='hierarchical_vae'
            )
            print(f"New best validation loss: {best_val_loss:.4f}")
        
        # Regular checkpoint saving (less frequent)
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, scaler, epoch, val_losses, args)
        
        # Early stopping check
        if early_stopping(val_losses['total'], model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # Memory cleanup between epochs
        del train_dataset, val_dataset
        aggressive_memory_cleanup()
        check_memory_usage(f"After epoch {epoch+1} cleanup")
    
    # Training completed
    print("\n=== Memory-Optimized Training Completed ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Final cleanup
    writer.close()
    if args.use_wandb:
        wandb.finish()
    
    print("Training completed successfully with memory optimization!")
#----------------------------------------------------------------------------------------------------------------------------------------------
   
if __name__ == "__main__":


    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if 'wandb' in globals():
            wandb.finish()
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        if 'wandb' in globals():
            wandb.finish()
        sys.exit(1)