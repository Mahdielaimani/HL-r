import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 1
class MultiHeadSelfAttentionConv(nn.Module):
    """Multi-head self-attention block."""
    def __init__(self, in_channels, num_heads=4):
        super(MultiHeadSelfAttentionConv, self).__init__()
        self.num_heads = num_heads
        self.query = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, length = x.size()
        
        q = self.query(x).view(batch_size, self.num_heads, channels // self.num_heads, -1)
        k = self.key(x).view(batch_size, self.num_heads, channels // self.num_heads, -1)
        v = self.value(x).view(batch_size, self.num_heads, channels // self.num_heads, -1)

        attn_scores = torch.matmul(q.transpose(-2, -1), k) / ((channels // self.num_heads) ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        v = v.view(batch_size, self.num_heads, -1, channels // self.num_heads)
        out = torch.matmul(attn_weights, v)
        out = out.view(batch_size, channels, length)
        return out


# class MultiHeadSelfAttentionConv(nn.Module):
#     """Multi-head self-attention with 1-D convolutions that keeps length."""
#     def __init__(self, in_channels: int, num_heads: int = 4):
#         super().__init__()
#         assert in_channels % num_heads == 0,\
#             "in_channels must be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim  = in_channels // num_heads

#         self.query = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
#         self.key   = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
#         self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, C, L = x.shape               # (batch, channels, length)
#         H, D    = self.num_heads, self.head_dim

#         # ---- shape to (B, H, L, D) -----------------------
#         q = self.query(x).reshape(B, H, D, L).transpose(-2, -1)   # (B,H,L,D)
#         k = self.key  (x).reshape(B, H, D, L)                     # (B,H,D,L)
#         v = self.value(x).reshape(B, H, D, L).transpose(-2, -1)   # (B,H,L,D)

#         # ---- scaled dot-product attention ----------------
#         scores = torch.matmul(q, k) / math.sqrt(D)                # (B,H,L,L)
#         attn   = torch.softmax(scores, dim=-1)

#         out = torch.matmul(attn, v)                               # (B,H,L,D)

#         # ---- back to (B, C, L) ----------------------------
#         out = out.transpose(-2, -1).contiguous().view(B, C, L)
#         return out

# -------------------------- 2
class MultiHeadTargetAttention(nn.Module):
    """Multi-head target attention block."""
    def __init__(self, in_channels, num_heads=4):
        super(MultiHeadTargetAttention, self).__init__()
        self.num_heads = num_heads
        self.target = nn.Parameter(torch.randn(1, in_channels, 1))
        self.key = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, length = x.size()
        t = self.target.expand(batch_size, -1, length)

        head_dim = channels // self.num_heads
        if channels % self.num_heads != 0:
            raise ValueError(f"Channels ({channels}) must be divisible by num_heads ({self.num_heads}).")

        k = self.key(x).view(batch_size, self.num_heads, head_dim, length)  
        v = self.value(x).view(batch_size, self.num_heads, head_dim, length)  

        attn_scores = torch.matmul(k.transpose(-2, -1), v) / (head_dim ** 0.5)  
        attn_weights = torch.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, v.transpose(-2, -1))  
        out = out.view(batch_size, channels, length)  
        return out

# -------------------------- 3
class VAE(nn.Module):
    """Variational Autoencoder (VAE) for network packets."""
    def __init__(self, input_dim=128, latent_dim=128, num_heads=4):
        super().__init__()

        # Encoder 
        self.word_embedding = nn.Embedding(257, latent_dim)

        self.multiheadAttention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.multihead_target_attention = MultiHeadTargetAttention(in_channels=latent_dim, num_heads=num_heads)
        self.conv1d_elu = nn.Conv1d(latent_dim, latent_dim, kernel_size=3) 
        self.conv1d_tanh = nn.Conv1d(latent_dim, latent_dim, kernel_size=3)
        self.conv1d_target = nn.Conv1d(latent_dim, latent_dim, kernel_size=3)
        
        self.conv1d_elu_s = nn.Conv1d(latent_dim, latent_dim, kernel_size=3) 
        self.conv1d_tanh_s = nn.Conv1d(latent_dim, latent_dim, kernel_size=3)
        self.conv1d_target_s = nn.Conv1d(latent_dim, latent_dim, kernel_size=3)

        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, latent_dim)

        # Sentence layers
        self.sentence_embedding = nn.Linear(latent_dim, latent_dim)
        self.sentence_positional_encoding = nn.Parameter(torch.zeros(1, 100, latent_dim))  
        self.sentence_multihead_elu = MultiHeadSelfAttentionConv(in_channels=latent_dim, num_heads=num_heads)
        self.sentence_multihead_tanh = MultiHeadSelfAttentionConv(in_channels=latent_dim, num_heads=num_heads)
        self.sentence_target_attention = MultiHeadTargetAttention(in_channels=latent_dim, num_heads=num_heads)

        self.document_embedding = nn.Linear(latent_dim, latent_dim)
        self.softmax = nn.Softmax(dim=-1)

        # Decoder
        self.decoder = nn.ModuleList([
        MultiHeadTargetAttention(in_channels=32, num_heads=num_heads),
        nn.ELU(),
        nn.ConvTranspose1d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=0),  
        nn.ELU(),
        nn.ConvTranspose1d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=0), 
        nn.ELU(),
        MultiHeadSelfAttentionConv(in_channels=128, num_heads=num_heads),
        nn.ELU(),
        nn.ConvTranspose1d(128, input_dim, kernel_size=3, stride=1, padding=1),  
        nn.Sigmoid()  
    ])
 # -------------------------------------------Positional Encoding-------------------------------------------
    def add_positional_encoding(self, x):
        seq_len, dim = x.shape[-2], x.shape[-1]
        pe = self.get_positional_encoding(seq_len, dim)
        return x + pe

    def get_positional_encoding(self, seq_len, dim):
        pe = torch.zeros(seq_len, dim).to(next(self.parameters()).device)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
# ------------------------------------------------------------------------------------------------   

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        print(f"Shape Input Encoder: {x.shape}")
        emb_input = self.word_embedding(x)
        print(f"Shape after word embedding: {x.shape}")

        emb_input = self.add_positional_encoding(emb_input)
        print(f"Shape after positional encoding: {emb_input.shape}")

        # -------------------------------------------Convolutional Multihead Self Attention (ELU)------------------------------------
        tmp = []
        for pkt in emb_input:
            pkt = pkt.transpose(1, 2)
            pkt = self.elu(self.conv1d_elu(pkt))
            pkt = pkt.transpose(1, 2)
            pkt, _ = self.multiheadAttention(pkt, pkt, pkt)
            tmp.append(pkt.unsqueeze(0))
        x = torch.cat(tmp, dim=0)
        att_elu = self.elu(x)
        # ----------------------------------------------------------------------------------------------------------------

        print(f"Shape after convolutional multihead self attention (ELU): {att_elu.shape}")

        # --------------------------------Convolutional Multihead Self Attention (Tanh)------------------------------------
        tmp = []
        for pkt in emb_input:
            pkt = pkt.transpose(1, 2)
            pkt = self.tanh(self.conv1d_tanh(pkt))
            pkt = pkt.transpose(1, 2)
            pkt, _ = self.multiheadAttention(pkt, pkt, pkt)
            tmp.append(pkt.unsqueeze(0))
        x = torch.cat(tmp, dim=0)
        att_tanh = self.tanh(x)
        # ------------------------------------------------------------------------------------------------

        print(f"Shape after convolutional multihead self attention (Tanh): {att_tanh.shape}")

        # --------------------------Layer Normalization, Elementwise Multiplication-----------------------
        x = att_elu * att_tanh
        x = nn.LayerNorm(x.shape[1:])(x)
        # ------------------------------------------------------------------------------------------------

        print(f"Shape after layer normalization: {x.shape}")

        # ---------------------------------Convolutional Multihead Target Attention------------------------------------
        tmp = []
        target = nn.Parameter(torch.randn(1, x.shape[-1]))
        stdv = 1. / math.sqrt(target.size(1))
        target.data.uniform_(-stdv, stdv).unsqueeze(0)
        target = target.repeat(x.shape[1], 1, 1)  

        for pkt in x:   
            pkt = pkt.transpose(1, 2)
            pkt = self.elu(self.conv1d_target(pkt))
            pkt = pkt.transpose(1, 2)
            pkt, _ = self.multiheadAttention(target, pkt, pkt)
            tmp.append(pkt.unsqueeze(0))
        x = torch.cat(tmp, dim=0)
        # ------------------------------------------------------------------------------------------------
        print(f"Shape after multihead target attention: {x.shape}")
        out = x.squeeze(-2)
        print(f"[Word Hierarchy Embedding] shape: {out.shape}")

        # ---------------------------------------------------------------------------
        # --------------------- END of Word Hierarchy Embedding ---------------------
        # ---------------------------------------------------------------------------

        # ---------------------------------------------------------------------------
        # -------------------- Sentence Hierarchy Embedding -------------------------  
        # ---------------------------------------------------------------------------
        
        sentence_emb = self.add_positional_encoding(out)
        print(f"Shape after adding sentence positional encoding: {sentence_emb.shape}")

        # --------------------------Convolutional Multihead Self-Attention (ELU)--------------------------
        print(f"Shape before transpose: {sentence_emb.shape}")
        sentence_emb = sentence_emb.transpose(1, 2)  
        sentence_emb = self.elu(self.conv1d_elu_s(sentence_emb))
        sentence_emb = sentence_emb.transpose(1, 2)  
        sentence_emb, _ = self.multiheadAttention(sentence_emb, sentence_emb, sentence_emb)
        att_elu = self.elu(sentence_emb)
        # ------------------------------------------------------------------------------------------------

        print(f"Shape after sentence multihead self-attention (ELU): {att_elu.shape}")


        # -------------------------Convolutional Multihead Self-Attention (Tanh)--------------------------
        sentence_emb = sentence_emb.transpose(1, 2)
        sentence_emb = self.tanh(self.conv1d_tanh_s(sentence_emb))
        sentence_emb = sentence_emb.transpose(1, 2)
        sentence_emb, _ = self.multiheadAttention(sentence_emb, sentence_emb, sentence_emb)
        att_tanh = self.tanh(sentence_emb)
        # ------------------------------------------------------------------------------------------------

        print(f"Shape after sentence multihead self-attention (Tanh): {att_tanh.shape}")

        # --------------------------Layer Normalization, Elementwise Multiplication-----------------------
        x = att_elu * att_tanh
        x = nn.LayerNorm(x.shape[1:])(x)
        # ------------------------------------------------------------------------------------------------

        print(f"Shape after layer normalization: {x.shape}")

        # -------------------------Convolutional Multihead Target Attention-------------------------------
        target = nn.Parameter(torch.randn(1, x.shape[-1]))
        stdv = 1. / math.sqrt(target.size(1))
        target.data.uniform_(-stdv, stdv).unsqueeze(0)
        target = target.repeat(x.shape[0], 1, 1)  

        sentence_emb = sentence_emb.transpose(1, 2)  
        sentence_emb = self.elu(self.conv1d_target_s(sentence_emb))
        sentence_emb = sentence_emb.transpose(1, 2) 

        sentence_emb, _ = self.multiheadAttention(target, x, x)  
        # ------------------------------------------------------------------------------------------------

        print(f"Shape after sentence target attention: {sentence_emb.shape}")

        doc_emb = sentence_emb.squeeze(-2)

        # --------------------- Document Embedding ---------------------
        # doc_emb = self.document_embedding(sentence_emb.mean(dim=1))  
        print(f"Shape after document embedding: {doc_emb.shape}")  

        # Softmax for final output
        output = self.softmax(doc_emb)
        print(f"Shape after softmax: {output.shape}")  

        # ---------------------------------------------------------------------------
        # --------------------- END of Sentence Hierarchy  --------------------------
        # ---------------------------------------------------------------------------

        mu, logvar = self.fc_mu(doc_emb), self.fc_logvar(doc_emb)
        z = self.reparameterize(mu, logvar)
        print(f"Shape of Latent vector: {z.shape}")
        x_reconstructed = self.decoder_input(z).view(z.shape[0], 32, -1)
        
        
        for layer in self.decoder:
            x_reconstructed = layer(x_reconstructed)
            if isinstance(layer, nn.ConvTranspose1d):
                x_reconstructed = nn.LayerNorm(x_reconstructed.shape[1:])(x_reconstructed)
        return x_reconstructed, mu, logvar
    
# -------------------------- 4
def vae_loss_function(recon_x, x, mu, logvar, epoch=1):
    """Computes VAE loss (Reconstruction Loss + KL Divergence)."""
    
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    if recon_x.shape != x.shape:
        output_size = recon_x.shape[2:]
        x = F.interpolate(x.unsqueeze(1), size=(recon_x.shape[1], *output_size), mode="nearest").squeeze(1)
    
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_weight = min(1.0, epoch / 50)

    return MSE + kl_weight * KLD
