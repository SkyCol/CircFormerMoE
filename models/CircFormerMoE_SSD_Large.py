import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_channels, num_blocks=18, base_channels=32):
        super(ResNet, self).__init__()
        self.in_channels = base_channels
        self.conv1 = nn.Conv1d(input_channels, base_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU()

        self.residual_layers = self.make_layers(base_channels, num_blocks)

    def make_layers(self, base_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(self.in_channels, base_channels))
            self.in_channels = base_channels 
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, sequence_length)
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.residual_layers(out)
        return out


from performer_pytorch import SelfAttention

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class PerformerAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(
            dim=dim,
            heads=heads,
            dim_head=dim // heads,
            dropout=dropout,
            causal=False
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, hidden_dim=dim * 4, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x



# class LinearAttention(nn.Module):
#     def __init__(self, dim, heads=8, dropout=0.1):
#         super().__init__()
#         self.dim = dim
#         self.heads = heads
#         self.head_dim = dim // heads
#         assert dim % heads == 0, "dim must be divisible by heads"

#         self.to_q = nn.Linear(dim, dim)
#         self.to_k = nn.Linear(dim, dim)
#         self.to_v = nn.Linear(dim, dim)

#         self.dropout = nn.Dropout(dropout)
#         self.out_proj = nn.Linear(dim, dim)

#     def forward(self, x):
#         b, n, d = x.shape
#         h = self.heads

#         q = self.to_q(x).view(b, n, h, self.head_dim)  # (b, n, h, hd)
#         k = self.to_k(x).view(b, n, h, self.head_dim)
#         v = self.to_v(x).view(b, n, h, self.head_dim)

#         q = F.elu(q) + 1
#         k = F.elu(k) + 1

#         kv = torch.einsum('bnhd,bnhv->bhdv', k, v)

#         z = 1 / (torch.einsum('bnhd,bhd->bnh', q, k.sum(dim=1)) + 1e-6)
#         z = z.unsqueeze(-1)

#         out = torch.einsum('bnhd,bhdv->bnhv', q, kv)
#         out = out * z

#         out = out.contiguous().view(b, n, d)
#         out = self.dropout(self.out_proj(out))
#         return out


# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout=0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout),
#         )
#     def forward(self, x):
#         return self.net(x)


# class LinearAttentionBlock(nn.Module):
#     def __init__(self, dim, heads=8, ff_hidden_dim=None, dropout=0.1):
#         super().__init__()
#         if ff_hidden_dim is None:
#             ff_hidden_dim = dim * 4

#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = LinearAttention(dim, heads, dropout)
#         self.norm2 = nn.LayerNorm(dim)
#         self.ff = FeedForward(dim, ff_hidden_dim, dropout)

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x))  # residual connection
#         x = x + self.ff(self.norm2(x))
#         return x



class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        device = x.device

        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / self.d_model))

        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return x + pe.unsqueeze(0)

class CLFormer(nn.Module):
    def __init__(self, d_out, d_in=64, heads=8, block_count=3, dropout_prob=0.1):
        super().__init__()
        self.pos_encoder = SinusoidalPositionalEncoding(d_in)
        self.layers = nn.ModuleList([
            PerformerAttentionBlock(d_in, heads, dropout=dropout_prob)
            for _ in range(block_count)
        ])
        self.final_layer = nn.Conv1d(d_in, d_out, kernel_size=1)

    def forward(self, x):
        x = self.pos_encoder(x)         # (batch, seq_len, d_model)
        for layer in self.layers:
            x = layer(x)                # attention + FF
        x = x.permute(0, 2, 1)          # (batch, d_model, seq_len)
        x = self.final_layer(x)
        return x


class CircFormer(nn.Module):
    def __init__(self, input_channels, num_classes, num_species,
                 num_res_blocks=18, heads=8, base_channels=64, num_layers=3):
        super(CircFormer, self).__init__()
        
        self.resnet = ResNet(input_channels, num_res_blocks, base_channels)
        self.clformer = CLFormer(base_channels, base_channels, heads, num_layers)

        self.species_heads = nn.ModuleList([
            nn.Conv1d(base_channels, num_classes, kernel_size=1) 
            for _ in range(num_species)
        ])

    def forward(self, x, species_ids=None):
        x = self.resnet(x)              # (batch, base_channels, seq_len)
        x = x.permute(0, 2, 1)         # (batch, seq_len, base_channels)
        x = self.clformer(x)            # (batch, num_classes, seq_len)

        splicing_out = torch.zeros(x.size(0), 1, x.size(2), device=x.device)
        for i in range(len(self.species_heads)):
            mask = (species_ids == i)
            if mask.any():
                splicing_out[mask] = self.species_heads[i](x[mask])

        return splicing_out.squeeze(1)


batch_size = 32
seq_len = 5001
input_channels = 4
num_classes = 1
num_species = 10
base_channels = 128
num_layers = 4

model = CircFormer(input_channels, num_classes, num_species, num_res_blocks=32, heads=8, base_channels=base_channels, num_layers=num_layers)

x = torch.randn(batch_size, seq_len, input_channels)
species_id = torch.randint(0, num_species, (batch_size,))  # e.g., species index for each sample

splicing_out = model(x, species_id)

print("Splicing output shape:", splicing_out.shape)   # → [32, 5001]



