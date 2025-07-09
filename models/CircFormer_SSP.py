import torch
import torch.nn as nn
import torch.nn.functional as F

## CircFormer_basic is a sequence-to-sequence model while 
## ChirFormer_Matcher is a sequence-to-class model with the same main architecture

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

# CLFormer
class MHSA(nn.Module):
    def __init__(self, emb_dim, heads):
        super(MHSA, self).__init__()
        self.dim = emb_dim
        self.heads = heads

    def forward(self, q, k, v):
        q = torch.flatten(q, 2).transpose(1, 2)
        k = torch.flatten(k, 2).transpose(1, 2)
        v = torch.flatten(v, 2).transpose(1, 2)
        if self.heads == 1:
            q, k = F.softmax(q, dim=2), F.softmax(k, dim=1)
            return q.bmm(k.transpose(2, 1)).bmm(v).transpose(1, 2)
        else:
            q = q.split(self.dim // self.heads, dim=2)
            k = k.split(self.dim // self.heads, dim=2)
            v = v.split(self.dim // self.heads, dim=2)
            atts = []
            for i in range(self.heads):
                att = F.softmax(q[i], dim=2).bmm(
                    F.softmax(k[i], dim=1).transpose(2, 1).bmm(v[i])
                )
                atts.append(att.transpose(1, 2))
            return torch.cat(atts, dim=1)


class CLFormer(nn.Module): ## CLFormer-like
    def __init__(self, d_out, d_in=32, heads=4, block_count=3, dropout_prob=0.1):
        super(CLFormer, self).__init__()
        self.mhsa_layers = nn.ModuleList([
            MHSA( d_in, heads) for _ in range(block_count)
        ])
        self.fc_layers = nn.ModuleList([
            nn.Sequential( ## FFN-like
                nn.Linear(d_in,  d_in),  
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(d_in,  d_in), 
                nn.GELU()
            ) for _ in range(block_count)
        ])
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.head_input = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.BatchNorm1d(d_in),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )

        self.final_layer = nn.Linear(d_in, d_out) 


    def forward(self, x):
        for mhsa, fc in zip(self.mhsa_layers, self.fc_layers):
            x = mhsa(x, x, x)  # Apply MHSA
            x = x.permute(0, 2, 1)  # (batch_size, seq_len, features)
            x = fc(x)    # Apply linear layer
            x = x.permute(0, 2, 1)  
        x = self.pooling(x)
        x = x.squeeze(-1) 
        # x = x.permute(0, 2, 1)  
        # print(x.shape)
        x = self.head_input(x)
        x = self.final_layer(x)
        # print(x.shape) 
        return x


class CircFormer_SSP(nn.Module):
    def __init__(self, input_channels, num_classes,  num_res_blocks=18, heads=4, base_channels=32, num_layers=3):
        super(CircFormer_SSP, self).__init__()
        self.resnet = ResNet(input_channels, num_res_blocks, base_channels, )
        self.clformer = CLFormer(num_classes,  d_in=base_channels,  heads=heads, block_count=num_layers)

    def forward(self, x):
        resnet_out = self.resnet(x)
        output = self.clformer(resnet_out)
        output = output.squeeze(1)  # Ensure output shape is correct
        return output


# input_channels = 4 
# num_classes = 3    
# seq_len = 801     
# base_channels = 64 
# num_layers = 4     

# model = CircFormer_DAR(input_channels, num_classes, num_res_blocks=16, heads=8, base_channels=base_channels, num_layers=num_layers)

# x = torch.randn(32, seq_len, input_channels)  # (batch_size, sequence_length, input_channels)

# output = model(x)
# print(output.shape)