# ## Dynamic with dropout
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Add(nn.Module):
#     def __init__(self, epsilon=1e-12):
#         super(Add, self).__init__()
#         self.epsilon = epsilon
#         self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.w_relu = nn.ReLU()

#     def forward(self, x):
#         w = self.w_relu(self.w)
#         weight = w / (torch.sum(w, dim=0) + self.epsilon)
#         return weight[0] * x[0] + weight[1] * x[1]

# class Embedding(nn.Module):
#     def __init__(self, d_in, d_out, seq_len, dropout_prob=0.1):
#         super(Embedding, self).__init__()
#         self.convs = nn.ModuleList([
#             nn.Conv1d(d_in, d_out // 4, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.Conv1d(d_in, d_out // 4, kernel_size=5, stride=1, padding=2, bias=False),
#             nn.Conv1d(d_in, d_out // 4, kernel_size=7, stride=1, padding=3, bias=False),
#             nn.Conv1d(d_in, d_out // 4, kernel_size=9, stride=1, padding=4, bias=False)
#         ])
#         self.act_bn = nn.Sequential(
#             nn.BatchNorm1d(d_out),
#             nn.GELU()
#         )
#         self.dropout = nn.Dropout(dropout_prob)

#     def forward(self, x):
#         signals = []
#         for conv in self.convs:
#             signals.append(conv(x))
#         return self.dropout(self.act_bn(torch.cat(signals, dim=1)))

# class projector(nn.Module):
#     def __init__(self, heads, dim, dropout_prob=0.1):
#         super(projector, self).__init__()
#         self.q_k_v = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(dim, dim, 3, stride=1 if i == 0 else 2, padding=1, groups=dim, bias=False),
#                 nn.BatchNorm2d(dim),
#                 nn.Conv2d(dim, dim, 1, 1, 0),
#                 nn.GELU()
#             )
#             for i in range(3)
#         ])
#         self.MHSA = MHSA(dim, heads)
#         self.add = Add()
#         self.bn = nn.BatchNorm1d(dim)
#         self.dropout = nn.Dropout(dropout_prob)

#     def forward(self, x):
#         b, c, l = x.size()
#         h = int(l ** 0.5)
#         w = l // h
#         if h * w != l:
#             h += 1
#             padding = h * w - l
#             x = F.pad(x, (0, padding))
#         maps = x.view(b, c, h, w)
#         MHSA = self.MHSA(
#             self.q_k_v[0](maps),
#             self.q_k_v[1](maps),
#             self.q_k_v[2](maps)
#         )
#         MHSA = MHSA.view(b, c, -1)
#         return self.bn(self.add([self.dropout(MHSA), x]))

# class MHSA(nn.Module):
#     def __init__(self, emb_dim, heads):
#         super(MHSA, self).__init__()
#         self.dim = emb_dim
#         self.heads = heads

#     def forward(self, q, k, v):
#         q = torch.flatten(q, 2).transpose(1, 2)
#         k = torch.flatten(k, 2).transpose(1, 2)
#         v = torch.flatten(v, 2).transpose(1, 2)
#         if self.heads == 1:
#             q, k = F.softmax(q, dim=2), F.softmax(k, dim=1)
#             return q.bmm(k.transpose(2, 1)).bmm(v).transpose(1, 2)
#         else:
#             q = q.split(self.dim // self.heads, dim=2)
#             k = k.split(self.dim // self.heads, dim=2)
#             v = v.split(self.dim // self.heads, dim=2)
#             atts = []
#             for i in range(self.heads):
#                 att = F.softmax(q[i], dim=2).bmm(
#                     F.softmax(k[i], dim=1).transpose(2, 1).bmm(v[i])
#                 )
#                 atts.append(att.transpose(1, 2))
#             return torch.cat(atts, dim=1)

# class FFN(nn.Module):
#     def __init__(self, dim, ratio=4, dropout_prob=0.1):
#         super(FFN, self).__init__()
#         self.MLP = nn.Sequential(
#             nn.Linear(dim, dim // ratio),
#             nn.GELU(),
#             nn.Dropout(dropout_prob),
#             nn.Linear(dim // ratio, dim),
#             nn.GELU(),
#             nn.Dropout(dropout_prob)
#         )
#         self.add = Add()
#         self.bn = nn.BatchNorm1d(dim)

#     def forward(self, x):
#         feature = self.MLP(x.transpose(1, 2))
#         return self.bn(self.add([feature.transpose(1, 2), x]))

# class CLFormer_block(nn.Module):
#     def __init__(self, d_in, d_out, seq_len, heads=1, blocks=1, dropout_prob=0.1):
#         super(CLFormer_block, self).__init__()
#         self.embed = Embedding(d_in, d_out, seq_len, dropout_prob)
#         self.block = nn.Sequential()
#         for i in range(blocks):
#             self.block.add_module(
#                 "block_" + str(i),
#                 nn.Sequential(projector(heads, d_out, dropout_prob), FFN(d_out, dropout_prob=dropout_prob))
#             )

#     def forward(self, x):
#         x = self.embed(x)
#         return self.block(x)

# class CLFormer(nn.Module):
#     def __init__(self, d_out, seq_len, embed_dims=[4, 8, 16, 32], heads=1, blocks=1, ffn_ratio=4, dropout_prob=0.1):
#         super(CLFormer, self).__init__()
#         self.Encoder = nn.Sequential(
#             CLFormer_block(1, embed_dims[0], seq_len, heads, blocks, dropout_prob),
#             CLFormer_block(embed_dims[0], embed_dims[1], seq_len // 4, heads, blocks, dropout_prob),
#             CLFormer_block(embed_dims[1], embed_dims[2], seq_len // 16, heads, blocks, dropout_prob),
#             nn.AdaptiveAvgPool1d(1)
#         )
#         self.head_input = nn.Sequential(
#             nn.Linear(embed_dims[2], embed_dims[3]),
#             nn.BatchNorm1d(embed_dims[3]),
#             nn.GELU(),
#             nn.Dropout(dropout_prob)
#         )
#         self.head_output = nn.Linear(embed_dims[3], d_out)
#         self.zero_last_layer_weight()

#     def zero_last_layer_weight(self):
#         self.head_output.weight.data = torch.zeros_like(self.head_output.weight)
#         self.head_output.bias.data = torch.zeros_like(self.head_output.bias)

#     def forward(self, signal):
#         feature = self.Encoder(signal)
#         feature = self.head_input(feature.squeeze(2))
#         return self.head_output(feature)

# # 测试模型
# if __name__ == "__main__":
#     model = CLFormer(d_out=10, seq_len=96, embed_dims=[4, 8, 16, 32], heads=2, blocks=2, ffn_ratio=2, dropout_prob=0.1)
#     model.eval()
#     x = torch.randn(2, 1, 96)
#     output = model(x)
#     print(output)


import torch
import torch.nn as nn
import torch.nn.functional as F

class Add(nn.Module):
    def __init__(self, epsilon=1e-12):
        super(Add, self).__init__()
        self.epsilon = epsilon
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_relu = nn.ReLU()

    def forward(self, x):
        w = self.w_relu(self.w)
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return weight[0] * x[0] + weight[1] * x[1]

class Embedding(nn.Module):
    def __init__(self, d_in, d_out, seq_len, dropout_prob=0.1):
        super(Embedding, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d_in, d_out // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv1d(d_in, d_out // 4, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Conv1d(d_in, d_out // 4, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Conv1d(d_in, d_out // 4, kernel_size=9, stride=1, padding=4, bias=False)
        ])
        self.act_bn = nn.Sequential(
            nn.BatchNorm1d(d_out),
            nn.GELU()
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        signals = []
        for conv in self.convs:
            signals.append(conv(x))
        return self.dropout(self.act_bn(torch.cat(signals, dim=1)))

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

class projector(nn.Module):
    def __init__(self, heads, dim, dropout_prob=0.1):
        super(projector, self).__init__()
        self.q_k_v = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, stride=1 if i == 0 else 2, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, 1, 1, 0),
                nn.GELU()
            )
            for i in range(3)
        ])
        self.MHSA = MHSA(dim, heads)
        self.add = Add()
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        b, c, l = x.size()
        h = int(l ** 0.5)
        w = l // h
        if h * w != l:
            h += 1
            padding = h * w - l
            x = F.pad(x, (0, padding))
        maps = x.view(b, c, h, w)
        MHSA = self.MHSA(
            self.q_k_v[0](maps),
            self.q_k_v[1](maps),
            self.q_k_v[2](maps)
        )
        MHSA = MHSA.view(b, c, -1)
        return self.bn(self.add([self.dropout(MHSA), x]))

class FFN(nn.Module):
    def __init__(self, dim, ratio=4, dropout_prob=0.1):
        super(FFN, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(dim, dim // ratio),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(dim // ratio, dim),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )
        self.add = Add()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        feature = self.MLP(x.transpose(1, 2))
        return self.bn(self.add([feature.transpose(1, 2), x]))

class CLFormer_block(nn.Module):
    def __init__(self, d_in, d_out, seq_len, heads=1, blocks=1, dropout_prob=0.1):
        super(CLFormer_block, self).__init__()
        self.embed = Embedding(d_in, d_out, seq_len, dropout_prob)
        self.block = nn.Sequential()
        for i in range(blocks):
            self.block.add_module(
                "block_" + str(i),
                nn.Sequential(projector(heads, d_out, dropout_prob), FFN(d_out, dropout_prob=dropout_prob))
            )

    def forward(self, x):
        x = self.embed(x)
        return self.block(x)

class CLFormer(nn.Module):
    def __init__(self, d_out, seq_len, d_in=4, embed_dims=[4, 8, 16, 32], heads=1, blocks=1, ffn_ratio=4, dropout_prob=0.1):
        super(CLFormer, self).__init__()
        self.Encoder = nn.Sequential(
            CLFormer_block(d_in, embed_dims[0], seq_len, heads, blocks, dropout_prob),
            CLFormer_block(embed_dims[0], embed_dims[1], seq_len // 4, heads, blocks, dropout_prob),
            CLFormer_block(embed_dims[1], embed_dims[2], seq_len // 16, heads, blocks, dropout_prob)
        )
        self.head_input = nn.Sequential(
            nn.Linear(embed_dims[2], embed_dims[3]),
            nn.BatchNorm1d(embed_dims[3]),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )
        self.head_output = nn.Linear(embed_dims[3], d_out)
        self.final_layer = nn.Linear(embed_dims[3], seq_len)  # 新增的线性层，使输出与输入序列长度一致
        self.zero_last_layer_weight()

    def zero_last_layer_weight(self):
        self.head_output.weight.data = torch.zeros_like(self.head_output.weight)
        self.head_output.bias.data = torch.zeros_like(self.head_output.bias)

    def forward(self, signal):
        feature = self.Encoder(signal)
        feature = self.head_input(feature.mean(dim=-1))  # 使用mean获取平均值，避免shape不一致
        feature = self.final_layer(feature)  # 调整后的输出层
        return feature

# 测试模型
if __name__ == "__main__":
    input_dim = 4  # 设置输入维度
    seq_len = 96  # 设置序列长度
    model = CLFormer(d_out=10, seq_len=seq_len, d_in=input_dim, embed_dims=[4, 8, 16, 32], heads=2, blocks=2, ffn_ratio=2, dropout_prob=0.1)
    model.eval()
    x = torch.randn(2, input_dim, seq_len)  # 输入形状调整为(2, input_dim, seq_len)
    output = model(x)
    print(output.shape)  # 输出与输入序列等长的序列，表示每个位置的概率


