import torch
import torch.nn as nn
import copy


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads, dropout_rate):  # 420， 768， 10， 0.4
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, mask=None):
        # x: (seq_len, batch_size, input_dim)
        # mask: (batch_size, seq_len)
        # print(x.shape)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + attn_output
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        return x


class Former(nn.Module):
    def __init__(self,  embedding_dim, hidden_dim, num_heads, n_layers, dropout_rate, *args, **kwargs):

        super(Former, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, hidden_dim, num_heads, dropout_rate)
            for _ in range(n_layers)  # 2层encoder
        ])

    def forward(self, x, mask=None):
        # x: (seq_len, batch_size, input_dim)
        # mask: (batch_size, seq_len)
        for layer in self.layers:
            x = layer(x, mask)
        # x = x.view(10, 64 * 420)
        # x = x.view(80640)
        x = x.mean(dim=1)
        return x
        # 10,64,420
        # print(x.shape)


"""
没改好的transformer
"""
# class LayerNorm(nn.Module):
#     # 针对一个样本做的均值归一化，，，BN是针对一个batch。。。层归一化，，，其实就是对输入进行norm
#     """
#     features表示词嵌入的维度。esp是一个足够小的数
#     """
#     def __init__(self, features, eps=1e-6):
#         super(LayerNorm, self).__init__()
#         self.a_2 = nn.Parameter(torch.ones(features))
#         self.b_2 = nn.Parameter(torch.zeros(features))
#         self.eps = eps
#
#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
#
#
# def clones(module, N):
#     # 复制多个layer返回ModuleList
#     return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
#
#
# class SublayerConnection(nn.Module):
#     """
#     残差连接
#     A residual connection followed by a layer norm.
#     Note for code simplicity the norm is first as opposed to last.
#     """
#
#     def __init__(self, size, dropout):
#         super(SublayerConnection, self).__init__()
#         self.norm = LayerNorm(size)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, sublayer):
#         return x + self.dropout(sublayer(self.norm(x)))
#     # 可调整
#
#
# class EncoderLayer(nn.Module):
#     def __init__(self, size, self_attn, feed_forward, dropout):
#         super(EncoderLayer, self).__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(size, dropout), 2)
#         self.size = size
#
#     def forward(self, x, mask):
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
#         return self.sublayer[1](x, self.feed_forward)
#
# class Encoder(nn.Module):
#     """
#     Encoder架构是将多层EncoderLayer连接
#     """
#
#     def __init__(self, layer, N):
#         super(Encoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.size)
#
#     def forward(self, x, mask):
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)