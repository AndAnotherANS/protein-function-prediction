import json
import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, embed_size, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        enc = torch.zeros(max_len, 1, embed_size)
        enc[:, 0, 0::2] = torch.sin(position * div_term)
        enc[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('enc', enc.squeeze())

    def forward(self, x):
        return self.enc[:x.shape[1]].unsqueeze(0).repeat(x.shape[0], 1, 1)


class Transformer(nn.Module):
    def __init__(self, vocab_len, embed_size, depth, n_special_tokens=2):
        super().__init__()
        self.vocab_len = vocab_len
        self.embedding = nn.Embedding(vocab_len + n_special_tokens, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)
        encoder_layer = nn.TransformerEncoderLayer(embed_size, 4, 256, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)

    def forward(self, x):
        emb = self.embedding(x) + self.pos_encoding(x)
        mask = x >= self.vocab_len
        result = self.encoder(emb, src_key_padding_mask=mask)
        return result[:, 0, :]
