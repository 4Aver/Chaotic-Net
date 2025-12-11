import torch
import torch.nn as nn
import torch.nn.functional as F
from 研究生课题.前沿研究.Chaotic_Net.对比实验.Layer.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from 研究生课题.前沿研究.Chaotic_Net.对比实验.Layer.SelfAttention_Family import FullAttention, AttentionLayer
from 研究生课题.前沿研究.Chaotic_Net.对比实验.Layer.Embed import DataEmbedding
import numpy as np


class Transformer_Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, seq_len, pred_len, enc_in):
        super(Transformer_Model, self).__init__()
        self.pred_len = pred_len
        e_layers = 3
        n_heads = 8
        d_model = 512
        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False), d_model, n_heads),
                    d_model) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.dense_feature = nn.Linear(d_model, 1)

    def forecast(self, x_enc):
        # Embedding
        x_enc = self.enc_embedding(x_enc, None)      # torch.Size([128, 50, 32])
        enc_out, attns = self.encoder(x_enc, attn_mask=None)  # torch.Size([128, 50, 32])

        enc_out = self.dense_feature(enc_out)
        return enc_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :].squeeze(-1)  # [B, L, D]



if __name__ == "__main__":
    x = torch.randn((128,32,15))        # [B, L, D]
    model = Transformer_Model(seq_len=32,pred_len=1,enc_in=15)
    out = model(x)
    print(out.shape)



