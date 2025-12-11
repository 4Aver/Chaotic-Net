# wo2_MoEE

import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Union, Tuple
from torch.distributions.normal import Normal
from 研究生课题.前沿研究.Chaotic_Net.models.DAGCN import DAGCN, Mulit_DAGCN

torch.manual_seed(21)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用gpu


# 生成旋转矩阵
def precompute_freqs_cis(atten_dim: int, seq_len: int, theta: float = 10000.0):
    theta_i_e = (torch.arange(0, atten_dim, 2)[:(atten_dim // 2)].float() / atten_dim) * -1
    freqs = theta ** theta_i_e.to(device)
    position = torch.arange(seq_len).to(device)  # [0, 1, 2, 3, ..., seq_len]
    freqs = torch.outer(position, freqs).float().to(device)  # 求向量的外积,维度为[seq_len,atten_dim]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs).to(device)  # 将上一步的结果写成复数的形式,模是1幅角是freqs
    return freqs_cis.view(1, 1, seq_len, atten_dim // 2)


def apply_rope(q: torch.Tensor, k: torch.Tensor, rotate_vecs: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    作用: 将q,k向量分别与旋转向量相乘,得到旋转后的q,k向量q/k_rotated。然后进行点乘得到具有位置信息的attention分数
    输入: q->weight_q(input_vecs), k->weight_k(input_vecs), rotaed_vecs->旋转向量
    """

    # 计算过程q:[batch_size,atten_heads,seq_len,atten_dim]->q_complex:[b,a_h,s,a_d//2,2]->[b,a_h,s,a_d//2]->[b,a_h,s,a_d//2,2]
    q_complex = torch.view_as_complex(
        q.float().reshape(*q.shape[:-1], -1, 2))  # [batch_size,atten_heads,seq_len,atten_dim//2,2]
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))  # 将一个大小为n的向量两两组合形成复数来计算
    # 位置编码只和向量的序列位置还有向量本身有关，和batch以及注意力头无关，所以只用关注第二维和第四维

    q_rotated = torch.view_as_real(q_complex * rotate_vecs).flatten(
        3)  # 恢复成原来的样子，将第三维之后压平，也就是(atten_dim//2,2)->(atten_dim)
    k_rotated = torch.view_as_real(k_complex * rotate_vecs).flatten(3)
    return q_rotated.type_as(q), k_rotated.type_as(q)


class Channel_Embedding(nn.Module):
    def __init__(self, list_input_dim, out_channels=10, emb_kernel_size=3, emb_stride=1):
        super(Channel_Embedding, self).__init__()

        self.list_input_dim = list_input_dim
        self.list_conv = nn.ModuleList()
        for input_dim in list_input_dim:
            # 定义PwConv将不同的嵌入维数映射到相同的嵌入维数，并没有提取时间信息，某方面说提取到了特征之间的信息
            conv_layer = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=out_channels, kernel_size=emb_kernel_size,
                          stride=emb_stride),
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
                )
            self.list_conv.append(conv_layer)
        self.inst_norm = nn.InstanceNorm1d(out_channels)  # 实例化归一化层

    def forward(self, x):
        index = 0
        combine_features = torch.Tensor(0)
        for i, dim in enumerate(self.list_input_dim):
            input_x = x[:, index:index + dim, :]
            index += dim
            out = self.list_conv[i](input_x)

            combine_features = torch.cat((combine_features, out.unsqueeze(1)), dim=1)
        return combine_features


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm((30, d_in), eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        return x


class MultiAttention(nn.Module):
    def __init__(self, input_dim, num_hiddens, num_heads, dropout=0.1):
        super(MultiAttention, self).__init__()
        self.num_heads = num_heads

        self.w_q = nn.Linear(input_dim, num_hiddens)
        self.w_k = nn.Linear(input_dim, num_hiddens)
        self.w_v = nn.Linear(input_dim, num_hiddens)

        self.w_o = nn.Linear(num_hiddens, input_dim, bias=False)

        self.drop_out = nn.Dropout(dropout)

        self.freqs_cis = precompute_freqs_cis(atten_dim=input_dim // num_heads, seq_len=30)

    def transpose_qk_RoPE(self, x, num_heads):
        '''
        :param x: shape(B,查询数或者键值对数,num_hiddens)
        :return:  shape(B,num_heads,查询数或者键值对数,num_hiddens/num_heads)      # num_hiddens为num_heads的整数倍
        '''
        x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
        x = x.permute(0, 2, 1, 3)
        return x

    def transpose_qkv(self, x, num_heads):
        '''
        :param x: shape(B,查询数或者键值对数,num_hiddens)
        :return:  shape(B*num_heads,查询数或者键值对数,num_heads,num_hiddens/num_heads)      # num_hiddens为num_heads的整数倍
        '''
        x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(-1, x.shape[2], x.shape[3])

    def transpose_output(self, x, num_heads):
        x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)

    def forward(self, x):
        B, time_len, input_dim = x.shape
        q = self.transpose_qk_RoPE(self.w_q(x), self.num_heads)
        k = self.transpose_qk_RoPE(self.w_k(x), self.num_heads)  # B,heads,time_len,dim/heads
        xq, xk = apply_rope(q=q, k=k, rotate_vecs=self.freqs_cis)

        q = xq.reshape(-1, time_len, input_dim // self.num_heads)
        k = xk.reshape(-1, time_len, input_dim // self.num_heads)

        v = self.transpose_qkv(self.w_v(x), self.num_heads)

        d = x.shape[-1]
        scores = F.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d),dim=-1)

        output = torch.bmm(self.drop_out(scores), v)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.w_o(output_concat)


class wo2_MoEE_Model(nn.Module):
    def __init__(self, M, time_len, list_input_dims,d=64, num_layers=1):
        super(wo2_MoEE_Model, self).__init__()
        self.M = M
        self.num_layers = num_layers
        self.list_input_dims = list_input_dims

        self.embedding = Channel_Embedding(list_input_dim=self.list_input_dims, out_channels=d)
        self.ln = nn.LayerNorm(normalized_shape=(30, d))

        self.indenpendet_transformer = nn.ModuleList(
            [MultiAttention(input_dim=d, num_hiddens=d, num_heads=8).to(device) for _ in range(M)])
        self.indenpendet_FFN = nn.ModuleList([PositionwiseFeedForward(d, d).to(device) for _ in range(M)])

        self.dagcn = Mulit_DAGCN(num_time_steps=30, num_nodes=M, in_dims=d, out_dims=d, cheb_k=3, embed_dim=2)


        self.FFN = PositionwiseFeedForward(d, d)

        self.linear_time = nn.Linear(30, 1)
        self.linear_feat = nn.Linear(M * d, 1)

    def forward(self, in_x):
        B, _, _ = in_x.shape
        x = in_x.permute(0, 2, 1)
        x = self.embedding(x)
        x = x.transpose(2, 3)  # B,M,L,D

        for _ in range(self.num_layers):
            list_independent_x = []
            for i in range(self.M):
                # print(x.shape)
                independent_x = self.indenpendet_transformer[i](x[:, i, :, :])
                independent_x = self.ln(x[:, i, :, :] + independent_x)
                independent_x = self.indenpendet_FFN[i](independent_x)
                list_independent_x.append(independent_x.unsqueeze(1))
            independent_x = torch.cat(list_independent_x, dim=1)
            independent_x = self.dagcn(independent_x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

            x = self.FFN(independent_x + x)

        out = self.linear_time(x.transpose(2, 3)).squeeze(-1)
        out = self.linear_feat(out.reshape(B, -1))
        return out, None, None


if __name__ == "__main__":
    x = torch.randn((128,32,15))        # B,L,D
    model = wo2_MoEE_Model(time_len=32,M=3,list_input_dims=[5,5,5])
    out,_,_ = model(x)
    print(out.shape)
