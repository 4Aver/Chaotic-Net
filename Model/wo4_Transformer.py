# wo4_Transformer

import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Union, Tuple
from torch.nn.utils import weight_norm
from torch.distributions.normal import Normal
from 研究生课题.前沿研究.Chaotic_Net.models.DAGCN import DAGCN, Mulit_DAGCN

torch.manual_seed(21)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用gpu



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        # print(x,x.shape)
        # print(x[:, :, :-self.chomp_size].contiguous(),x[:, :, :-self.chomp_size].contiguous().shape)
        # print('以进行裁剪')
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化
        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        # print(self.conv1.weight.shape)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs,seq_len, num_channels,pred_len, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25，也就代表有25个卷积核在这次卷积中执行
        :param kernel_size: int, 卷积核尺寸，只需要定义卷积核长度
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,                        # 膨胀卷积
                                     padding=(kernel_size - 1) * dilation_size,     # 因果卷积
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.dense_pred = nn.Linear(seq_len,pred_len)
        self.dense_feature = nn.Linear(num_channels[-1],1)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        为什么输入和输出数据结构是这样的
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        x = x.permute(0,2,1)
        x = self.network(x)
        x = x.permute(0,2,1)
        return x


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
    def __init__(self, list_input_dim, num_experts=4, out_channels=10, emb_kernel_size=3, emb_stride=1):
        super(Channel_Embedding, self).__init__()
        self.k = 4
        self.noisy_gating = True
        self.num_experts = num_experts
        self.out_channels = out_channels
        self.list_input_dim = list_input_dim

        self.list_m_experts = nn.ModuleList()
        self.list_m_gating = []
        self.list_m_noise = []

        for input_dim in list_input_dim:
            experts = nn.Sequential(
                nn.Conv1d(in_channels=input_dim, out_channels=out_channels, kernel_size=emb_kernel_size,
                          stride=emb_stride),
                nn.Tanh(),
                nn.Conv1d(in_channels=out_channels, out_channels=out_channels * num_experts, kernel_size=1))

            # gating = nn.Linear(input_dim, num_experts)
            # noise = nn.Linear(input_dim,num_experts)

            w_gate = nn.Parameter(torch.zeros(input_dim * 5, num_experts), requires_grad=True).to(device)
            w_noise = nn.Parameter(torch.zeros(input_dim * 5, num_experts), requires_grad=True).to(device)

            self.list_m_experts.append(experts)
            self.list_m_gating.append(w_gate)
            self.list_m_noise.append(w_noise)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        """计算样本变异系数
        The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`(标量).
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, m, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.list_m_gating[m]  # 计算每个expert的权重
        if self.noisy_gating and train:  # 在训练中加入残差等
            raw_noise_stddev = x @ self.list_m_noise[m]  # 根据输入数据设置噪声权重
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        index, out_loss = 0, 0
        list_combine_features = []
        list_spatial_x = []
        list_gates = []
        for i, dim in enumerate(self.list_input_dim):
            input_x = x[:, index:index + dim, :]
            list_spatial_x.append(x[:, index, :].unsqueeze(1))
            index += dim

            B, d, L = input_x.shape
            # gates, load = self.noisy_top_k_gating(input_x[:,:,-1], i, self.training)
            gates, load = self.noisy_top_k_gating(input_x[:, :, -6:-1].reshape(B, dim * 5), i, self.training)
            list_gates.append(gates.unsqueeze(-1))

            # calculate importance loss
            importance = gates.sum(0)  # 将每个expert的gates权重加和,计算总的贡献值
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= loss_coef
            out_loss = out_loss + loss

            out_raw = self.list_m_experts[i](input_x)  # B,D*E,l
            out_raw = out_raw.permute(0, 2, 1).reshape(B, -1, self.out_channels, self.num_experts)  # B,l,D,E

            moe_out = torch.einsum("BLDE,BE->BLD", out_raw, gates).permute(0, 2, 1)

            list_combine_features.append(moe_out.unsqueeze(1))
        combine_features = torch.cat(list_combine_features, dim=1)
        gates = torch.cat(list_gates, dim=-1)
        return combine_features, out_loss, gates


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


class wo4_Transformer_Model(nn.Module):
    def __init__(self, M, time_len, list_input_dims,type_model='MLP', d=64, num_layers=1):
        super(wo4_Transformer_Model, self).__init__()
        self.M = M
        self.type_model = type_model
        self.num_layers = num_layers
        self.list_input_dims = list_input_dims

        self.embedding = Channel_Embedding(list_input_dim=self.list_input_dims, out_channels=d)
        self.ln = nn.LayerNorm(normalized_shape=(30, d))

        if type_model == 'MLP':
            self.indenpendet_layer = nn.ModuleList([PositionwiseFeedForward(d, d).to(device) for _ in range(M)])

        if type_model == 'LSTM':
            layer = nn.Sequential(nn.LSTM(d, d, batch_first=True),
                                  )
            self.indenpendet_layer = nn.ModuleList([layer.to(device) for _ in range(M)])

        if type_model == 'TCN':
            layer = nn.Sequential(TemporalConvNet(num_inputs=d, num_channels=[d*2,d], kernel_size=2,seq_len=time_len,pred_len=1))
            self.indenpendet_layer = nn.ModuleList([layer.to(device) for _ in range(M)])

        self.dagcn = Mulit_DAGCN(num_time_steps=30, num_nodes=M, in_dims=d, out_dims=d, cheb_k=3, embed_dim=2)

        self.weights = nn.Parameter(torch.ones(2) / 2)  # 可学习的权重参数

        self.FFN = PositionwiseFeedForward(d, d)

        self.linear_time = nn.Linear(30, 1)
        self.linear_feat = nn.Linear(M * d, 1)

    def forward(self, in_x):
        B, _, _ = in_x.shape
        x = in_x.permute(0, 2, 1)
        x, loss, gates = self.embedding(x)
        x = x.transpose(2, 3)  # B,M,L,D

        for _ in range(self.num_layers):
            list_independent_x = []
            for i in range(self.M):
                if self.type_model == 'LSTM':
                    independent_x,_ = self.indenpendet_layer[i](x[:, i, :, :])
                else:
                    independent_x = self.indenpendet_layer[i](x[:, i, :, :])
                independent_x = self.ln(x[:, i, :, :] + independent_x)
                list_independent_x.append(independent_x.unsqueeze(1))
            independent_x = torch.cat(list_independent_x, dim=1)
            independent_x = self.dagcn(independent_x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)

            x = self.FFN(independent_x + x)

        out = self.linear_time(x.transpose(2, 3)).squeeze(-1)
        out = self.linear_feat(out.reshape(B, -1))
        return out, loss, gates

    def print_concat_weights(self):
        print('Independent Transformer and DAGCN concat weights:', self.dn_embeddings)


if __name__ == "__main__":
    x = torch.randn((128,32,15))        # B,L,D
    model = wo4_Transformer_Model(time_len=32,M=3,list_input_dims=[5,5,5],type_model='TCN')
    out, _, gates = model(x)
    print(out.shape)
