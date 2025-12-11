# wo5_DAGCN

import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Union, Tuple
from torch.distributions.normal import Normal

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
    def __init__(self, list_input_dim, num_experts=8, out_channels=10, emb_kernel_size=3, emb_stride=1):
        super(Channel_Embedding, self).__init__()
        self.k = 2
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
        scores = F.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d), dim=-1)

        output = torch.bmm(self.drop_out(scores), v)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.w_o(output_concat)


class var_Attention1(nn.Module):
    def __init__(self, num_nodes, embed_dim, num_time_steps, in_dims, out_dims, num_layers=1):
        super(var_Attention1, self).__init__()

        self.w_q = nn.Conv2d(in_channels=num_nodes, out_channels=num_nodes, kernel_size=1, groups=num_nodes)
        self.w_k = nn.Conv2d(in_channels=num_nodes, out_channels=num_nodes, kernel_size=1, groups=num_nodes)

        self.w_v = nn.Conv2d(in_channels=num_nodes, out_channels=num_nodes, kernel_size=1)
        self.w_o = nn.Conv2d(in_channels=num_nodes, out_channels=num_nodes, kernel_size=1)

        self.num_time_steps = num_time_steps
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_layers = num_layers

    def forward(self, x):
        B, M, L, D = x.shape

        for _ in range(self.num_layers):
            res_x = x
            unit = torch.eye(self.num_nodes)

            q = F.tanh(self.w_q(x)).reshape(B, M, -1)
            k = F.tanh(self.w_k(x)).reshape(B, M, -1)

            # q = self.w_q(x).reshape(B, M, -1)
            # k = self.w_k(x).reshape(B, M, -1)

            d = x.shape[-1]

            scores = F.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d), dim=-1).reshape(B, M, M) + unit  # B, M, M
            A = scores

            x = torch.einsum('bnm, bmld -> bnld', A, self.w_v(x))
            x = self.w_o(x) + res_x
        return x, _, _


class var_DAGCN(nn.Module):
    def __init__(self, num_nodes, embed_dim, num_time_steps, in_dims, out_dims, num_layers=1):
        super(var_DAGCN, self).__init__()
        self.num_time_steps = num_time_steps
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 动态空间节点嵌入向量
        self.dn_embeddings = nn.Parameter(0.5 * torch.ones(num_nodes,
                                                           embed_dim),
                                          requires_grad=True)  # [T, N, embed_dim]
        self.weights = nn.Parameter(torch.ones(1) / 2, requires_grad=True)  # 可学习的权重参数

        self.w_v = nn.Conv2d(in_channels=num_nodes, out_channels=num_nodes, kernel_size=1)
        self.w_o = nn.Conv2d(in_channels=num_nodes, out_channels=num_nodes, kernel_size=1)

    def forward(self, x):
        B, M, L, D = x.shape

        for _ in range(self.num_layers):
            res_x = x

            supports = F.softmax(F.relu(torch.einsum('ne, se -> ns', self.dn_embeddings, self.dn_embeddings)), dim=-1)
            unit = torch.eye(self.num_nodes).to(supports.device)
            supports = supports + unit

            A = supports
            x = torch.einsum('nm, bmld -> bnld', A, self.w_v(x))
            x = self.w_o(x) + res_x

        return x, _, _

class wo5_DCMC_Model(nn.Module):
    def __init__(self, M, time_len, list_input_dims, d=64, num_layers=1, type_name='DCMC'):
        super(wo5_DCMC_Model, self).__init__()
        self.M = M
        self.num_layers = num_layers
        self.list_input_dims = list_input_dims
        self.type_name = type_name

        self.embedding = Channel_Embedding(list_input_dim=self.list_input_dims, out_channels=d)
        self.ln = nn.LayerNorm(normalized_shape=(30, d))

        self.indenpendet_transformer = nn.ModuleList(
            [MultiAttention(input_dim=d, num_hiddens=d, num_heads=8).to(device) for _ in range(M)])
        self.indenpendet_FFN = nn.ModuleList([PositionwiseFeedForward(d, d).to(device) for _ in range(M)])

        if type_name == 'AGCN':
            self.layer = var_DAGCN(num_time_steps=30, num_nodes=M, embed_dim=3, in_dims=d,out_dims=d)
        if type_name == 'Attention':
            self.layer = var_DAGCN(num_time_steps=30, num_nodes=M, embed_dim=3, in_dims=d,out_dims=d)

        self.FFN = PositionwiseFeedForward(d, d)

        self.weights = nn.Parameter(torch.ones(2) / 2)  # 可学习的权重参数

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
                # print(x.shape)
                independent_x = self.indenpendet_transformer[i](x[:, i, :, :])
                independent_x = self.ln(x[:, i, :, :] + independent_x)
                independent_x = self.indenpendet_FFN[i](independent_x)
                list_independent_x.append(independent_x.unsqueeze(1))
            independent_x = torch.cat(list_independent_x, dim=1)

            if self.type_name == 'DCMC':
                dagcn_x= independent_x + x
            if self.type_name == 'AGCN':
                dagcn_x,_,_ = self.layer(independent_x + x)
            if self.type_name == 'Attention':
                dagcn_x,_,_ = self.layer(independent_x + x)

            x = self.FFN(self.ln(dagcn_x) + x) + x

        out = self.linear_time(x.transpose(2, 3)).squeeze(-1)
        out = self.linear_feat(out.reshape(B, -1))
        return out, loss, gates, _, _

    def print_concat_weights(self):
        print('Independent Transformer and DAGCN concat weights:', self.dn_embeddings)


if __name__ == "__main__":
    x = torch.randn((128,32,15))        # B,L,D
    model = wo5_DCMC_Model(time_len=32,M=3,list_input_dims=[5,5,5],type_name='Attention')
    out, _, gates,_ ,_ = model(x)
    print(out.shape)
