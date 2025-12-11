import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Union, Tuple
from torch.distributions.normal import Normal
# from 研究生课题.前沿研究.Chaotic_Net.models.DAGCN import DAGCN, Mulit_DAGCN
# from 研究生课题.前沿研究.Chaotic_Net.models.DAGCN import Attention_DAGCN

# from einops import rearrange
from typing import Union, Tuple
from torch.distributions.normal import Normal

torch.manual_seed(21)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用gpu


class Channel_Embedding_ablation(nn.Module):
    def __init__(self, input_dim, num_experts=4, out_channels=10, k=4, emb_kernel_size=3, emb_stride=1):
        super(Channel_Embedding_ablation, self).__init__()
        self.k = k
        self.noisy_gating = True
        self.num_experts = num_experts
        self.out_channels = out_channels

        self.list_m_experts = nn.ModuleList()
        self.list_m_gating = []
        self.list_m_noise = []

        experts = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=out_channels, kernel_size=emb_kernel_size,
                      stride=emb_stride),
            nn.Tanh(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels * num_experts, kernel_size=1)
        )
        # w_gate = nn.Parameter(0.005*torch.randn(input_dim * 5, num_experts), requires_grad=True).to(device)
        # w_noise = nn.Parameter(0.005*torch.randn(input_dim * 5, num_experts), requires_grad=True).to(device)
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

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
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
        clean_logits = x @ self.list_m_gating[0]  # 计算每个expert的权重
        if self.noisy_gating and train:  # 在训练中加入残差等
            raw_noise_stddev = x @ self.list_m_noise[0]  # 根据输入数据设置噪声权重
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
        list_gates = []

        B, d, L = x.shape
        # gates, load = self.noisy_top_k_gating(input_x[:,:,-1], i, self.training)
        gates, load = self.noisy_top_k_gating(x[:, :, -6:-1].reshape(B, d*5), self.training)
        list_gates.append(gates.unsqueeze(-1))

        # calculate importance loss
        importance = gates.sum(0)  # 将每个expert的gates权重加和,计算总的贡献值
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        out_loss = out_loss + loss

        out_raw = self.list_m_experts[0](x)  # B,D*E,l
        out_raw = out_raw.permute(0, 2, 1).reshape(B, -1, self.out_channels, self.num_experts)  # B,l,D,E

        moe_out = torch.einsum("BLDE,BE->BLD", out_raw, gates).permute(0, 2, 1)

        return moe_out


import torch
import torch.nn as nn
import torch.optim as optim


class LSTM_Model(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim1, hidden_dim2, pred_len):
        # def __init__(self, seq_len, input_dim, hidden_dim1, hidden_dim2, pred_len):
        super(LSTM_Model, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        # self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(hidden_dim2, hidden_dim2, batch_first=True)
        self.fc_feature = nn.Linear(hidden_dim2, 1)
        # self.fc_feature = nn.Linear(hidden_dim2, 80)

        self.fc_time = nn.Linear(seq_len, pred_len)

        self.embedding = Channel_Embedding_ablation(input_dim, out_channels=input_dim, num_experts=2, k=2)

    def forward(self, x):
        B, _, _ = x.shape
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = x.transpose(1, 2)  # B,M,L,D

        # x = x.permute(0,2,1)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        # x = self.dropout2(x) # Uncomment if needed
        x, _ = self.lstm3(x)
        # Flatten the output for the Dense layer

        x = self.fc_feature(x).squeeze(-1)
        x = self.fc_time(x)

        # x = self.fc_feature(x).permute(0,2,1)
        # x = self.fc_time(x).squeeze(-1)
        return x


if __name__ == "__main__":
    x = torch.randn((128, 32, 15))
    model = LSTM_Model(seq_len=30, input_dim=15, hidden_dim1=128, hidden_dim2=64, pred_len=1)
    out = model(x)
    print(out.shape)
