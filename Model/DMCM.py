import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用gpu


class DAGCN(nn.Module):
    def __init__(self, num_time_steps, num_nodes, in_dims, out_dims, cheb_k, embed_dim):
        super(DAGCN, self).__init__()
        self.num_time_steps = num_time_steps
        self.num_nodes = num_nodes
        self.cheb_k = cheb_k
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.embed_dim = embed_dim

        # 动态空间节点嵌入向量
        self.dn_embeddings = nn.Parameter(0.01 * torch.zeros(num_time_steps,
                                                             num_nodes,
                                                             embed_dim),
                                          requires_grad=True)  # [T, N, embed_dim]
        # Theta = E*W--->(tnd,d)
        self.weights_pool = nn.Parameter(torch.randn(embed_dim,
                                                     cheb_k,
                                                     in_dims,
                                                     out_dims))
        self.bias_pool = nn.Parameter(torch.randn(embed_dim, out_dims))

    def forward(self, x):  # x-->[B,T,N,C]
        supports = F.softmax(F.relu(torch.einsum('tne, tse->tns', self.dn_embeddings, self.dn_embeddings)), dim=-1)

        unit = torch.stack([torch.eye(self.num_nodes).to(supports.device) for _ in range(self.num_time_steps)])
        support_set = [unit, supports]

        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.einsum('tnn, tns->tns', 2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=1)  # [T, cheb_k, N, N]
        # print(supports.shape)

        # theta
        theta = torch.einsum('tnd, dkio->tnkio', self.dn_embeddings, self.weights_pool)  # T, N, cheb_k, dim_in, dim_out
        bias = torch.einsum('tnd, do->tno', self.dn_embeddings, self.bias_pool)  # T, N, dim_out
        x_g = torch.einsum('tknm, btmc->btknc', supports, x)
        x_gconv = torch.einsum('btkni, tnkio->btno', x_g, theta) + bias
        return x_gconv  # [B, T, N, dim_out]


# 多层GCN的层数（一般2到3层最好）
class Mulit_DAGCN(nn.Module):
    def __init__(self, num_time_steps, num_nodes, in_dims, out_dims, cheb_k, embed_dim, num_layers=1):
        super(Mulit_DAGCN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = num_nodes
        self.input_dim = in_dims
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()  # 模型列表
        self.dcrnn_cells.append(DAGCN(num_time_steps, num_nodes, in_dims, out_dims, cheb_k, embed_dim))
        for _ in range(1, num_layers):  # 定义多层
            self.dcrnn_cells.append(DAGCN(num_time_steps, num_nodes, in_dims, out_dims, cheb_k, embed_dim))

    def forward(self, x):
        state = None
        for i in range(self.num_layers):
            x = self.dcrnn_cells[i](x)
        return x  # torch.Size([71, 100, 1])


class Attention_DAGCN(nn.Module):
    def __init__(self, num_nodes, embed_dim, num_time_steps, num_layers=1, is_Dimension_Reduction=False, out_var_dims = 3):
        super(Attention_DAGCN, self).__init__()

        if is_Dimension_Reduction == False:
            out_var_dims = num_nodes
            self.w_q = nn.Conv2d(in_channels=num_nodes, out_channels=out_var_dims, kernel_size=7, groups=num_nodes)
            self.w_k = nn.Conv2d(in_channels=num_nodes, out_channels=out_var_dims, kernel_size=7, groups=num_nodes)

        else:
            out_var_dims = out_var_dims
            self.w_q = nn.Conv2d(in_channels=num_nodes, out_channels=out_var_dims, kernel_size=7)
            self.w_k = nn.Conv2d(in_channels=num_nodes, out_channels=out_var_dims, kernel_size=7)

        self.w_v = nn.Conv2d(in_channels=num_nodes, out_channels=out_var_dims, kernel_size=1)
        self.w_o = nn.Conv2d(in_channels=out_var_dims, out_channels=out_var_dims, kernel_size=1)
        self.w_res = nn.Conv2d(in_channels=num_nodes, out_channels=out_var_dims, kernel_size=1)

        self.num_time_steps = num_time_steps
        self.out_var_dims = out_var_dims
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 动态空间节点嵌入向量
        self.dn_embeddings = nn.Parameter(0.5 * torch.ones(out_var_dims,
                                                           embed_dim),
                                          requires_grad=True)  # [T, N, embed_dim]
        self.weights = nn.Parameter(torch.ones(1) / 2, requires_grad=True)  # 可学习的权重参数

    def forward(self, x):
        B, M, L, D = x.shape
        scores, supports = None, None

        res_x = x
        for _ in range(self.num_layers):

            supports = F.softmax(F.relu(torch.einsum('ne, se -> ns', self.dn_embeddings, self.dn_embeddings)), dim=-1)
            unit = torch.eye(self.out_var_dims).to(supports.device)
            supports = supports + unit

            q = F.tanh(self.w_q(x)).reshape(B, self.out_var_dims, -1)
            k = F.tanh(self.w_k(x)).reshape(B, self.out_var_dims, -1)

            # q = self.w_q(x).reshape(B, self.out_var_dims, -1)
            # k = self.w_k(x).reshape(B, self.out_var_dims, -1)

            d = x.shape[-1]

            scores = F.softmax(torch.bmm(q, k.transpose(1, 2))
                               / math.sqrt(d), dim=-1).reshape(B, self.out_var_dims, self.out_var_dims) + unit  # B, M, M

            A = F.softmax(self.weights * scores + (1 - self.weights) * supports, dim=-1) + unit

            x = torch.einsum('bnm, bmld -> bnld', A, self.w_v(x))
            x = self.w_o(x) + self.w_res(res_x)
        return x, scores, supports


if __name__ =='__main__':
    x = torch.rand(128, 20, 36, 10) * 2 - 1
    model = Attention_DAGCN(num_time_steps=36, num_nodes=20, embed_dim=5, in_dims=10, out_dims=64, is_Dimension_Reduction=True)
    out, scores, supports = model(x)
    print(out.shape)