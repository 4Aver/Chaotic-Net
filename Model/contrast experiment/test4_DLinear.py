import torch
import torch.nn as nn
import torch.nn.functional as F


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self,seq_len, pred_len, enc_in,decompsition_kernel_size=3, individual=False):
        super(Model, self).__init__()
        self.seq_len = seq_len          # 输入序列长度
        self.pred_len = pred_len        # 输出序列长度
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(decompsition_kernel_size)     # 使用平均池化达到切分季节和趋势的目的
        self.individual = individual        # True代表通道独立性，反之不使用通道独立性
        self.channels = enc_in            # 选择通道个数

        # 使用通道独立性，对每一个通道都使用不同的Linear进行预测，然后将预测结果统一
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # 不使用通道独立性，考虑到了通道之间的关系
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        self.dense_feature = nn.Linear(enc_in,1)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)        # x: B, L(seq_len), D(多变量个数)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)     # B, D, L

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)       # B, D, pred_len
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)                # B, D, pred_len
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forward(self, x_enc):
        dec_out = self.encoder(x_enc)
        dec_out = self.dense_feature(dec_out).squeeze(-1)
        return dec_out[:, -self.pred_len:]  # [B, L, D]


if __name__ == "__main__":
    x = torch.randn((128,32,15))        # [B, L, D]
    model = Model(seq_len=32,pred_len=1,enc_in=15,individual=False)
    out = model(x)
    print(out.shape)

