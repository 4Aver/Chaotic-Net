import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Att_CNN_LSTM(nn.Module):
    def __init__(self, in_channels, out_channels,hidden_size,seq_len,kernel_size,dilation_size,pred_len):
        super(Att_CNN_LSTM, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,
                                           stride=1, padding=(kernel_size - 1) * dilation_size, dilation=dilation_size)
        self.conv = nn.Sequential(
            # nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            self.conv1,
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        self.dense1 = nn.Linear(seq_len, seq_len)
        self.lstm = nn.LSTM(input_size=out_channels+in_channels, hidden_size=hidden_size,
                            num_layers=2, batch_first=True)
        self.dense2 = nn.Linear(hidden_size, hidden_size)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_size,num_heads=4,batch_first=True,
                                               dropout=0.2)

        self.fc = nn.Linear(hidden_size, 1)
        self.fc_time = nn.Linear(seq_len,pred_len)
        self.dropout = nn.Dropout(0.1)

    def forward(self, emb_x):
        '''
        x:(batch,in_channel,seq_len)        作为conv1d输入
          (batch,seq_len-kernel_size+1,out_channel)     # 作为LSTM的输入
        '''
        emb_x = emb_x.permute(0,2,1)
        conv_x = self.conv(emb_x)            # (batch,out_channel,seq_len-kernel_size*2+2)       # 因为有池化层的kernel_size
        conv_x = self.dense1(conv_x)
        x = torch.cat((conv_x,emb_x),dim=1)
        x = x.permute(0, 2, 1)      # (batch,seq_len-kernel_size*2+2,out_channel)
        x, _ = self.lstm(x)         # (batch,seq_len-kernel_size*2+2,hidden_size)
        x = self.dropout(x)
        x = self.dense2(x)
        x, _ = self.attention(x,x,x)
        x = self.fc(x).squeeze(-1)
        x = self.fc_time(x)
        return x


if __name__ == '__main__':
    x = torch.randn((128,50,15))        # [B, L, D]
    model = Att_CNN_LSTM(in_channels=15,out_channels=16,hidden_size=64,seq_len=50,kernel_size=2,dilation_size=1,pred_len=2)
    out = model(x)
    print(out.shape)
