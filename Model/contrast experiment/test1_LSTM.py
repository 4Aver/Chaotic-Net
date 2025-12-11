import torch
import torch.nn as nn
import torch.optim as optim


class LSTM_Model(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim1, hidden_dim2, pred_len):
        super(LSTM_Model, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        #self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(hidden_dim2, hidden_dim2, batch_first=True)
        self.fc_feature = nn.Linear(hidden_dim2, 1)
        self.fc_time = nn.Linear(seq_len,pred_len)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        #x = self.dropout2(x) # Uncomment if needed
        x, _ = self.lstm3(x)
        # Flatten the output for the Dense layer
        x = self.fc_feature(x).squeeze(-1)
        x = self.fc_time(x)
        return x


if __name__ == "__main__":
    x = torch.randn((128,32,15))        # [B, L, D]
    model = LSTM_Model(seq_len=32,input_dim=15, hidden_dim1=128, hidden_dim2=64, pred_len=1)
    out = model(x)
    print(out.shape)

