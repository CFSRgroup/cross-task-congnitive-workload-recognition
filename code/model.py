import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义模型
class LstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LstmModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        #self.batchnormalize = nn.BatchNorm1d
        #nn.Dropout(dropout_rate))
        self.fc = nn.Linear(hidden_size, num_classes)
        #self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # print(h0.shape)
        # Forward propagate LSTM
        # a=(h0, c0)
        # print(np.shape(a[0]))
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        # Decode the hidden state of the last time step
        out = torch.dropout(out, p=0.5, train=self.training)
        out = self.fc(out[:, -1, :])
        # torch.tensor(hidden)
        # out = self.fc(hidden_cat)
        return out

