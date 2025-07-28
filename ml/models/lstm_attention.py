import torch
import torch.nn as nn

class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                  # (B, T, H)
        attn_w = torch.softmax(self.attn(lstm_out), dim=1)  # (B, T, 1)
        context = torch.sum(attn_w * lstm_out, dim=1)       # (B, H)
        return self.fc(context)
