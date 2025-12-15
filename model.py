import torch
import torch.nn as nn

class PowerDemandLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(PowerDemandLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # LSTM 출력: (batch, seq_len, hidden_size)
        out, _ = self.lstm(x)
        
        # Many-to-One: 마지막 시점의 Hidden State만 사용
        last_out = out[:, -1, :]
        prediction = self.fc(last_out)
        return prediction