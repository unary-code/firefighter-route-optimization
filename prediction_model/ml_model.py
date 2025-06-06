import torch.nn as nn


class LSTMBlockagePredictor(nn.Module):
    """
    Defines a class for a PyTorch neural network that will
    be useful for predicting blockage for the next 20 minutes.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 10)  # 10 outputs for 10 future steps

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn.squeeze(0))
        return out  # shape: (batch_size, 10)
