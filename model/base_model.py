import torch
import torch.nn as nn

class WaterQualityModel(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(WaterQualityModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim) # Output raw logits for CrossEntropyLoss
        )

    def forward(self, x):
        return self.network(x)
