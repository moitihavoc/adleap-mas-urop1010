import torch
import torch.nn as nn
import torch.nn.functional as F

class TraceCNN(nn.Module):
    def __init__(self, input_channels=9, num_actions=5):
        super(TraceCNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 256),
            nn.ReLU()
        )
        
        # Policy Head (Outputs raw logits for 5 actions)
        self.policy_head = nn.Linear(256, num_actions)
        
        # Value Head (Outputs single float for expected return)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        features = self.conv(x)
        pi_logits = self.policy_head(features)
        value = self.value_head(features)
        return pi_logits, value

