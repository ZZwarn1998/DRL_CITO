import torch
from torch import nn
import numba as nb


class DuelingNetwork(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_actions: int):
        super(DuelingNetwork, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)