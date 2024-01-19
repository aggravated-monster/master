import torch
from torch import nn
import numpy as np


class LinearAgentNN(nn.Module):
    def __init__(self, input_shape, n_actions, freeze=False):
        super().__init__()
        # Linear layers
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(4*5*16, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        if freeze:
            self._freeze()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, x):
        #return self.network(x)
        x = self.flatten(x)
        logits = self.network(x)
        return logits

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        # np.prod returns the product of array elements over a given axis
        return int(np.prod(o.size()))

    def _freeze(self):
        for p in self.network.parameters():
            p.requires_grad = False
