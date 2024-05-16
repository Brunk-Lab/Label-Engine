import torch
import torch.nn as nn


class interModel(nn.Module):
    def __init__(self, num_in_channels, num_out_channels):
        super().__init__()
        
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels

    def forward(self):
        raise NotImplementedError("Subclasses should implement this!")