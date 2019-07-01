import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F

class L2NormScale(nn.Module):
    def __init__(self, n_channels, init_scale):
        super(L2NormScale, self).__init__()
        self.n_channels = n_channels
        self.init_scale = init_scale
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        init.constant(self.weight, self.init_scale)
        
    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return x