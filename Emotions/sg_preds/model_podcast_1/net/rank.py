import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

class RankNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RankNet, self).__init__()
        input_dim = kwargs.get("input_dim", args[0])
        
        
        self.out = nn.Sequential(
                nn.Linear(input_dim, 1), nn.Sigmoid()
            )
    def forward(self, x):
        score = self.out(x)
        return score