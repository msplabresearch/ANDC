import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

class HLD(nn.Module):
    def __init__(self, *args, **kwargs):
        super(HLD, self).__init__()
        input_dim = kwargs.get("input_dim", args[0])
        hidden_dim = args[1]
        num_layers = args[2]
        output_dim = args[3]
        self.prediction_type = args[4]
        self.label_learning = args[5]
        p = kwargs.get("dropout", 0.5)
        
        # self.prediction_type = kwargs.get("prediction_type", "dimensional") 
        assert self.prediction_type in ["categorical", "dimensional"]

        self.fc=nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
            )
        ])
        for lidx in range(num_layers-1):
            self.fc.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
                )
            )
        self.out = nn.Sequential(
                nn.Linear(hidden_dim, output_dim)
            )
        self.drop1d = nn.Dropout(p)
    def get_repr(self, x):
        h = x
        for lidx, fc in enumerate(self.fc):
            if lidx < 2:
                h = self.drop1d(h)
            h=fc(h)
        return h
        
    def forward(self, x):
        h = self.get_repr(x)
        result = self.out(h)

        return result
