import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

class EmotionRegression(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EmotionRegression, self).__init__()
        input_dim = kwargs.get("input_dim", args[0])
        hidden_dim = args[1]
        num_layers = args[2]
        output_dim = args[3]
        p = kwargs.get("dropout", 0.5)
        self.prediction_type = kwargs.get("prediction_type", "dimensional") 
        assert self.prediction_type in ["categorical", "dimensional"]

        self.fc=nn.ModuleList([
            nn.Sequential(
                # nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
            )
        ])
        for lidx in range(num_layers-1):
            self.fc.append(
                nn.Sequential(
                    # nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
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
    
    def apply_MC(self, x, p=0.5):
        h = x
        for lidx, fc in enumerate(self.fc):
            h = nn.Dropout(p)(h)
            h=fc(h)
        result = self.out(h)
        return result
        
    def forward(self, x):
        h=self.get_repr(x)
        result = self.out(h)
        if self.prediction_type == "categorical":
            result = F.softmax(result, dim=1)
        return result

class ASR(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ASR, self).__init__()
        input_dim = kwargs.get("input_dim", args[0])
        hidden_dim = args[1]
        output_dim = args[2]
       
        self.fc=nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        
        self.out = nn.Sequential(
                nn.Linear(hidden_dim, output_dim)
            )
        
    def forward(self, x):
        # h= x #self.fc(x)
        h = self.fc(x)
        o = self.out(h)
        result = F.log_softmax(o, dim=2)
        return result


class ASR_SER(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ASR_SER, self).__init__()
        input_dim = kwargs.get("input_dim", args[0])
        hidden_dim = args[1]
        output_dim = args[2]
       
        self.fc=nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        
        self.out = nn.Sequential(
                nn.Linear(hidden_dim, output_dim)
            )
        
    def forward(self, x):
        # h= x #self.fc(x)
        h = self.fc(x)
        o = self.out(h)
        result = F.log_softmax(o, dim=2)
        return result