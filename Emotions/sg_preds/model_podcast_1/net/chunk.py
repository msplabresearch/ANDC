import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
wav2vec feature encoder
conv_feature_layers: str = field(
    default="[(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1), (512, 1, 1)]",
    metadata={
        "help": "convolutional feature extraction layers [(dim, kernel_size, stride), ...]"
    },
)
"""
class CNN_GatedVec(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CNN_GatedVec, self).__init__()
        self.input_dim = kwargs.get("input_dim", args[0])
        self.window_size = kwargs.get("window_size", 50)
        self.chunk_num = kwargs.get("chunk_num", 26)
        

        self.encoder=nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.input_dim, 128, 3, stride=1, padding=1), nn.BatchNorm1d(128), nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, 3, stride=1, padding=1), nn.BatchNorm1d(128), nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(128, 64, 3, stride=1, padding=1), nn.BatchNorm1d(64), nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(64, 64, 3, stride=1, padding=1), nn.BatchNorm1d(64), nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv1d(64, 32, 3, stride=2, padding=1), nn.BatchNorm1d(32), nn.ReLU()
            ),
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Sequential(
                nn.Linear(int(32*self.window_size/2), self.input_dim), nn.ReLU()
            )
            
        ])
        self.gate_W = nn.parameter.Parameter(torch.ones((self.input_dim)))
        self.gate_b = nn.parameter.Parameter(torch.zeros((self.input_dim)))
        
    def forward(self, x):
        # Input x: (N, 26, 512, 50)
        # (N, 26, 512, 50) => (N*26, 512, 50) => (N*26, 512) => (N, 26, 512) 
        # => (N, 26, 512) * (N, 26, 1) => (N, 26, 512) => (N, 512)
        
        # Change shape for the CNN encoder
        h = x.view(-1, self.input_dim, self.window_size)
        for enc in self.encoder:
            h = enc(h)
        h = h.view(-1, self.chunk_num, self.input_dim)

        # Temporal aggregation by using GatedVec
        g = self.gate_W * h + self.gate_b
        z = torch.sum(g*h, dim=1)

        return z

class LSTM_AttenVec(nn.Module):
    def __init__(self, *args, **kwargs):
        super(LSTM_AttenVec, self).__init__()
        self.input_dim = kwargs.get("input_dim", args[0])
        self.hidden_dim = kwargs.get("hidden_dim", args[1])
        self.window_size = kwargs.get("window_size", 50)
        self.chunk_num = kwargs.get("chunk_num", 11)
        

        self.encoder=nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=2, dropout=0.5, batch_first=True)
        self.enc_batchnorm = nn.BatchNorm1d(self.hidden_dim)

        self.Attn=nn.RNN(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.Attn_general_mat=nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.Attn_vector=nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.input_dim, bias=False), nn.Tanh()
        )
        
    def forward(self, x):
        # Input x: (N, 11, 512, 50)
        # (N, 11, 512, 50) => (N*11, 512, 50) => (50, N*11, 512)
        # => (50, N*11, 128) => (N*11, 128) => (N, 11, 128) 
        
        # Change shape for the RNN encoder
        h = x.view(-1, self.input_dim, self.window_size) # (N*11, 512, 50)
        h = h.permute(0, 2, 1) # (N*11, 50, 512)
        h, _ = self.encoder(h) # (N*11, 50, 128) 
        h = h[:, -1, :] # (N*11, 128) 

        h = self.enc_batchnorm(h) # (N*11, 128) 
        h = h.view(-1, self.chunk_num, self.hidden_dim) # (N, 11, 128) 
        
        encode, _ = self.Attn(h) # (N, 11, 128)
        last_encode = encode[:, -1, :] # (N, 128)

        Wahs = self.Attn_general_mat(encode) # (N, 11, 128)
        
        ht = last_encode.unsqueeze(2) # (N, 128, 1)
        score = torch.matmul(Wahs, ht)  #(N, 11, 128)*(N, 128, 1) = (N, 11, 1)
        score = score.squeeze(2)
        
        attn_weight = nn.Softmax(dim=1)(score) # (N, 11)
        attn_weight = attn_weight.unsqueeze(1) # (N, 1, 11)
        
        context_vector = torch.matmul(attn_weight, encode) # (N, 1, 11) * (N, 11, 128) = (N, 1, 128)
        context_vector = context_vector.squeeze(1) # (N, 128)

        pre_activation = torch.cat([context_vector, last_encode], dim=1) # (N, 256)
        z = self.Attn_vector(pre_activation) # (N, 512)

        return z