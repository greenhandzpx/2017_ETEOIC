import torch
from torch import nn
import torch.nn.functional as f

class F_k(nn.Module):
    def __init__(self, channels, isFinal = False):
        super().__init__()
        #print(channels)
        self.H = nn.Parameter(torch.normal(0, 0.01, size=(1, channels, 1, 1)))
        self.b = nn.Parameter(torch.normal(0, 0.01, size=(1, channels, 1, 1)))
        if isFinal == False:
            self.a = nn.Parameter(torch.normal(0, 0.01, size=(1, channels, 1, 1)))
        self.isFinal = isFinal
    def forward(self, X):
        if self.isFinal:
            return torch.sigmoid(f.softplus(self.H) * X + self.b) 
        X = f.softplus(self.H) * X + self.b
        return X + torch.tanh(self.a) * torch.tanh(X)


class Culmulative(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.f1 = F_k(channels)
        self.f2 = F_k(channels)
        self.f3 = F_k(channels)
        self.f4 = F_k(channels, True)

    def forward(self, X):
        X = self.f1(X)
        X = self.f2(X)
        X = self.f3(X)
        return self.f4(X) 