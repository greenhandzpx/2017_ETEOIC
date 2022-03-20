# from this import d
# from turtle import forward
import torch
#from operator import inv
#from numpy import pad
import math
from torch import nn
from .gdn import GDN

import torch.nn.functional as F


class Analysis(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.conv9_1 = nn.Conv2d(
            in_channels=3, out_channels=256,
            kernel_size=9)
        torch.nn.init.xavier_normal_(self.conv9_1.weight.data, (math.sqrt(2 * (3 + 256) / (6))))
        torch.nn.init.constant_(self.conv9_1.bias.data, 0.01)
        self.downSample4_1 = nn.AvgPool2d(kernel_size=4, stride=4)
        #self.downSample4 = F.interpolate(X, scale_factor=0.25)
        self.GDN1 = GDN(ch=256, device=device)
        self.GDN2 = GDN(ch=256, device=device)
        self.GDN3 = GDN(ch=256, device=device)

        self.conv5_1 = nn.Conv2d(
            in_channels=256, out_channels=256,
            kernel_size=5)
        torch.nn.init.xavier_normal_(self.conv5_1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv5_1.bias.data, 0.01)

        self.conv5_2 = nn.Conv2d(
            in_channels=256, out_channels=256,
            kernel_size=5)
        torch.nn.init.xavier_normal_(self.conv5_2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv5_2.bias.data, 0.01)

        self.downSample2_1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.downSample2_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        #self.downSample2 = F.interpolate(input=256, scale_factor=0.5)

    def forward(self, X):
        X = self.GDN1(self.downSample4_1(self.conv9_1(X)))
        #print(X.shape)
        #X = self.GDN(F.interpolate(self.conv9(X), scale_factor=0.25))
        X = self.GDN2(self.downSample2_1(self.conv5_1(X)))
        # print(X[0][0][10][15])
        #X = self.GDN(F.interpolate(self.conv5(X), scale_factor=0.5))
        X = self.GDN3(self.downSample2_2(self.conv5_2(X)))
        #X = self.GDN(F.interpolate(self.conv5(X), scale_factor=0.5))
        return X
    

class Synthesis(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.conv9_1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=3,
            kernel_size=9)
        torch.nn.init.xavier_normal_(self.conv9_1.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.conv9_1.bias.data, 0.01)

        self.upSample2_1 = nn.Upsample(scale_factor=2)
        self.upSample2_2 = nn.Upsample(scale_factor=2)
        self.upSample4_1 = nn.Upsample(scale_factor=4)
        # IGDN
        self.IGDN1 = GDN(ch=256, device=device, inverse=True)
        self.IGDN2 = GDN(ch=256, device=device, inverse=True)
        self.IGDN3 = GDN(ch=256, device=device, inverse=True)

        self.conv5_1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=256,
            kernel_size=5)
        torch.nn.init.xavier_normal_(self.conv5_1.weight.data, (math.sqrt(2 * 1)))
        torch.nn.init.constant_(self.conv5_1.bias.data, 0.01)

        self.conv5_2 = nn.ConvTranspose2d(
            in_channels=256, out_channels=256,
            kernel_size=5)
        torch.nn.init.xavier_normal_(self.conv5_2.weight.data, (math.sqrt(2 * 1)))
        torch.nn.init.constant_(self.conv5_2.bias.data, 0.01)
        #self.upSample2 = F.interpolate(input=256, scale_factor=2)

        self.upSample_final = nn.Upsample(scale_factor=1.035)
    def forward(self, X):
        #print(X[0][0][5][9])
        X = self.conv5_1(self.upSample2_1(self.IGDN1(X)))
        X = self.conv5_2(self.upSample2_2(self.IGDN2(X)))
        #print(X[0][0][30][30])
        #print(self.IGDN(X)[0][0][30][30])
        Y = self.upSample4_1(self.IGDN3(X))
        #print(Y[0][0][40][40])
        X = self.conv9_1(Y)
        # X = self.conv9_1(self.upSample4(self.IGDN(X)))
        # print(X[0][0][56][45])
        X = self.upSample_final(X)
        #X = nn.Upsample(scale_factor=1.035)(X)
        #print(X.shape)

        return X


#Synthesis.apply()

#Synthesis.apply()