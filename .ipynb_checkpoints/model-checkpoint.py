#from statistics import mode
#from zmq import device
import math
from models.gdn import GDN
from models import prob_culmulative, models
import torch
from torch import nn 
from compressai.entropy_models import EntropyBottleneck

class Network(nn.Module):
    def __init__(self, device, N=256, training=True):
        super().__init__()

        self.device = device
        self.training = training
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.encode = nn.Sequential(
            nn.Conv2d(3, N, 9, stride=4),
            GDN(N, device=self.device),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            GDN(N, device=self.device),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            GDN(N, device=self.device)
        )

        self.decode = nn.Sequential(
            GDN(N, inverse=True, device=self.device),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True, device=self.device),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True, device=self.device),
            nn.ConvTranspose2d(N, 3, 9, stride=4, padding=4, output_padding=3)
            # deconv(N, N),
            # GDN(N, inverse=True),
            # deconv(N, N),
            # GDN(N, inverse=True),
            # deconv(N, 3),
        )

    def forward(self, x):
        y = self.encode(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        
#         if self.training:
#             noise = nn.init.uniform_(torch.empty(y.shape), -0.5, 0.5)  
#             noise = noise.to(self.device)
#             #print(self.device)
#             #print(noise.shape)
#             q = y + noise
#             #print(q)
#         else:
#             q = torch.round(y)
        #x_hat = self.decode(q)
    
        x_hat = self.decode(y_hat)
        #clipped_X_hat = X_hat.clamp(0., 1.)

#         def calculate_rate(Y):
#             cumulative = prob_culmulative.Culmulative(Y.shape[1]).to(self.device)
#             # 两个概率累积的差值即为对应点的概率
#             p_y = cumulative(Y + 0.5) - cumulative(Y - 0.5)
#             sum_of_bits = torch.sum(-torch.log2(p_y))
#             # 这里不确定要不要除以输入图片的通道数
#             return sum_of_bits / (x.shape[0] * x.shape[2] * x.shape[3])
        
        def calculate_rate(y_likelihoods):
            N, _, H, W = x.size()
            num_pixels = N * H * W
            return torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)
            
        #rate = calculate_rate(q)
        rate = calculate_rate(y_likelihoods)
        #distortion = torch.mean(torch.square(X - X_hat))
        return x_hat, rate



class Compress_and_DeCompress(nn.Module):
    def __init__(self, device, training=False):
        super().__init__()
        self.analysis = models.Analysis(device=device)
        # self.density = density.Density()
        self.synthesis = models.Synthesis(device=device)
        self.training = training
        self.device = device

    def forward(self, X):
        Y = self.analysis(X)
        if self.training:
            noise = nn.init.uniform_(torch.empty(Y.shape), -0.5, 0.5)  
            noise = noise.to(self.device)
            #print(self.device)
            #print(noise.shape)
            q = Y + noise
            #print(q)
            
        else:
            q = torch.round(Y)

        X_hat = self.synthesis(q)
        #clipped_X_hat = X_hat.clamp(0., 1.)

        def calculate_rate(Y):
            cumulative = prob_culmulative.Culmulative(Y.shape[1]).to(self.device)
            # 两个概率累积的差值即为对应点的概率
            p_y = cumulative(Y + 0.5) - cumulative(Y - 0.5)
            sum_of_bits = torch.sum(-torch.log2(p_y))
            # 这里不确定要不要除以输入图片的通道数
            return sum_of_bits / (X.shape[0] * X.shape[2] * X.shape[3])

        rate = calculate_rate(q)
        #distortion = torch.mean(torch.square(X - X_hat))

        return X_hat, rate
        



