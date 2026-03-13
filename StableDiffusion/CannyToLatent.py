import torch
import torch.nn as nn
from .ZeroConvLayer import ZeroConvLayer

class CannyToLatent(nn.Module):
    def __init__(self,):
        super().__init__()
        self.cannyToLatent = nn.ModuleList([
            
            #maintain stage B,3,512,512 ->B,16,512,512
            nn.Conv2d(in_channels =3,out_channels =16,kernel_size=3,padding = 1),
            nn.SiLU(), 
            nn.Conv2d(in_channels =16,out_channels =16,kernel_size=3,padding = 1),
            nn.SiLU(),   
            
            #down sampling 512->256
            #B,16,512,512 -> B,32,256,256
            nn.Conv2d(in_channels =16,out_channels =32,kernel_size=3,stride = 2,padding = 1),
            nn.SiLU(),
            nn.Conv2d(in_channels =32,out_channels =32,kernel_size=3,padding = 1),
            nn.SiLU(),               
            
            #down sampling 256->128
            #B,32,256,256 -> B,96,128,128
            nn.Conv2d(in_channels =32,out_channels =96,kernel_size=3,stride = 2,padding = 1),
            nn.SiLU(),
            nn.Conv2d(in_channels =96,out_channels =96,kernel_size=3,padding = 1),
            nn.SiLU(),     
            
            # down sampling 128->64
            #B,96,128,128 -> B,256,64,64
            nn.Conv2d(in_channels =96,out_channels =256,kernel_size=3,stride = 2,padding = 1),
            nn.SiLU(),
            
            #B,256,64,64 -> B,320,64,64
            ZeroConvLayer(inFeatures = 256,outFeatures=320,
                          kernelSize=3,stride = 1,
                          padding = 1,bias=True),
            
        ])
   
    def forward(self,input:torch.Tensor)->torch.Tensor:
        # input: B,3,512,512
        # output: B,320,64,64
        for layer in self.cannyToLatent:
            input = layer(input)
        return input