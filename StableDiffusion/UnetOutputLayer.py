import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import CLIPTokenizer
from StableDiffusion.TimeEmbedding import TimeEmbedding
from StableDiffusion.Attention import MHSelfAttention
from StableDiffusion.Attention import MHCrossAttention
from StableDiffusion.UnetResidualBlock import UnetResidualBlock
from StableDiffusion.UnetGlobalCrossAttentionBlock import UnetGlobalCrossAttentionBlock


class UnetOutputLayer(nn.Module):
    def __init__(self,inChannels=320,outChannels=4):
        super(UnetOutputLayer,self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels  
        self.groupnorm = nn.GroupNorm(num_groups=32,num_channels=self.inChannels)
        self.conv = nn.Conv2d(in_channels=self.inChannels, out_channels = self.outChannels,kernel_size=3,padding = 1)
            
    def forward(self,lantentInput):
        x = lantentInput # B 320 64 64
        x = self.groupnorm(x)
        #diffusion_model.out.0  320 
        x = F.silu(x)
        x = self.conv(x) 
        # B 320 64 64 -> B 4 64 64
        #diffusion_model.out.2  4,320,3,3
        return x         
    