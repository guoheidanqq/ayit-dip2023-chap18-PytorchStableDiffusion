import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import CLIPTokenizer
from StableDiffusion.TimeEmbedding import TimeEmbedding
from StableDiffusion.Attention import MHSelfAttention
from StableDiffusion.Attention import MHCrossAttention
from StableDiffusion.UnetResidualBlock import UnetResidualBlock
from StableDiffusion.UnetGlobalCrossAttentionBlock import UnetGlobalCrossAttentionBlock
from StableDiffusion.UnetOutputLayer import UnetOutputLayer
class UpScaleTwo(nn.Module):
    def __init__(self,inChannels):
        super(UpScaleTwo,self).__init__()
        self.inChannels = inChannels
        #self.upScaleTwo = nn.Upsample(scale_factor=2,mode='nearest')
        
        self.conv =  nn.Conv2d(in_channels=self.inChannels,out_channels=self.inChannels,kernel_size=3,padding=1)
    def forward(self,inputs):
        x = inputs
        #x  = self.upScaleTwo(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        #add 20260219 to replace nn.Upsample(scale_factor=2,mode='nearest')
        x = self.conv(x)
        return x 