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
from StableDiffusion.UpScaleTwo import UpScaleTwo



class SequentialAdapter(nn.Sequential):
    def __init__(self,*modules):
        super(SequentialAdapter,self).__init__(*modules)
    
    def forward(self,latentX,contextY,timeStep1280):
        x = latentX # B C H W 
        y = contextY  # B N D 
        time = timeStep1280  # B 1280
        
        for module in self._modules.values():
            if isinstance(module,UnetResidualBlock):
                x = module(x,time)
            elif isinstance(module,UnetGlobalCrossAttentionBlock):
                x = module(x,y)
            else: 
                x = module(x)
        
        return x
                