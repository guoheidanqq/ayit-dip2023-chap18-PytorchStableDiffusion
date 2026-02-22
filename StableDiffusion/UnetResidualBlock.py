import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import CLIPTokenizer
from StableDiffusion.TimeEmbedding import TimeEmbedding
from StableDiffusion.Attention import MHSelfAttention
from StableDiffusion.Attention import MHCrossAttention


class UnetResidualBlock(nn.Module):
    def __init__(self,inChannels,outChannels,timeEmbeddingDimension = 1280):
        super(UnetResidualBlock,self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.timeEmbeddingDimension = timeEmbeddingDimension
        self.silu = nn.SiLU()
        self.groupnorm_feature = nn.GroupNorm(num_groups=32, num_channels=self.inChannels)
        #diffusion_model.input_blocks.0.0.in_layers.0
        self.conv_feature = nn.Conv2d(in_channels=self.inChannels,out_channels=self.outChannels,kernel_size=3,padding=1)
        #diffusion_model.input_blocks.0.0.in_layers.2
        
        self.linear_time = nn.Linear(in_features=self.timeEmbeddingDimension,out_features=self.outChannels)
        # diffusion_model.inputs.0.0.emb_layers.1   out,1280
        
        self.groupnorm_merged = nn.GroupNorm(num_groups=32, num_channels=self.outChannels)
        #diffusion_model.middle_block.0.out_layers.0
        self.conv_merged = nn.Conv2d(in_channels=self.outChannels,out_channels=self.outChannels,kernel_size=3,padding=1)
        #diffusion_model.middle_block.0.out_layers.3
        
        self.residual_layer = None
        if self.inChannels == self.outChannels:
            self.residual_layer = nn.Identity()
            
        if self.inChannels != self.outChannels:
            self.residual_layer = nn.Conv2d(in_channels=self.inChannels,out_channels=self.outChannels,kernel_size=1,padding=0)
            #diffusion_model.output_blocks.7.0.skip_connection.weight out,in,3,3
    
    def forward(self,inputs,time1280):
        # the time is 1280,then  is linear_time  to the inputs channels
        x = inputs  #  B inChannels H W   
        residual = x 
        #B inChannels H W 

        x  = self.groupnorm_feature(x)
        x =  self.silu(x)
        x = self.conv_feature(x)
        #B inChannels H W -> B outChannels H W
        
        timeX  = time1280  #  1 1280  
        timeX = self.silu(timeX)
        timeX = self.linear_time(timeX) # 1 1280 -> B outChannels 
        timeX = timeX[:,:,None,None]   # 1 outChannels 1 1 
        
        mergedX= x + timeX 
        mergedX = self.groupnorm_merged(mergedX)
        mergedX = self.silu(mergedX)
        mergedX = self.conv_merged(mergedX)
        #B outChannels H W -> B outChannels H W
        
        
        residualOut = self.residual_layer(residual)
        mergedX = mergedX + residualOut       
        return mergedX

