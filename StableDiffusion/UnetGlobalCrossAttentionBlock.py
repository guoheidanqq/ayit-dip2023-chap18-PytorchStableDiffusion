import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import CLIPTokenizer
from StableDiffusion.TimeEmbedding import TimeEmbedding
from StableDiffusion.Attention import MHSelfAttention
from StableDiffusion.Attention import MHCrossAttention
from StableDiffusion.UnetResidualBlock import UnetResidualBlock

class UnetGlobalCrossAttentionBlock(nn.Module):
    def __init__(self,numHeads=8,latentEmbeddingDimension=1280,contextEmbeddingDimension=768):
        super(UnetGlobalCrossAttentionBlock,self).__init__()
        self.numHeads = numHeads
        self.latentEmbeddingDimension = latentEmbeddingDimension
        self.headDimension = self.latentEmbeddingDimension // self.numHeads
        self.contextEmbeddingDimension = contextEmbeddingDimension
        
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=self.latentEmbeddingDimension,eps=1e-6)
        self.conv_input = nn.Conv2d(in_channels = self.latentEmbeddingDimension,out_channels=self.latentEmbeddingDimension,kernel_size=1,padding=0)
        #diffusion_model.inputs_blocks.1.1.proj_in 
        
        self.layernorm_1 =  nn.LayerNorm(self.latentEmbeddingDimension)
        #diffusion_model.input_blocks.1.1.transformer_blocks.0.norm1
        self.attention_1 = MHSelfAttention(numHeads=self.numHeads,embeddingDimensions=self.latentEmbeddingDimension,isInProjBias=False,isOutProjBias=True)
        
        self.layernorm_2 = nn.LayerNorm(self.latentEmbeddingDimension)
        #diffusion_model.input_blocks.1.1.transformer_blocks.0.norm2
        self.attention_2 = MHCrossAttention(numHeads=self.numHeads,latentEmbeddingDimension=self.latentEmbeddingDimension,\
                                            contextEmbeddingDimension=self.contextEmbeddingDimension,isInProjBias=False,isOutProjBias=True)
        
        self.layernorm_3 = nn.LayerNorm(self.latentEmbeddingDimension)
        #diffusion_model.input_blocks.1.1.transformer_blocks.0.norm3
        self.linear_geglu_1 = nn.Linear(in_features=self.latentEmbeddingDimension,out_features=self.latentEmbeddingDimension *4 * 2)
        #diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj   8x,x
        self.linear_geglu_2 = nn.Linear(in_features=4*self.latentEmbeddingDimension,out_features=self.latentEmbeddingDimension)
        #diffusion_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.proj   x,4x
        self.conv_output = nn.Conv2d(in_channels = self.latentEmbeddingDimension,out_channels=self.latentEmbeddingDimension,kernel_size=1,padding=0)
        #diffusion_model.inputs_blocks.1.1.proj_out 
    
    def forward(self,latentInput,contextInput):
        latentX = latentInput   #  B C H W 
        BatchLatent,C,H,W = latentX.shape
        contextY = contextInput  # B N D
        BatchContext,N,D = contextY.shape
        
        
        residual = latentX 
        
        latentX = self.groupnorm(latentX)   
        latentX = self.conv_input(latentX)  # B C H W 
        
        #print(f'unet global cross attention block foward pass {latentX.shape} {contextInput.shape}')
        latentX = latentX.reshape(BatchLatent,C,H*W)   # B C H*W 
        #print(f'unet global cross attention block foward pass her{latentX.shape}')
        latentX = latentX.permute(0,2,1)  # B H*W C
        #latentX = latentX.view(B, C, H * W).permute(0, 2, 1).contiguous()
        
        residualSelfAttention = latentX 
        latentX = self.layernorm_1(latentX)
        latentX = self.attention_1(latentX)
        latentX = residualSelfAttention + latentX
        
        residualCrossAttention = latentX  
        latentX = self.layernorm_2(latentX)
        latentX = self.attention_2(latentX,contextY)        
        latentX = latentX + residualCrossAttention
        
        residualGeGlu = latentX 
        latentX = self.layernorm_3(latentX)
        latentX = self.linear_geglu_1(latentX)  # B H*W C*4*2 
        latentX,latentXGate = latentX.chunk(chunks = 2, dim = -1 )
        latentX  = latentX * F.gelu(latentXGate)   # B H*W 4*C   
        latentX = self.linear_geglu_2(latentX)  # B H*W C 
        latentX = latentX + residualGeGlu
        
        latentX = latentX.permute(0,2,1)
        latentX = latentX.reshape(BatchLatent,C,H,W)
        #latentX = latentX.permute(0, 2, 1).contiguous().view(B, C, H, W)
        
        latentX = self.conv_output(latentX)
        latentX = latentX+ residual
        
        return latentX