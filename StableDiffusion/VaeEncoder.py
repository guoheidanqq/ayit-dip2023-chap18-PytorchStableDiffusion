import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import CLIPTokenizer
from typing import Optional
from StableDiffusion.Attention import MHSelfAttention


class VaeResidualBlock(nn.Module):
    def __init__(self,inChannels,outChannels):
        super(VaeResidualBlock,self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.groupnorm_1 = nn.GroupNorm(num_groups=32, num_channels=self.inChannels,affine = True)
        # params numChannels weight + numChannels bias 
        self.conv_1 = nn.Conv2d(in_channels=self.inChannels,out_channels=self.outChannels,kernel_size=3,padding=1)
        
        self.groupnorm_2 = nn.GroupNorm(num_groups=32, num_channels=self.outChannels,affine = True)
        self.conv_2 = nn.Conv2d(in_channels= self.outChannels,out_channels= self.outChannels,kernel_size = 3, padding = 1)
        
        self.residual_layer = None 
        if self.inChannels == self.outChannels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels=self.inChannels,out_channels=self.outChannels,kernel_size=1,padding=0)
    
    def forward(self,inputs):
        x = inputs    # B C H W
        residual =  x 
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)  # B inChannels H W ->B outChannels H W
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)  # B outChannels H W -> B outChannels H W
        x = x + self.residual_layer(residual)
        
        return x



class VaeGlobalSelfAttention(nn.Module):
    def __init__(self,numHeads=1,embeddingDimensions=512,isCausalMask = False):
        super(VaeGlobalSelfAttention,self).__init__()
        #B 64*64 512
        self.numHeads = numHeads
        self.embeddingDimensions = embeddingDimensions
        self.isCausalMask = isCausalMask
        self.groupnorm =  nn.GroupNorm(num_groups=32, num_channels=self.embeddingDimensions)
        self.attention = MHSelfAttention(numHeads = self.numHeads,
                                         embeddingDimensions=self.embeddingDimensions,
                                         isCausalMask=False,
                                         isInProjBias = True,
                                         isOutProjBias = True)
        # in_proj 1536 512
        # out_proj 512 512
        # attention with bias
    def forward(self,inputs):
        x = inputs  # Batch cLatent hLatent wLatent ->B 512 64 64 
        residual = x 
        N,C,H,W = inputs.shape   # B 512 64 64
        x = self.groupnorm(x)
        x = x.reshape(N,C,H*W)   # B 512 4096
        x = x.permute(0,2,1)     # B 4096 512
        x = self.attention(x)   # B 4096 512
        x = x.permute(0,2,1)   # B 512 4096
        x = x.reshape(N,C,H,W)   # B 512 64 64
        x = x + residual 
        return x 
    



class VaeEncoder(nn.Sequential):
    def __init__(self):        
        blocks = [
                 nn.Conv2d(in_channels=3,out_channels=128,kernel_size=3,stride=1,padding=1),  
                 # B 3 512 512 -> B 128 512 512
                 # down_blocks.0.downsamplers.0.conv_in
                 VaeResidualBlock(inChannels=128,outChannels=128),
                 # B 128 512 512->B 128 512 512
                 # down_blocks.0.resnets.0 
                 VaeResidualBlock(inChannels=128,outChannels=128), 
                 #  B 128 512 512->B 128 512 512
                 #down_blocks.0.resnets.1
                 
                 

                 nn.Conv2d(in_channels=128,out_channels = 128,kernel_size = 3,stride = 2 ,padding = 0),  # B 128 512 512 ->B 128 256 256
                 # down_blocks.1.downsamplers.0.conv_in
                 # down sampling  B,C,H/2, W/2
                 #B 128 512 512 ->B 256 256 256
                 VaeResidualBlock(inChannels=128,outChannels=256),
                 #down_blocks.1.resnets.0
                 VaeResidualBlock(inChannels=256,outChannels=256),
                 #down_blocks.1.resnets.1
                 
                 
                 nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=0),
                 # down sampling B,C,H/4, W/4
                 # down_blocks.2.downsamplers.0.conv_in
                 # B 256 256 256 ->B 256 128 128
                 VaeResidualBlock(inChannels=256,outChannels=512),
                 #down_blocks.2.resnets.0
                 VaeResidualBlock(inChannels=512,outChannels=512 ),   # B 512 128 128 ->B 512 128 128
                 #down_blocks.2.resnets.1
                 
                 
                 nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=2,padding=0), #B 512 128 128 ->B 512 64 64
                 # down sampling B,C,H/8, W/8
                 # down_blocks.3.downsamplers.0.conv_in
                 VaeResidualBlock(inChannels=512,outChannels=512),
                 #down_blocks.3.resnets.0
                 VaeResidualBlock(inChannels=512,outChannels=512),
                 #down_blocks.3.resnets.1
                 
                 
                 
                 
                 VaeResidualBlock(inChannels=512,outChannels=512),
                 #mid.block_1     
                 #  add attention layer
                 VaeGlobalSelfAttention(numHeads=1,embeddingDimensions=512),                  
                 # B 512 64 64 ->B 512 64 64
                 # midblock.attn_1
                 VaeResidualBlock(inChannels=512,outChannels=512),
                 #mid.block_2
                 # nonlinear layer
                 
                 
                 nn.GroupNorm(num_groups=32, num_channels=512),
                 #encoder.norm_out
                 nn.SiLU(),                 
                 # reduce dimension
                 nn.Conv2d(in_channels=512,out_channels=8,kernel_size=3,padding=1), 
                 # B 512 64 64 ->B 8 64 64
                 # encoder.conv_out  8 512 1 1
                 
                 nn.Conv2d(in_channels=8,out_channels=8,kernel_size=1,padding=0)
                 # encoder.quant_conv 8 8 1 1
                 # B 8 64 64 ->B 8 64 64
                 # will chunk to mean B,4,64,64 and variance B,4,64,64
                 
                                  
                 ]
        super(VaeEncoder,self).__init__(*blocks)
    
    
    def forward(self,inputs:torch.Tensor,inputNoise:Optional[torch.Tensor]=None)->torch.Tensor:
        x = inputs    # B 8 64 64 
        #print(f'inputs shape {x.shape}')
        if inputNoise is not None:
            noise = inputNoise  # 1 8 64 64
            print(f'vae encoder input noise.shape {noise.shape}')
        else:
            noise =  torch.randn(1,4,64,64).to(inputs.device)
            print(f'vae encoder input noise is none  use zeros')
        #print(f'noise shape {noise.shape}')
        for name,layer in self._modules.items():            
            if isinstance(layer,nn.Conv2d) and layer.stride ==(2,2):
                x = F.pad(x,(0,1,0,1))
            x = layer(x)
            #print(name,layer,x.shape)
        
        mean,logVariance = torch.chunk(x,chunks=2,dim = 1)
        logVarianceClamp = torch.clamp(logVariance,-30,20)
        varianceClamp = torch.exp(logVarianceClamp)
        stdVarianceClamp = torch.sqrt(varianceClamp)
        #print(f'mean shape {mean.shape}')
        #print(f'variance shape {varianceClamp.shape}')      
        
        latentImageNoised =  mean + stdVarianceClamp * noise 
        latentImageNoised = latentImageNoised * 0.18125
        #encoder output B 4,64,64
               
        return latentImageNoised