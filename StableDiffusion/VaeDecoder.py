import torch
import torch.nn as nn
from StableDiffusion.VaeEncoder import VaeResidualBlock
from StableDiffusion.VaeEncoder import VaeGlobalSelfAttention


class VaeDecoder(nn.Sequential):
    def __init__(self, ):
        blocks = [  #increase dimension 4->512
                    nn.Conv2d(in_channels=4,out_channels=4,kernel_size = 1,padding=0),
                    # B 4 64 64 ->B 4 64 64
                    #post_conv_out 4,4,1,1
                    
                    nn.Conv2d(in_channels=4,out_channels=512,kernel_size = 3,padding=1), 
                    # B 4 64 64 ->B 512 64 64
                    #docoder.con_in 512,4,3,3
                    
                    # add self attention layer
                    VaeResidualBlock(inChannels=512,outChannels=512),
                    #mid.block_1
                    VaeGlobalSelfAttention(numHeads=1,embeddingDimensions=512),
                    #mid.attn_1                    
                    # four residual blocks   # B 512 64 64 ->B 512 64 64
                    VaeResidualBlock(inChannels=512,outChannels=512),
                    #mid.block_2
                    
                    
                    VaeResidualBlock(inChannels=512,outChannels=512),
                    #decoder.up.3.block.0
                    VaeResidualBlock(inChannels=512,outChannels=512),
                    #decoder.up.3.block.1
                    VaeResidualBlock(inChannels=512,outChannels=512),
                    #decoder.up.3.block.2                    
                    # scale up sampling  # B 512 64 64 ->B 512 128 128
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
                    #decoder.up.3.upsample.conv
                    
                    
                    VaeResidualBlock(inChannels=512,outChannels=512),
                    #decoder.up.2.block.0   
                    VaeResidualBlock(inChannels=512,outChannels=512),
                    #decoder.up.2.block.1
                    VaeResidualBlock(inChannels=512,outChannels=512),
                    #decoder.up.2.block.2
                    #  scale up sampling  # B 512 128 128 ->B 256 256 256
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
                    #decoder.up.2.upsample.conv
                    
                    
                    VaeResidualBlock(inChannels=512,outChannels=256),
                    #decoder.up.1.block.0
                    VaeResidualBlock(inChannels=256,outChannels=256),
                    #decoder.up.1.block.1
                    VaeResidualBlock(inChannels=256,outChannels=256),  
                    #decoder.up.1.block.2
                    # scale up sampling  # B 256 256 256 ->B 128 512 512
                    nn.Upsample(scale_factor=2,mode='nearest'),
                    nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
                    #decoder.up.1.upsample.conv
                    
                    
                    VaeResidualBlock(inChannels=256,outChannels=128),
                    #decoder.up.0.block.0
                    VaeResidualBlock(inChannels=128,outChannels=128),
                    #decoder.up.0.block.1
                    VaeResidualBlock(inChannels=128,outChannels=128),
                    #decodr.up.0.block.2
                    
                    
                    # group norm and silu
                    nn.GroupNorm(num_groups=32, num_channels=128),
                    #decoder.norm_out
                    nn.SiLU(),
                    nn.Conv2d(in_channels=128,out_channels=3,kernel_size=3,padding=1),
                    #decoder.con_out 3,128,3,3
                    
                    
        ]
        super(VaeDecoder, self).__init__(*blocks)
    
    
    def forward(self,inputs):
        # inputs: B 4 64 64
        x = inputs 
        x = x /0.18125
        for name,layer in self._modules.items():
            x = layer(x)
            #print(name,layer,x.shape)
        # x : B 3 512 512
        return x
    