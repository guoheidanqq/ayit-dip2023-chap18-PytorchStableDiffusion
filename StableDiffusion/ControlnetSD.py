import torch
import torch.nn as nn
import copy
from typing import Optional
from .UnetDenoise import UnetDenoise
from .ControlnetTimeEmbedding import ControlnetTimeEmbedding
from .CannyToLatent import CannyToLatent
from .ZeorConvLayer import ZeroConvLayer



class ControlnetSD(nn.Module):
    def __init__(self,unetDesnoise:UnetDenoise):
        super().__init__()
        self.unet = unetDesnoise
        self.encoderChannels = [320,320,320,320, 
                                640,640,640,
                                1280,1280,1280,1280,1280
                                ]
                            # 0,1,2,3 320 
                            #4,5,6 640
                            #7,8,9,10,11 1280
        self.input_blocks = copy.deepcopy(self.unet.encoders)
        self.middle_block = copy.deepcopy(self.unet.bottleneck)
        
        for name,param in self.unet.named_parameters():
            param.requires_grad_(False)
        
        self.zero_convs = nn.ModuleList()
        self.time_embed = ControlnetTimeEmbedding()
        self.input_hint_block = CannyToLatent()
        for channels in self.encoderChannels:
            zeroConvLayer = ZeroConvLayer(inFeatures=channels,outFeatures=channels,kernelSize=1,stride=1,padding = 0,bias=True)
            self.zero_convs.append(zeroConvLayer)
        self.middle_block_out = ZeroConvLayer(inFeatures=1280,outFeatures=1280,kernelSize=1,stride=1,padding = 0,bias=True)
    
    
    def forward(self,latentInput:torch.Tensor,contextInput:torch.Tensor,
                timeStep:torch.Tensor,
                controlHint:Optional[torch.Tensor]=None)->torch.Tensor:
        # timeStep  B,
        # input B,4,64,64
        # output B,320,64,64
        latent = latentInput 
        context = contextInput
    
        timeEmb1280 = self.time_embed(timeStep)
        cannyLatent = self.input_hint_block(controlHint)
        
        skipConnections = []
        for i,layer in enumerate(self.input_blocks):
            if controlHint is not None and i == 0:
                latent = layer(latent,context,timeEmb1280)
                latent = latent + cannyLatent
                print(f'latent shape {latent.shape} cannyLatent shape {cannyLatent.shape}')
                controlHint = None
            else:
                latent = layer(latent,context,timeEmb1280)
            currentZeroConv = self.zero_convs[i]
            zeroOutput = currentZeroConv(latent)
            skipConnections.append(zeroOutput)
        
        latent = self.middle_block(latent,context,timeEmb1280)
        zeroOuput = self.middle_block_out(latent)
        skipConnections.append(zeroOuput)
           
        return skipConnections