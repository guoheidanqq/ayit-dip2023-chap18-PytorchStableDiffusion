import torch
import torch.nn as nn
from typing import Tuple
from .UnetDenoise import UnetDenoise
from .TimeEmbedding import TimeEmbedding
from .Utils import Utils



class ControlnetSDUnet(nn.Module):
    def __init__(self,unetDesnoise:UnetDenoise):
        super().__init__()
        self.unet = unetDesnoise
        self.time_embedding = TimeEmbedding(embeddingDimension=320)
        for name,param in self.unet.named_parameters():
            param.requires_grad_(False)
    
    
    def forward(self,latentInput:torch.Tensor,contextInput:torch.Tensor,
                timeSteps:torch.Tensor,controlnetOutputs:Tuple[torch.Tensor])->torch.Tensor:
        # inputs shape: B,4,64,64
        # output shape: B,320,64,64
        with torch.no_grad():
            device = next(self.parameters()).device
            latent = latentInput
            context = contextInput
            timeSteps320 = Utils.getTimeEmbeddingBatchTorch(timeSteps,device=device)
            timeStep1280 = self.time_embedding(timeSteps320)
            time = timeStep1280
            skipConnections = []
            for i,layer in enumerate(self.unet.encoders):
                latent = layer(latent,context,time)
                #print(f'encoder layer {i} ,latent shape {latent.shape} ')       
                skipConnections.append(latent)
            
            latent = self.unet.bottleneck(latent,context,time)
            #print(f'bottle neck layer {12} ,latent shape {latent.shape} ')  
            if controlnetOutputs is not None:
                controlOut = controlnetOutputs.pop()
                latent = latent + controlOut
            
            for i,layer in enumerate(self.unet.decoders):
                skipLatent = skipConnections.pop()
                if controlnetOutputs is not None:
                    controlOut = controlnetOutputs.pop()
                    skipLatent = skipLatent + controlOut
                latent = torch.cat([latent,skipLatent],dim=1)
                latent = layer(latent,context,time)
                #print(f'decoder layer {11-i} ,latent shape {latent.shape} ')  
        
        
        return latent