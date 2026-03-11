import torch
import torch.nn as nn
from typing import Optional
from .UnetDenoise import UnetDenoise
from .TimeEmbedding import TimeEmbedding
from .ControlnetSDUnet import ControlnetSDUnet
from .ControlnetSD import ControlnetSD
from .UnetOutputLayer import UnetOutputLayer
from .Utils import Utils

class DiffusionProcessControlnet(nn.Module):
    def __init__(self,controlnetUnet:Optional[ControlnetSDUnet]=None,controlnetOutputs:Optional[ControlnetSD]=None):
        super().__init__()
        self.time_embedding = TimeEmbedding(embeddingDimension=320)
        self.unet = UnetDenoise()
        self.final = UnetOutputLayer(inChannels=320,outChannels=4)
        for name,param in self.unet.named_parameters():
            param.requires_grad = False
        
        for name,param in self.time_embedding.named_parameters():
            param.requires_grad = False
        
        for name,param in self.final.named_parameters():
            param.requires_grad = False
        self.controlUnet = None
        self.controlOutput = None
        if controlnetUnet is not None:
            self.controlUnet = controlnetUnet
        if controlnetOutputs is not None:
            self.controlOutput = controlnetOutputs

    
    def forward(self,latentInput:torch.Tensor,
                contextInput:torch.Tensor,
                timeSteps:torch.Tensor,
                controlHint:torch.Tensor)->torch.Tensor:
        # latentX B,4,64,64
        # contextY B,77,768
        # timeStep320 1,320
        #controlHint B,3,512,512
        #ouput shape B,4,64,64
        device = next(self.parameters()).device
        latentX = latentInput
        contextY = contextInput
        print(f'lantex.shape after unet {latentX.shape}')
        controlOutputs = self.controlOutput(latentX,contextY,timeSteps,controlHint)
        print(f'controlOutputs.length {len(controlOutputs)}')
        latentX = self.controlUnet(latentX,contextY,timeSteps,controlOutputs)
        #predicted noise B,4,64,64
        latentX = self.final(latentX)
        #output predicted latent noise B,4,64,64
        return latentX