import torch
import torch.nn as nn
from typing import Optional
from .UnetDenoise import UnetDenoise
from .TimeEmbedding import TimeEmbedding
from .ControlnetSDUnet import ControlnetSDUnet
from .ControlnetSD import ControlnetSD
from .UnetOutputLayer import UnetOutputLayer
from .Utils import Utils
from .ControlnetModelConverter import ControlnetModelConverter

class DiffusionProcessControlnet(nn.Module):
    def __init__(self):
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
        self.controlnetOutput = None
        self.controlnetUnet = None
    
    
    def loadControlnetWeightsDict(self,controlWeightsDict:dict):        
        if controlWeightsDict is not None:
            device = next(self.parameters()).device            
            self.controlnetOutput = ControlnetSD(self.unet).to(device)
            newControlWeightsDict = ControlnetModelConverter(controlWeightsDict)
            misssingKeys, unexpectedKeys = self.controlnetOutput.load_state_dict(newControlWeightsDict,strict=False)
            #print(f'misssingKeys {misssingKeys}, unexpectedKeys {unexpectedKeys}')
            self.controlnetUnet = ControlnetSDUnet(self.unet,self.time_embedding).to(device)



    
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
        #print(f'lantex.shape after unet {latentX.shape}')
        controlOutputs = self.controlnetOutput(latentX,contextY,timeSteps,controlHint)
        #print(f'controlOutputs.length {len(controlOutputs)}')
        latentX = self.controlnetUnet(latentX,contextY,timeSteps,controlOutputs)
        #predicted noise B,4,64,64
        latentX = self.final(latentX)
        #output predicted latent noise B,4,64,64
        return latentX