import torch
import torch.nn as nn
from .UnetDenoise import UnetDenoise
from .TimeEmbedding import TimeEmbedding
from .ControlnetSDUnet import ControlnetSDUnet
from .ControlnetSD import ControlnetSD
from .UnetOutputLayer import UnetOutputLayer
from .Utils import Utils

class DiffusionProcessControlnet(nn.Module):
    def __init__(self,unetDesnoise:UnetDenoise):
        super().__init__()
        self.time_embedding = TimeEmbedding(embeddingDimension=320)
        self.unet = unetDesnoise
        self.controlUnet = ControlnetSDUnet(unetDesnoise)
        self.controlOutput = ControlnetSD(unetDesnoise)        
        self.final = UnetOutputLayer(inChannels=320,outChannels=4)
    
    def forward(self,latentInput:torch.Tensor,contextInput:torch.Tensor,timeSteps:torch.Tensor,controlHint:torch.Tensor)->torch.Tensor:
        # latentX B,4,64,64
        # contextY B,77,768
        # timeStep320 1,320
        #controlHint B,3,512,512
        #ouput shape B,4,64,64
        device = next(self.parameters()).device
        latentX = latentInput
        contextY = contextInput
        
        timeStepsEmb320 = Utils.getTimeEmbeddingBatchTorch(timeSteps,device=device)
        #print(f'latentX.dtype {latentX.dtype} contextY.dtype {contextY.dtype}timeStep320.dtype {timeStepsEmb320.dtype} controlHint.dtype {controlHint.dtype}')
        #print(f'latentX.shape {latentX.shape} contextY.shape {contextY.shape} timeStep320.shape {timeStepsEmb320.shape} controlHint.shape {controlHint.dtype}')
        timeStep1280 = self.time_embedding(timeStepsEmb320)
        
        #print(f'latentX.dtype {latentX.dtype} contextY.dtype {contextY.dtype}timeStep1280.dtype {timeStep1280.dtype} controlHint.dtype {controlHint.dtype}')
        #print(f'latentX.shape {latentX.shape} contextY.dtype {contextY.shape}timeStep1280.shape {timeStep1280.shape} controlHint.dtype {controlHint.shape}')
        #timeStep1280 B,320->B,1280
        originUnetOut = self.unet(latentX,contextY,timeStep1280)
        print(f'lantex.shape after unet {latentX.shape}')
        controlOutputs = self.controlOutput(latentX,contextY,timeSteps,controlHint)
        print(f'controlOutputs.length {len(controlOutputs)}')
        latentX = self.controlUnet(latentX,contextY,timeStep1280,controlOutputs)
        #predicted noise B,4,64,64
        latentX = self.final(latentX)
        #output predicted latent noise B,4,64,64
        return latentX