
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import CLIPTokenizer
from StableDiffusion.TimeEmbedding import TimeEmbedding
from StableDiffusion.Attention import MHSelfAttention
from StableDiffusion.Attention import MHCrossAttention
from StableDiffusion.UnetDenoise import UnetDenoise,UnetOutputLayer



class DiffusionProcess(nn.Module):
    def __init__(self):
        super(DiffusionProcess,self).__init__()
        self.time_embedding = TimeEmbedding(embeddingDimension=320)
        self.unet = UnetDenoise()
        self.final = UnetOutputLayer(inChannels=320,outChannels=4)
    
    def forward(self,latentInput:torch.Tensor,contextInput:torch.Tensor,timeStep320:torch.Tensor)->torch.Tensor:
        # latentX B,4,64,64
        # contextY B,77,768
        # timeStep320 1,320
        latentX = latentInput
        contextY = contextInput
        timeStep1280 = self.time_embedding(timeStep320)
        #timeStep1280 B,320->B,1280
        latentX = self.unet(latentX,contextY,timeStep1280)
        #predicted noise B,4,64,64
        latentX = self.final(latentX)
        #output predicted latent noise B,4,64,64
        return latentX