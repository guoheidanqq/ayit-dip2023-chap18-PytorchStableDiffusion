import torch
import torch.nn as nn
from .DitConfig import DitConfig

class DitUnPatchEmbedding(nn.Module):
    def __init__(self,ditConfig:DitConfig):
        super().__init__()
        self.ditConfig=ditConfig
        self.numChannels = ditConfig.numChannels #4
        self.imageSize = ditConfig.imageSize #64
        self.patchSize = ditConfig.patchSize #4
        self.patchHeight =self.patchSize 
        self.patchWidth = self.patchSize
        self.hiddenSize = ditConfig.hiddenSize #1152       
        self.patchesHeightNum = self.imageSize // self.patchSize
        self.patchesWidthNum = self.imageSize // self.patchSize 
        self.numPatches = (self.imageSize // self.patchSize) ** 2 
        # (64/4) = 16  256
        self.outFeatures = self.numChannels * (self.patchSize ** 2)
        self.projectToLatent = nn.Linear(self.hiddenSize,self.outFeatures,bias=True)
    
    
    def forward(self,input:torch.Tensor)->torch.Tensor:
        #input B,256,1152
        #output B,4,64,64
        x = input
        Batch,SeqLen,HiddenSize =input.shape
        latent = self.projectToLatent(x) #B,256,4*4*4
        latent = latent.reshape(Batch,self.patchesHeightNum,self.patchesWidthNum,self.numChannels,self.patchHeight,self.patchWidth)
        #B,pH,pW,C,h,w -> B,C,pnH,pW,h,w ->B,C,pnH,h,pnW,w->B,C,pnH,pnW,h,w->B,C,pnH,h,pnW,w
        #0,1,2,3,4,5->     0,3,1,4,2,5
        latent = latent.permute(0,3,1,4,2,5)
        hiddenHeight = self.patchesHeightNum * self.patchHeight
        hiddenWidth = self.patchesWidthNum * self.patchWidth
        latent = latent.reshape(Batch,self.numChannels,hiddenHeight,hiddenWidth)
        return latent