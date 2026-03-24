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
        self.layerNorm = nn.LayerNorm(self.hiddenSize, elementwise_affine = False,eps=ditConfig.layerNormEps)
        self.projectToLatent = nn.Linear(self.hiddenSize,self.outFeatures,bias=True)
        
        self.adaptiveLinearNorm = nn.Sequential(nn.SiLU(),
                                    nn.Linear(self.hiddenSize,self.hiddenSize * 2,bias = True))
    def modulate(self,x:torch.Tensor,scaleControl:torch.Tensor,shiftControl:torch.Tensor)->torch.Tensor:
        # x:B,N,D 
        # scaleControl: B,D 
        # shiftControl: B,D 
        scaleControl = scaleControl[:,None,:]
        shiftControl = shiftControl[:,None,:]
        x = x*(1.0 + scaleControl) + shiftControl
        return x
        
    
    
    def forward(self,input:torch.Tensor,classEmbedBatch:torch.Tensor,timeEmbedBatch:torch.Tensor)->torch.Tensor:
        #input B,256,1152
        #output B,4,64,64
        
        controlBatch = classEmbedBatch + timeEmbedBatch # B  1152
        controlBatch = self.adaptiveLinearNorm(controlBatch) # B  1152
        mlpScale,mlpShift = controlBatch.chunk(2,dim=1)
        
        latent = input
        Batch,SeqLen,HiddenSize =input.shape
        latent = self.layerNorm(latent)
        latent = self.modulate(latent,mlpScale,mlpShift)
        latent = self.projectToLatent(latent) #B,256,4*4*4
        latent = latent.reshape(Batch,self.patchesHeightNum,self.patchesWidthNum,self.numChannels,self.patchHeight,self.patchWidth)
        #B,pH,pW,C,h,w -> B,C,pnH,pW,h,w ->B,C,pnH,h,pnW,w->B,C,pnH,pnW,h,w->B,C,pnH,h,pnW,w
        #0,1,2,3,4,5->     0,3,1,4,2,5
        latent = latent.permute(0,3,1,4,2,5)
        hiddenHeight = self.patchesHeightNum * self.patchHeight
        hiddenWidth = self.patchesWidthNum * self.patchWidth
        latent = latent.reshape(Batch,self.numChannels,hiddenHeight,hiddenWidth)
        return latent