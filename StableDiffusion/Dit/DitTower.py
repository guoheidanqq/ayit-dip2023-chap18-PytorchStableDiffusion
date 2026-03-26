import torch
from torch import nn
from .DitConfig import DitConfig
from .DitTransformer import DitTransformer
from .DitPatchEmbedding import DitPatchEmbedding
from .DitUnPatchEmbedding import DitUnPatchEmbedding
from .DitTimeEmbedding import DitTimeEmbedding
from .DitPositionEmbedding import DitPositionEmbedding
from .DitClassEmbedding import DitClassEmbedding



class DitTower(nn.Module):

    def __init__(self, ditConfig:DitConfig):
        super().__init__()
        #super().__init__()
        self.ditConfig = ditConfig
        self.hiddenSize = ditConfig.hiddenSize
        self.vision_model = DitTransformer(ditConfig)
        self.patchEmbedding = DitPatchEmbedding(ditConfig)
        self.unPatchEmbedding = DitUnPatchEmbedding(ditConfig)
        self.timeEmbedding = DitTimeEmbedding(ditConfig)
        self.positionEmbedding = DitPositionEmbedding(ditConfig)
        self.classEmbedding = DitClassEmbedding(ditConfig)
        

    
    
    def forward(self, latentInputBatch:torch.Tensor,classIdBatch:torch.Tensor,timeSteps:torch.Tensor)->torch.Tensor:
        #inputs:latent of vaeencoder  Batchsize 4 64 64
        #outputs:predicted noise  Batchsize 4 64 64
        device = latentInputBatch.device
        BatchSize,Channels,Height,Width = latentInputBatch.shape
        timeEmbedBatch = self.timeEmbedding(timeSteps).to(device)
        classEmbedBatch = self.classEmbedding(classIdBatch).to(device)        
        positionEmbedding = self.positionEmbedding.get2DPositionEncoding(self.hiddenSize).to(device)
        inputImgBatch = latentInputBatch
        hiddenStates = self.patchEmbedding(inputImgBatch)
        hiddenStates = hiddenStates + positionEmbedding
        hiddenStates = self.vision_model(hiddenStates,classEmbedBatch,timeEmbedBatch)
        hiddenStates = self.unPatchEmbedding(hiddenStates,classEmbedBatch,timeEmbedBatch) 
        
        return hiddenStates