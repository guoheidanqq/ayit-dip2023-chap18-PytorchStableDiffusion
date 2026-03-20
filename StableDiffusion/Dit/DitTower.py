import torch
from torch import nn
from .DitConfig import DitConfig
from .DitTransformer import DitTransformer
from .DitPatchEmbedding import DitPatchEmbedding
from .DitUnPatchEmbedding import DitUnPatchEmbedding


class DitTower(nn.Module):

    def __init__(self, ditConfig:DitConfig):
        super().__init__()
        #super().__init__()
        self.ditConfig = ditConfig
        self.vision_model = DitTransformer(ditConfig)
        self.unPatchEmbedding = DitUnPatchEmbedding(ditConfig)
        self.patchEmbedding = DitPatchEmbedding(ditConfig)

    
    
    def forward(self, imgBatch:torch.Tensor)->torch.Tensor:
        #inputs:latent of vaeencoder  Batchsize 4 64 64
        #outputs:predicted noise  Batchsize 4 64 64
        inputImgBatch = imgBatch
        hiddenStates = self.patchEmbedding(inputImgBatch)
        hiddenStates = self.vision_model(hiddenStates)
        hiddenStates = self.unPatchEmbedding(hiddenStates) 
        
        return hiddenStates