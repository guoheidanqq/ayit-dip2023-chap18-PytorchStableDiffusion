import torch 
from torch import nn
from .DitConfig import DitConfig
from .DitEncoder import DitEncoder
from .DitPatchEmbedding  import DitPatchEmbedding

class DitTransformer(nn.Module):
    def __init__(self,ditConfig:DitConfig):
        super().__init__()
        #super().__init__()
        self.ditConfig = ditConfig
        #self.embeddings = DitPatchEmbedding(ditConfig) # B 3 224 224 -> B 256 1152
        self.encoder = DitEncoder(ditConfig) # B 256 1152 -> B 256 1152  layers 27
        self.post_layernorm = nn.LayerNorm(ditConfig.hiddenSize, eps=ditConfig.layerNormEps)
        
    
    
    def forward(self,imgBatch:torch.Tensor)-> torch.Tensor:
        #  Batchsize 3 224  224
        inputImgBatch = imgBatch
        #hiddenStates = self.embeddings(inputImgBatch)
        #B 3 224 224 -> B 256 1152
        hiddenStates = self.encoder(inputImgBatch)
        #B 256 1152 -> B 256 1152
        hiddenStates = self.post_layernorm(hiddenStates)
        #B 256 1152 -> B 256 11152
        return hiddenStates