import torch
from torch import nn
from .DitConfig import DitConfig
'''
  "vision_config": {
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "model_type": "siglip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "num_image_tokens": 256,
    "patch_size": 14,
    "projection_dim": 2048,
    "projector_hidden_act": "gelu_fast",
    "vision_use_head": false
  },
'''

class DitPatchEmbedding(nn.Module):
    def __init__(self,ditConfig:DitConfig):
        super().__init__()
        #super().__init__()
        self.ditConfig=ditConfig
        self.imageSize = ditConfig.imageSize #64
        self.patchSize = ditConfig.patchSize #4
        self.hiddenSize = ditConfig.hiddenSize #1152
        
        self.numPatches = (self.imageSize // self.patchSize) ** 2 # (64/4) = 16  256
        self.register_buffer("positionIds",torch.arange(self.numPatches).expand(1,-1),persistent=False)
        #positionIds shape: 1,256   [[0,1,...,255]]
        self.patch_embedding = nn.Conv2d(in_channels=self.ditConfig.numChannels,
                                         out_channels=self.ditConfig.hiddenSize,
                                         kernel_size=self.ditConfig.patchSize,
                                         stride=self.ditConfig.patchSize,
                                         padding='valid')
        # BatchSize,3,224,224 -> BatchSize,1152,16,16
        self.position_embedding = nn.Embedding(num_embeddings=self.numPatches,embedding_dim=self.hiddenSize)
        # BatchSize,1152,16*16     256 1152
    
    
    def forward(self,imageBatch:torch.FloatTensor)->torch.Tensor:
        BatchSize,Channels,Height,Width = imageBatch.shape
        # (BatchSize,3,224,224)
        patchEmbeddings = self.patch_embedding(imageBatch)
        # (BatchSize,3,224,224) -> (BatchSize,1152,16,16)
        patchEmbeddings = patchEmbeddings.reshape(BatchSize,self.ditConfig.hiddenSize,self.numPatches)
        # BatchSize,1152,16,16 -> BatchSize,1152,16*16
        patchEmbeddings = patchEmbeddings.permute(0,2,1)
        positionEmbeddings = self.position_embedding(self.positionIds)
        # BatchSize,256,1152
        fusedEmbeddings = patchEmbeddings + positionEmbeddings
        
        return fusedEmbeddings
        
    
