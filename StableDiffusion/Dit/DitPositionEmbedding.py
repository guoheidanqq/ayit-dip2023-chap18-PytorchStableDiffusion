import torch
from .DitConfig import DitConfig


class DitPositionEmbedding:
    def __init__(self, ditConfig:DitConfig):
        self.hiddenSize = ditConfig.hiddenSize
        self.imageSize = ditConfig.imageSize
        self.hiddenSize = ditConfig.hiddenSize
        
    
    
    
    
    
    def get1DPositionEncoding(self) -> torch.Tensor:
        pass
    
    def get2DPositionEncoding(self) -> torch.Tensor:
       
       
    