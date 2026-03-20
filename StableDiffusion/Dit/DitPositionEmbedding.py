import torch
from .DitConfig import DitConfig


class DitPositionEmbedding:
    def __init__(self, ditConfig:DitConfig):
        self.hiddenSize = ditConfig.hiddenSize
        self.imageSize = ditConfig.imageSize
        self.hiddenSize = ditConfig.hiddenSize
    
    def get1DPositionEncoding(self,positionBatch:torch.Tensor,embedDims:int) -> torch.Tensor:
        p = positionBatch  # [M]
        p = p[:,None]
        D = embedDims//2
        i = torch.arange(0,D,dtype=torch.float)
        omega = -2.0*i/D
        omega = 10000.0**omega    
        omega = omega[None,:]    
        sinPart = torch.sin(p*omega)
        cosPart = torch.cos(p*omega)
        positionEmbed = torch.cat([sinPart,cosPart],dim=-1)
        return  positionEmbed
    
    def get2DPositionEncoding(self,gridSize:torch.Tensor,embedDims:int) -> torch.Tensor:
        gridH = torch.arange(gridSize,dtype = torch.float32)
        gridW = torch.arange(gridSize,dtype = torch.float32)
        grid = torch.meshgrid(gridW,gridH,indexing='xy')        
        grid = torch.stack(grid,dim=0)        
        grid = grid[:,None,:,:]
        #2,1,H,W
        grid0 = grid[0].reshape(-1) # be [H*W]
        grid1 = grid[1].reshape(-1)
        
        halfD = embedDims//2
        embH = self.get1DPositionEncoding(grid0,halfD)# [H*W,D/2]
        embW = self.get1DPositionEncoding(grid1,halfD)# [H*W,D/2]
        embHW = torch.cat([embH,embW],dim=1) # [H*W,D]
         
        return embHW
       
       
    