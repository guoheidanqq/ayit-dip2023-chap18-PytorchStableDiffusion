import torch
import torch.nn as nn
class ZeroConvLayer(nn.Module):
    def __init__(self,inFeatures:int,outFeatures:int,kernelSize=3,stride=1,padding = 1,bias=True):
        super(ZeroConvLayer,self).__init__()
        self.zeroConv = nn.Conv2d(inFeatures,outFeatures,kernel_size=kernelSize,
                                  stride = stride,padding = padding,bias=bias)
        nn.init.zeros_(self.zeroConv.weight)
        nn.init.zeros_(self.zeroConv.bias)
    
    
    def forward(self,input:torch.Tensor)->torch.Tensor:
        output = self.zeroConv(input)
        return output