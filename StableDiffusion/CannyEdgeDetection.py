import numpy as np
import torch
import cv2

class CannyEdgeDetection:
    def __init__(self,lowThreshold:int=100,highThreshold:int =200):
        self.lowThreshold = lowThreshold
        self.highThreshold = highThreshold    
    
    def __call__(self,img:np.ndarray)->torch.Tensor:
        canny = cv2.Canny(img,self.lowThreshold,self.highThreshold)
        canny = canny[:,:,None]
        cannyHW3 = np.concatenate([canny,canny,canny],axis = -1)
        cannyHW3 = cannyHW3.astype(np.float32)/255.0
        cannyTensor = torch.from_numpy(cannyHW3)
        cannyTensor = cannyTensor.permute(2,0,1)
        cannyTensor =cannyTensor[None,:,:,:] 
        cannyTensor = cannyTensor*2.0 - 1.0
        # output range [-1,1]
        
        return cannyTensor