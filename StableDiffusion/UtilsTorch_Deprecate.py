import torch
import numpy as np
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from torchvision import transforms
class UtilsTorch:
    
    @staticmethod
    def getTimeEmbedding(timeStep):
        t = timeStep
        frequency = np.arange(0,160) 
        frequency = -frequency/160
        frequency = 10000**frequency    
        xk = t *frequency 
        timeEmbeddingCos = np.cos(xk)
        timeEmbeddingSin = np.sin(xk)
        timeEmbedding = np.concatenate([timeEmbeddingCos,timeEmbeddingSin])
        timeEmbedding = timeEmbedding[None,:]   
        return timeEmbedding
    
    @staticmethod
    def rescaleImageRange(inputImage,oldRange,newRange,isClamp = False):
        oldMin,oldMax = oldRange
        newMin,newMax = newRange
        outputImage = (inputImage - oldMin) * ((newMax - newMin) / (oldMax - oldMin)) + newMin
        if isClamp == True:
            outputImage = np.clip(outputImage,a_min=newMin, a_max=newMax)
        return outputImage
    
    
    @staticmethod
    def loadImage(filePath:str)->torch.Tensor:

        img1 = cv2.imread(filePath)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        preprocess = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((512,512)),
                        transforms.ToTensor(),   # (0 255) -> (-1, 1)
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])# 
                        ])
        imgTensor = preprocess(img1)
        imgBatch = imgTensor[None, :, :, :]
        print(imgBatch.shape)
        plt.imshow(img1)
        plt.show()     
    
    
       
                