import torch
import numpy as np
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Tuple,Union
from transformers import CLIPTokenizer
class Utils:
    
    @staticmethod
    def getTimeEmbedding(timeStep:int)->np.ndarray:
        t = timeStep
        frequency = np.arange(0,160.0) 
        frequency = -frequency/160.0
        frequency = 10000**frequency    
        xk = t *frequency 
        timeEmbeddingCos = np.cos(xk)
        timeEmbeddingSin = np.sin(xk)
        timeEmbedding = np.concatenate([timeEmbeddingCos,timeEmbeddingSin])
        timeEmbedding = timeEmbedding[None,:]   
        return timeEmbedding
    
    @staticmethod
    def rescaleImageRange(inputImage:np.ndarray,oldRange:Tuple[float,float]=[-1,1],newRange:Tuple[float,float]=[0,1],isClamp:bool = True)->np.ndarray:
        oldMin,oldMax = oldRange
        newMin,newMax = newRange
        outputImage = (inputImage - oldMin) * ((newMax - newMin) / (oldMax - oldMin)) + newMin
        if isClamp == True:
            outputImage = np.clip(outputImage,a_min=newMin, a_max=newMax)
        return outputImage
    @staticmethod
    def showBatchImage(inputImageBatch:torch.tensor):
        decoderImg = inputImageBatch
        decoderImg = decoderImg.detach().cpu().numpy()
        decoderTest = decoderImg[0,:,:,:].transpose(1,2,0)
        decoderTest =Utils.rescaleImageRange(decoderTest,[-1,1],[0,1],True)
        decoderImg = (decoderImg[0].transpose(1,2,0)+1)/2
        decoderImg = decoderImg.clip(0,1)
        plt.imshow(decoderImg)
        plt.show()
        
    @staticmethod
    def loadImageBatch(filePath:str,device='cuda')->torch.Tensor:

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
        return imgBatch.to(device)   
    
    def getPromptTokens(prompt:str,device='cuda')->Tuple[torch.LongTensor,torch.LongTensor]:
        promptTokenizer = CLIPTokenizer(vocab_file='../models/sd15models/vocab.json',
                                        merges_file='../models/sd15models/merges.txt')
        promptTokens = promptTokenizer(prompt,padding='max_length',max_length=77,truncation=True,return_tensors='pt')['input_ids'].to(device)
        attentionMask = promptTokenizer(prompt,padding='max_length',max_length=77,truncation=True,return_tensors='pt')['attention_mask'].to(device)
        return promptTokens,attentionMask
            
            
    
    
    
        