import cv2
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from StableDiffusion.Utils import *
class DitDataSet(Dataset):
    def __init__(self):
        super().__init__()
        self.dataPath = '/home/aistudio/models/ginkgo'
        self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop((1024,1024)),
                transforms.Resize((512,512)),
                transforms.ToTensor(),   # (0 255) -> (-1, 1)
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])# 
                ])
        self.imageFileList = []        
        for fileName in os.listdir(self.dataPath):
            if fileName.endswith('.jpg'):
                self.imageFileList.append(os.path.join(self.dataPath,fileName))
            

    
    def loadImageBatch(self,filePath:str,device='cuda')->torch.Tensor:
        img1 = cv2.imread(filePath)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        imgTensor = self.preprocess(img1)
        imgBatch = imgTensor[None, :, :, :]
        return imgBatch[0].to(device)       

    def __getitem__(self,index):    
        filePath = self.imageFileList[index]
        imgTensor = self.loadImageBatch(filePath)
        return imgTensor

    def __len__(self):
        return len(self.imageFileList)