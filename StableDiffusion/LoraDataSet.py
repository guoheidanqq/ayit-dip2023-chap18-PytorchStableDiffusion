import cv2
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from StableDiffusion.Utils import *
class LoraDataSet(Dataset):
    def __init__(self):
        super(LoraDataSet,self).__init__()
        self.dataPath = '/home/aistudio/PytorchStableDiffusion/images/xfx_512_512'
        self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512,512)),
                transforms.ToTensor(),   # (0 255) -> (-1, 1)
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])# 
                ])
        self.promptTokenizer = CLIPTokenizer(vocab_file='../models/sd15models/vocab.json',
                                        merges_file='../models/sd15models/merges.txt')
        self.imageFileList = []
        self.textFileList = []
        for fileName in os.listdir(self.dataPath):
            if fileName.endswith('.jpg'):
                self.imageFileList.append(os.path.join(self.dataPath,fileName))
            
            if fileName.endswith('.txt'):
                self.textFileList.append(os.path.join(self.dataPath,fileName))
    
    def loadImageBatch(self,filePath:str,device='cuda')->torch.Tensor:
        img1 = cv2.imread(filePath)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        imgTensor = self.preprocess(img1)
        imgBatch = imgTensor[None, :, :, :]
        return imgBatch[0].to(device)   
    
    def getPromptTokens(self,prompt:str,device='cuda')->Tuple[torch.LongTensor,torch.LongTensor]:

        promptTokens = self.promptTokenizer(prompt,padding='max_length',
                                            max_length=77,truncation=True,return_tensors='pt')['input_ids']
        attentionMask = self.promptTokenizer(prompt,padding='max_length',
                                             max_length=77,truncation=True,return_tensors='pt')['attention_mask']
        return promptTokens[0].to(device),attentionMask[0].to(device)
    
    def __getitem__(self,index):    
        filePath = self.imageFileList[index]
        with open(self.textFileList[index],'r',encoding='utf-8') as f:
            prompt = f.read()
        imgTensor = self.loadImageBatch(filePath)
        promptTokens,attentionMask = self.getPromptTokens(prompt)

        return imgTensor,promptTokens,attentionMask

    def __len__(self):
        return len(self.imageFileList)