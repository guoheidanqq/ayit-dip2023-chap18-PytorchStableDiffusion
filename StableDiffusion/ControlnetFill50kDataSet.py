import cv2
import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from StableDiffusion.Utils import * 
from .CannyEdgeDetection import CannyEdgeDetection
class ControlnetFill50kDataSet(Dataset):
    def __init__(self):
        super().__init__()
        self.dataPath = '/home/aistudio/models/fill50k/target'
        self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512,512)),
                transforms.ToTensor(),   # (0 255) -> (-1, 1)
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])# 
                ])
        self.promptTokenizer = CLIPTokenizer(vocab_file='../models/sd15models/vocab.json',
                                        merges_file='../models/sd15models/merges.txt')
        self.imageFileList = []
        self.promptList = []
        # Define the filename
        fill50Path = '/home/aistudio/models/fill50k'
        jsonFile = os.path.join(fill50Path, 'prompt.json')

        with open(jsonFile, 'r') as f:
            for line in f:
                # Load the JSON object from the current line
                data = json.loads(line.strip())
                
                # Extract the specific fields
                fileName = data['target'].split('/')[-1]  # Gets '0.png' from 'target/0.png'
                promptStr = data['prompt']
                self.imageFileList.append(os.path.join(self.dataPath, fileName))
                self.promptList.append(promptStr)
        
        self.imageFileList = self.imageFileList[:100]
        self.promptList = self.promptList[:100]

    
    def loadImageBatch(self,filePath:str,device='cuda')->torch.Tensor:
        img1 = cv2.imread(filePath)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img1 = cv2.resize(img1, (512, 512))
        cannyDetector = CannyEdgeDetection()
        controlHint = cannyDetector(img1)
        controlHint =controlHint[0,:,:,:]
        imgTensor = self.preprocess(img1)
        imgBatch = imgTensor[None, :, :, :]
        return imgBatch[0].to(device),controlHint.to(device)
        
    
    def getPromptTokens(self,prompt:str,device='cuda')->Tuple[torch.LongTensor,torch.LongTensor]:

        promptTokens = self.promptTokenizer(prompt,padding='max_length',
                                            max_length=77,truncation=True,return_tensors='pt')['input_ids']
        attentionMask = self.promptTokenizer(prompt,padding='max_length',
                                             max_length=77,truncation=True,return_tensors='pt')['attention_mask']
        return promptTokens[0].to(device),attentionMask[0].to(device)
    
    def __getitem__(self,index):    
        filePath = self.imageFileList[index]
        prompt = self.promptList[index]

        imgTensor,controlHintTensor = self.loadImageBatch(filePath)
        promptTokens,attentionMask = self.getPromptTokens(prompt)

        return imgTensor,controlHintTensor,promptTokens,attentionMask

    def __len__(self):
        return len(self.imageFileList)