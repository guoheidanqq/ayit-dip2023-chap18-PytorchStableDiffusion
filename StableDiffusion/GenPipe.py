import torch
import torch.nn as nn
import numpy as np 
from tqdm import tqdm
from typing import List,Optional
from .Utils import Utils
from .DdpmSamplerTorch import DdpmSamplerTorch
from .Utils import Utils

class GenPipe:
    def __init__(self,
                 vaeEncoder:nn.Module,
                 vaeDecoder:nn.Module,
                 clipEncoder:nn.Module,
                 diffusionProcess:nn.Module,
                 device = 'cuda'):
        self.diffusionProcess = diffusionProcess
        self.vaeEncoder = vaeEncoder
        self.vaeDecoder = vaeDecoder
        self.clipEncoder = clipEncoder
        self.device = device
        self.inputImage = None
        self.isDoingCfg = True        
        self.numInferenceSteps = 20        
        self.cfgScale = 7.5
        self.seed = 42
        self.noiseStength = 0.3
        
        # all the above provide a default value for the parameters
    
    
    def getRandomGenerator(self)->torch.Generator:
        seed = self.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        randomGenerator = torch.Generator(device=self.device)
        randomGenerator.manual_seed(seed)
        return randomGenerator
    
    def genImage(self,prompPositive:str='a cat',
                promptNegative:str='',refImage:Optional[torch.Tensor]=None,
                noiseStength:float=0.3,
                device='cuda')->List[torch.Tensor]:   
        diffusionProcess = self.diffusionProcess
        vaeEncoder = self.vaeEncoder
        vaeDecoder = self.vaeDecoder
        clipEncoder = self.clipEncoder
        cfgScale = self.cfgScale
        inputImage = self.inputImage
        if refImage is not None:
            inputImage = refImage
        else: 
            inputImage = self.inputImage
        isDoingCfg = self.isDoingCfg
        numInferenceSteps = self.numInferenceSteps
        Width = 512
        Height = 512
        LatentWidth = Width//8
        LatentHeight = Height//8
        LatentShape = (1,4,LatentHeight,LatentWidth) 
        randomGenerator = self.getRandomGenerator()
        sampler = DdpmSamplerTorch(randomGenerator=randomGenerator,
                                numTrainingSteps=1000,
                                numInferenceSteps=numInferenceSteps)
        promptPositive = [prompPositive]        
        promptNegative =[promptNegative]
        promptPositiveTokens,attentionMaskPositive = Utils.getPromptTokens(promptPositive,device=device)
        promptNegativeTokens,attentionMaskNegative = Utils.getPromptTokens(promptNegative,device=device)
        clipInputsPositive =promptPositiveTokens
        clipInputsNegative = promptNegativeTokens
        vaeEncoder.eval()
        vaeDecoder.eval()
        diffusionProcess.eval()
        clipEncoder.eval()
        imgStepList = []
        with torch.no_grad():
            clipOutputsPositive = clipEncoder(clipInputsPositive,attentionMask = attentionMaskPositive)
            clipOutputsNegative = clipEncoder(clipInputsNegative,attentionMask = attentionMaskNegative)   
            
            if isDoingCfg == True:
                clipOutputs = torch.cat([clipOutputsPositive,clipOutputsNegative])
            else:
                clipOutputs = clipOutputsPositive
                
            
            if inputImage is not None:  
                inputNoise =torch.randn(LatentShape,generator=randomGenerator,device=device)      
                latentNoised = vaeEncoder(inputImage,inputNoise)
                sampler.setAddNoiseStrength(noiseStength)
                time = sampler.getInferenceTimeSteps()[0].to(device)
                latentNoised = sampler.addNoise(latentNoised,time)
                imageDecodedDirect = vaeDecoder(latentNoised)       
                Utils.showBatchImage(imageDecodedDirect)
                
            else: 
                print(f'input image is none,use random noise instead')
                latentNoised = torch.randn(LatentShape,generator=randomGenerator,device=device)
            
            timesteps = sampler.getInferenceTimeSteps()    
            timesteps = tqdm(timesteps)            
            for i,time in enumerate(timesteps):
                #print(f'i step: {i} time step: {time} {time.device}')              
                timeEmbedding320 = Utils.getTimeEmbedding(time)    
                timeEmbedding320= sampler.numpy2Tensor(timeEmbedding320,device=device)
                modelNoisedLatentInput = latentNoised
                contextInput = clipOutputs
                if isDoingCfg == True:
                    modelNoisedLatentInput = modelNoisedLatentInput.repeat(2,1,1,1)

                modelEstimatedNoiseInLatent  = diffusionProcess(modelNoisedLatentInput,contextInput,timeEmbedding320)
                
                if isDoingCfg == True:
                    positiveEstimate,negativeEstimate = modelEstimatedNoiseInLatent.chunk(2,dim=0)
                    modelEstimatedNoiseInLatent = cfgScale * positiveEstimate  + \
                                                (1-cfgScale) * negativeEstimate
                    
                    modelNoisedLatentInput = latentNoised
 
                cleanerLatent = sampler.removeNoiseFromLatent(modelNoisedLatentInput,modelEstimatedNoiseInLatent,time)        
                latentNoised  = cleanerLatent
                imageDecoded = vaeDecoder(latentNoised)
                imgStepList.append(imageDecoded)

        Utils.showBatchImage(imgStepList[-1]) 

        return imgStepList
