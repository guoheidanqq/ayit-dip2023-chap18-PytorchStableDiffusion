import numpy as np
import torch
class DdpmSamplerTorch:
    def __init__(self,randomGenerator = None,
                 numTrainingSteps = 1000,
                 numInferenceSteps = 50,
                 betaStart= 0.00085,
                 betaEnd =0.0120):
        self.randomGenerator = randomGenerator
        self.numTrainingSteps = numTrainingSteps
        self.numInferenceSteps = numInferenceSteps
        self.betaStart = betaStart
        self.betaEnd = betaEnd
        self.deltaSteps  = self.numTrainingSteps // self.numInferenceSteps
        self.betas = torch.linspace(self.betaStart**0.5, self.betaEnd**0.5,self.numTrainingSteps,dtype=torch.float32)**2  # linear schedule
        #self.betas = torch.linspace(self.betaStart, self.betaEnd,self.numTrainingSteps,dtype=torch.float32)  # linear schedule
        self.alphas = 1.0- self.betas
        self.alphasBar = torch.cumprod(self.alphas,dim=0,dtype=torch.float32)
        self.betasBar =1.0- self.alphasBar
       
        self.alphaBarPreT = torch.cat([torch.tensor([1.0]),self.alphasBar[0:-1]],dim=0)
        self.sqrtAlphasBar = self.alphasBar**0.5     
        # sqrt alphasBar is signal strength   
        self.sqrtOneMinusAlphasBar = (1.0-self.alphasBar)**0.5
        # sqrt oneMinusAlphasBar is noise strength
        self.trainingTimeSteps = torch.from_numpy(np.arange(self.numTrainingSteps)[::-1].copy())
        self.trainingTimeSteps =self.trainingTimeSteps.to(torch.int64)
        self.inferenceTimeSteps = torch.from_numpy(np.arange(self.numInferenceSteps)[::-1].copy()*self.deltaSteps)
        self.fullInferenceTimeSteps = self.inferenceTimeSteps.clone()
        self.inferenceTimeSteps =self.inferenceTimeSteps.to(torch.int64)
        
    def to(self,device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphasBar = self.alphasBar.to(device)
        self.alphaBarPreT = self.alphaBarPreT.to(device)
        self.sqrtAlphasBar = self.sqrtAlphasBar.to(device)
        self.sqrtOneMinusAlphasBar = self.sqrtOneMinusAlphasBar.to(device)
        self.trainingTimeSteps = self.trainingTimeSteps.to(device)
        self.inferenceTimeSteps = self.inferenceTimeSteps.to(device)
        
                                                                   
                                                                   
                                                                   
    def tensor2Numpy(self,input):
        output = input.detach().cpu().numpy()
        return output
    
    def numpy2Tensor(self,input,device='cuda'):
        output = torch.from_numpy(input.astype(np.float32)).to(device)
        return output
    
    def getInferenceTimeSteps(self)->torch.Tensor:
        inferTimeSteps = self.inferenceTimeSteps
        return inferTimeSteps

    def getVariance(self,timeStep):
        currentT = timeStep
        preT = self.getPreviousTimeStep(timeStep)
        if preT < 0 :
            alphaBarPreT = 1.0
        else:
            alphaBarPreT = self.alphasBar[preT]        
        alphaBartT = self.alphasBar[currentT]
        alphaTEffective = alphaBartT/alphaBarPreT
        betaTEffective = 1.0- alphaBartT/alphaBarPreT
        variance = (1.0-alphaBarPreT)/(1.0-alphaBartT)*betaTEffective        
        variance = variance.clamp(min=1e-20)
        return variance
    def  getPreviousTimeStep(self,timeStepCurrent):
        preT = timeStepCurrent - self.deltaSteps
        return preT 

    def setAddNoiseStrength(self,noiseStrength=1.0):
        startStep = int(self.numInferenceSteps*(1-noiseStrength))
        print(f'in ddpm start step is {startStep}')
        self.inferenceTimeSteps = self.fullInferenceTimeSteps[startStep:]
        
    
    def removeNoiseFromLatent(self,latentInputs:torch.Tensor,estimateNoise:torch.Tensor,timeStep:int):
        xt = latentInputs
        device = latentInputs.device
        epsilonT = estimateNoise
        t = timeStep 
        tPre = self.getPreviousTimeStep(t)     
        alphaT = self.alphas[t].to(device)
        betaT = self.betas[t].to(device)
        
        alphaBarT = self.alphasBar[t].to(device)  
        betaBarT = self.betasBar[t].to(device)
        sqrtAlphaBarT = self.sqrtAlphasBar[t].to(device)
        sqrtOneMinusAlphaBarT = self.sqrtOneMinusAlphasBar[t].to(device)
        if tPre<0:
            alphaBarTPre = torch.tensor(1.0).to(device)
        else:
            alphaBarTPre = self.alphasBar[tPre].to(device)
        betaBarTPre =1.0 -alphaBarTPre
        alphaBarTEffective = alphaBarT/alphaBarTPre
        betaTEffective = 1.0- alphaBarTEffective
            
        
        #x0EstImage = (xt-torch.sqrt(1-alphaBarT)*epsilonT)/torch.sqrt(alphaBarT) 
        x0EstImage =(xt-sqrtOneMinusAlphaBarT * epsilonT)/sqrtAlphaBarT
        #  compute x0 estimate 
        x0EstImage =  x0EstImage.clamp(min=-1,max=1)
        #x0EstCoeff =  torch.sqrt(alphaBarTPre) * betaT /(1-alphaBarT)
        x0EstCoeff =  torch.sqrt(alphaBarTPre) * betaTEffective /betaBarT
        xTCoeff  =  torch.sqrt(alphaBarTEffective)*betaBarTPre/betaBarT
        meanT = x0EstCoeff * x0EstImage + xTCoeff * xt
        noise = torch.randn(*meanT.shape,generator = self.randomGenerator,dtype=torch.float32,device=device)
        varianceT = 0 
        if t>0:
            varianceT = self.getVariance(t).to(device)
        elif t<=0:
            varianceT = torch.tensor(0.0).to(device)       
        
        stdVarianceT = torch.sqrt(varianceT).to(device)
        xTPre =  meanT + stdVarianceT * noise 
        return xTPre
    
    

        
    
    def addNoise(self,latentInputs,timeStep:int):
        x0 = latentInputs
        #B,C,H,W = x0.shape
        t = timeStep
        #x0Coeff = self.alphasBar[t]**0.5
        x0Coeff = self.sqrtAlphasBar[t]
        #epsilonCoeff = (1- self.alphasBar[t])**0.5
        epsilonCoeff = self.sqrtOneMinusAlphasBar[t]
        noise =torch.randn(*x0.shape,generator = self.randomGenerator,dtype=torch.float32,device=x0.device)
        xt = x0Coeff * x0 + epsilonCoeff * noise
        return xt
    
    def addNoiseBatchTrain(self,latentInputs:torch.Tensor,noise:torch.Tensor,timeStep:int):
        x0 = latentInputs
        B,C,H,W = x0.shape
        t = timeStep
        #x0Coeff = self.alphasBar[t]**0.5
        x0Coeff = self.sqrtAlphasBar[t].reshape(B,1,1,1)
        #epsilonCoeff = (1- self.alphasBar[t])**0.5
        epsilonCoeff = self.sqrtOneMinusAlphasBar[t].reshape(B,1,1,1)
        #noise =torch.randn(*x0.shape,generator = self.randomGenerator,dtype=torch.float32,device=x0.device)
        xt = x0Coeff * x0 + epsilonCoeff * noise
        return xt
        
        
        
         
        
        
        
    
    
    
    
        
        
        