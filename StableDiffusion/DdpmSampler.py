import numpy as np
import torch
class DdpmSampler:
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
        self.betas = np.linspace(self.betaStart**0.5, self.betaEnd**0.5, self.numTrainingSteps,endpoint = True)**2  # linear schedule
        #self.betas = torch.linspace(self.betaStart, self.betaEnd, self.numTrainingSteps,endpoint = True)  # linear schedule
        self.alphas = 1- self.betas
        self.alphasBar = np.cumprod(self.alphas)
        self.trainingTimeSteps = np.arange(self.numTrainingSteps)[::-1]
        self.inferenceTimeSteps = np.arange(self.numInferenceSteps)[::-1]*self.deltaSteps
        
    
    def tensor2Numpy(self,input):
        output = input.detach().cpu().numpy()
        return output
    
    def numpy2Tensor(self,input,device='cuda'):
        output = torch.from_numpy(input.astype(np.float32)).to(device)
        return output
    
    def getInferenceTimeSteps(self):
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
        betaTEffective = 1- alphaBartT/alphaBarPreT
        variance = (1-alphaBarPreT)/(1-alphaBartT)*betaTEffective        
        variance = variance.clip(min=1e-20)
        return variance
    def  getPreviousTimeStep(self,timeStepCurrent):
        preT = timeStepCurrent - self.deltaSteps
        return preT 

    def setAddNoiseStrength(self,noiseStrength=1.0):
        startStep = int(self.numInferenceSteps*(1-noiseStrength))
        print(f'in ddpm start step is {startStep}')
        self.inferenceTimeSteps = self.inferenceTimeSteps[startStep:]
        
    
    def removeNoiseFromLatent(self,latentInputs,estimateNoise,timeStep):
        xt = latentInputs
        epsilonT = estimateNoise
        t = timeStep 
        tPre = self.getPreviousTimeStep(t)       
        alphaT = self.alphas[t]
        betaT = self.betas[t]
        
        alphaBarT = self.alphasBar[t]  
        if tPre<0:
            alphaBarTPre = 1.0
        else:
            alphaBarTPre = self.alphasBar[tPre]
        betaTEffective = 1- alphaBarT/alphaBarTPre
            
        
        x0EstImage = (xt-np.sqrt(1-alphaBarT)*epsilonT)/np.sqrt(alphaBarT)  #  compute x0 estimate 
        x0EstImage =  x0EstImage.clip(min=-1,max=1)
        x0EstCoeff =  np.sqrt(alphaBarTPre) * betaT /(1-alphaBarT)
        xTCoeff  =  np.sqrt(alphaT)*(1-alphaBarTPre)/(1-alphaBarT)
        meanT = x0EstCoeff * x0EstImage + xTCoeff * xt
        noise = np.random.normal(0,1,meanT.shape)
        varianceT = 0 
        if t>0:
            varianceT = self.getVariance(t)
        elif t<=0:
            varianceT = 0       
        
        stdVarianceT = np.sqrt(varianceT)
        xTPre =  meanT + stdVarianceT* noise 
        return xTPre
    
    

        
    
    def addNoise(self,latentInputs,timeStep:int):
        x0 = latentInputs
        #B,C,H,W = x0.shape
        t = timeStep
        x0Coeff = self.alphasBar[t]**0.5
        epsilonCoeff = (1- self.alphasBar[t])**0.5
        xt = x0Coeff * x0 + epsilonCoeff * np.random.normal(0,1,x0.shape)
        return xt
        
        
        
         
        
        
        
    
    
    
    
        
        
        