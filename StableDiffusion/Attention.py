import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import CLIPTokenizer
from typing import Optional,List


class MHSelfAttention(nn.Module):
    def __init__(self,numHeads=12,embeddingDimensions=768,\
                isCausalMask = True,\
                isInProjBias = False,isOutProjBias = True):
        super(MHSelfAttention,self).__init__()
        self.isInProjBias = isInProjBias 
        self.isOutProjBias = isOutProjBias
        self.embeddingDimensions = embeddingDimensions 
        self.numHeads = numHeads 
        self.isCausalMask = isCausalMask 
        self.headDimensions = self.embeddingDimensions//self.numHeads 
        self.in_proj  = nn.Linear(in_features = embeddingDimensions, out_features = 3*embeddingDimensions,bias = self.isInProjBias)
        #diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q
        #diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k
        #diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v
        self.out_proj  = nn.Linear(in_features = embeddingDimensions, out_features = embeddingDimensions,bias = self.isOutProjBias)
        #diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out
    def forward(self,inputs,isCausalMask = True,attentionMask:Optional[torch.tensor]=None):
        self.isCausalMask = isCausalMask
        B,N,D = inputs.shape 
        

         
        x = inputs 
        x = self.in_proj(x)
        Q,K,V = x.chunk(3,dim = -1 )
        Q = Q.reshape(B,N,self.numHeads,self.headDimensions)  # B,N,H,Dh
        Q = Q.permute(0,2,1,3)
        K = K.reshape(B,N,self.numHeads,self.headDimensions)
        K = K.permute(0,2,1,3)
        V = V.reshape(B,N,self.numHeads,self.headDimensions)
        V = V.permute(0,2,1,3)
        attention = Q @ K.permute(0,1,3,2) 
        attention = attention * self.headDimensions**(-0.5)
        if self.isCausalMask ==True:
            mask = torch.ones(N,N,dtype = torch.bool,device=attention.device).triu(diagonal = 1) 
            # not including the digonal 
            attention.masked_fill_(mask==1,-torch.inf)
        
        if attentionMask is not None:
            #Batch SeqLen->B,1,1,SeqLen
            attentionMask = attentionMask[:,None,None,:]
            attention.masked_fill_(attentionMask==0,-torch.inf)
        
        attention = F.softmax(attention,dim = -1 )    
        x = attention @ V # B,H,N,Dh
        x = x.permute(0,2,1,3)
        x = x.reshape(B,N,self.embeddingDimensions) # B,N,D
        x = self.out_proj(x)   
        # bug fix  20260214 add self.out_proj(x)
        
        return x 


       
class MHCrossAttention(nn.Module):
    def __init__(self,numHeads=8,latentEmbeddingDimension=1280,contextEmbeddingDimension = 768,
                 isInProjBias = False,isOutProjBias = True): 
        super(MHCrossAttention,self).__init__()
        self.numHeads = numHeads 
        self.latentEmbeddingDimension = latentEmbeddingDimension
        self.contextEmbeddingDimension = contextEmbeddingDimension
        self.isInProjBias = isInProjBias
        self.isOutProjBias = isOutProjBias
        self.headDimension = self.latentEmbeddingDimension//self.numHeads
        self.q_proj = nn.Linear(in_features=self.latentEmbeddingDimension,
                                out_features=self.latentEmbeddingDimension,
                                bias=self.isInProjBias)
        #diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q
        self.k_proj = nn.Linear(in_features= self.contextEmbeddingDimension,
                                out_features=self.latentEmbeddingDimension,
                                bias = self.isInProjBias)
        #diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k
        self.v_proj = nn.Linear(in_features=self.contextEmbeddingDimension,
                                out_features=self.latentEmbeddingDimension,
                                bias=self.isInProjBias)
        #diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v
        self.out_proj = nn.Linear(in_features=self.latentEmbeddingDimension,
                                  out_features=self.latentEmbeddingDimension,
                                  bias=self.isOutProjBias)
        #diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out
        #to_out default is  bias = True
        #q_proj k_proj v_proj is bias = False
        
        
    
    def forward(self,inputLatent,inputContext):
        x = inputLatent  # B 4096 1280
        y = inputContext  # B 77 768
        B,latentN,latentD = inputLatent.shape 
        B,contextN,contextD = inputContext.shape 
        
        q = self.q_proj(x) # B 4096 1280
        k = self.k_proj(y)  # B 77 1280
        v = self.v_proj(y) # B 77 1280
        
        q = q.reshape(B,latentN,self.numHeads,self.headDimension) # B 4096 8 160
        q = q.permute(0,2,1,3)  # B 8 4096 160
        k = k.reshape(B,contextN,self.numHeads,self.headDimension)  
        k = k.permute(0,2,1,3) # B 8 77  160
        v = v.reshape(B,contextN,self.numHeads,self.headDimension)  
        v = v.permute(0,2,1,3)# B 8 77  160
        #print(f'q shape{q.shape} , k shape {k.shape},  v shape {v.shape}')
        
        attentionWeight = q @ k.permute(0,1,3,2)  # B 8 4096 77
        attentionWeight = attentionWeight * (self.headDimension ** -0.5)
        attentionWeight = F.softmax(attentionWeight,dim = -1)
        
        out = attentionWeight @ v  # B 8 4096 160
        out = out.permute(0,2,1,3)
        out = out.reshape(B,latentN,-1)
        out = self.out_proj(out)       
        
        x = out
        return x  