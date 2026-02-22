
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import CLIPTokenizer
from typing import Optional
from .Attention import MHSelfAttention

class ClipEmbedding(nn.Module):
    def __init__(self,numVocabularies = 49408,numEmbeddings = 768,numTokens = 77):
        super(ClipEmbedding,self).__init__()
        self.numVocabularies = numVocabularies
        self.numEmbeddings = numEmbeddings 
        self.numTokens = numTokens
        #
        self.token_embedding = nn.Embedding(num_embeddings=
                                            self.numVocabularies,embedding_dim=self.numEmbeddings)
        # 49408,768
        #transformer.text_model.embeddings.token_embedding
        self.position_embedding = nn.Parameter(torch.zeros(self.numTokens,self.numEmbeddings))
        # 77,768
        #transformer.text_model.embeddings.position_embedding
    
    def forward(self,inputs):
        x = inputs 
        # B 77
        x = self.token_embedding(x)
        # B,77,768
        print(f'token embedding shape {x.shape}')
        x = x + self.position_embedding
        # B,77,768
        return x 



class ClipEncoderLayer(nn.Module):
    def __init__(self,numHeads=12,embeddingDimensions = 768):
        super(ClipEncoderLayer,self).__init__()
        self.numHeads = numHeads 
        self.embeddingDimensions = embeddingDimensions
        self.layernorm_1 = nn.LayerNorm(normalized_shape=self.embeddingDimensions) 
        #transformer.text_model.encoder.layers.?.layernorm_1
        self.attention  =  MHSelfAttention(numHeads=self.numHeads,
                                           embeddingDimensions=self.embeddingDimensions,
                                           isCausalMask = True,
                                           isInProjBias = True,
                                           isOutProjBias = True)
        #openai clip textmode attention proj_q proj_k proj_v is with bias 
        self.layernorm_2 = nn.LayerNorm(normalized_shape = self.embeddingDimensions)
        #transformer.text_model.encoder.layers.?.layernorm_2
        self.linear_1  = nn.Linear(in_features = self.embeddingDimensions,
                                   out_features = 4 * self.embeddingDimensions)
        #transformer.text_model.encoder.layers.?.mlp.fc1  weightshape 3072 768
        self.linear_2  = nn.Linear(in_features= 4* self.embeddingDimensions,
                                   out_features = self.embeddingDimensions)
        #transformer.text_model.encoder.layers.?.mlp.fc2 weightshape 768 3072
    
    
    def _quickGelu(self,x):
        x = x * torch.sigmoid(1.702*x)
        return x 
    
    def forward(self,inputs,attentionMask:Optional[torch.tensor]=None):
        x = inputs  # B,N,D -> B,77,768
        
        # Self Attention Layer
        residual = x 
        x = self.layernorm_1(x)
        x = self.attention(x,isCausalMask=True,attentionMask=attentionMask)
        x = residual + x 
        
        # Feed Forward Layer
        residual = x         
        x = self.layernorm_2(x)
        x = self.linear_1(x) #  B,77,3072
        x = self._quickGelu(x)
        x = self.linear_2(x) # B,77,768
        x = residual + x       
        
        return x 



class ClipEncoder(nn.Module):
    def __init__(self):
        super(ClipEncoder,self).__init__()
        self.embedding = ClipEmbedding(numVocabularies = 49408,numEmbeddings = 768,numTokens = 77)
        self.layers = nn.ModuleList()
        self.numLayers = 12
        for i in range(self.numLayers):
            self.layers.append(ClipEncoderLayer(numHeads=12,embeddingDimensions=768))
        
        self.layernorm = nn.LayerNorm(normalized_shape = 768)
        
        
    def forward(self,inputs,attentionMask:Optional[torch.Tensor] = None):
        x = inputs   # B,77
        x = x.to(torch.long)
        x = self.embedding(x)
        # B,77,768
        for layer in self.layers:
            x  = layer(x,attentionMask)
        x = self.layernorm(x)    
        return x
            
'''
{
  "_name_or_path": "openai/clip-vit-large-patch14",
  "architectures": [
    "CLIPTextModel"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "dropout": 0.0,
  "eos_token_id": 2,
  "hidden_act": "quick_gelu",
  "hidden_size": 768,
  "initializer_factor": 1.0,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 77,
  "model_type": "clip_text_model",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "projection_dim": 768,
  "torch_dtype": "float32",
  "transformers_version": "4.22.0.dev0",
  "vocab_size": 49408
}

'''