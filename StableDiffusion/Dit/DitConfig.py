'''
  "vision_config": {
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "model_type": "siglip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "num_image_tokens": 256,
    "patch_size": 14,
    "projection_dim": 2048,
    "projector_hidden_act": "gelu_fast",
    "vision_use_head": false
  },
'''
class DitConfig:
    def __init__(self,
                 numChannels = 4,
                 patchSize = 4,
                 imageSize = 64,                 
                 embedDims = 1152,
                 hiddenDims = 4304,                 
                 numAttentionHeads = 16,
                 numHiddenLayers = 4,
                 layerNormEps = 1e-6,
                 attentionDropoutRate = 0,                 
                 numImageTokens = 256,
                 projectionDim = 2048,
                 **keywordsArgs):
        super(DitConfig,self).__init__()
        #super().__init__()
        self.numChannels = numChannels # 3
        self.patchSize = patchSize #14
        self.imageSize = imageSize #224
        self.hiddenSize = embedDims  # 1152
        self.intermediateSize = hiddenDims# 4304
        self.numAttenionHeads = numAttentionHeads #16
        self.numHiddenLayers = numHiddenLayers #27
        self.layerNormEps = layerNormEps #1e-6
        self.attentionDropoutRate = attentionDropoutRate#0
        self.numImageTokens = numImageTokens #256
        self.projectionDim = projectionDim #2048
    
    def show(self):
        print("numChannels : ", self.numChannels)
        print("patchSize : ", self.patchSize)
        print("imageSize : ", self.imageSize)
        print("hiddenSize : ", self.hiddenSize)
        print("intermediateSize : ", self.intermediateSize)
        print("numAttentionHeads : ", self.numAttenionHeads)
        print("numHiddenLayers : ", self.numHiddenLayers)
        print("layerNormEps : ", self.layerNormEps)
        print("attentionDropoutRate : ", self.attentionDropoutRate)
        print("numImageTokens : ", self.numImageTokens)



        