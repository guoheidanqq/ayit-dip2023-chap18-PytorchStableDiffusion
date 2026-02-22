import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import CLIPTokenizer
from StableDiffusion.TimeEmbedding import TimeEmbedding
from StableDiffusion.Attention import MHSelfAttention
from StableDiffusion.Attention import MHCrossAttention
from StableDiffusion.UnetResidualBlock import UnetResidualBlock
from StableDiffusion.UnetGlobalCrossAttentionBlock import UnetGlobalCrossAttentionBlock
from StableDiffusion.UnetOutputLayer import UnetOutputLayer
from StableDiffusion.UpScaleTwo import UpScaleTwo
from StableDiffusion.SequentialAdapter import SequentialAdapter

class UnetDenoise(nn.Module):
    def __init__(self,):
        super(UnetDenoise,self).__init__() 
        self.encoders = nn.ModuleList([
                        # stage 1 B 4 64 64 -> B 320 64 64
                            # level 0 
                        SequentialAdapter(nn.Conv2d(in_channels=4,out_channels = 320,kernel_size = 3,padding = 1)), 
                            #diffusion_model.inputs.0.0  320,4,3,3
                        # B 4 64 64 -> B 320 64 64
                            # level 1
                        SequentialAdapter(UnetResidualBlock(inChannels=320,outChannels= 320,timeEmbeddingDimension=1280),\
                                        #diffusion_model.inputs.1.0.in_layers.0 groupnorm
                                        #diffusion_model.inputs.1.0.in_layers.2 conv
                                        #diffusion_model.inputs.1.0.out_layers.0 groupnorm
                                        #diffusion_model.inputs.1.0.out_layers.2 conv
                                            UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension=320,contextEmbeddingDimension=768)),  
                                        #diffusion_model.inputs.1.1.norm
                                        #diffusion_model.inputs.1.1.transformer_block.0.attn1.to_k
                                        #diffusion_model.inputs.1.1.transformer_block.0.attn2.to_v
                            
                            # level 2
                        SequentialAdapter(UnetResidualBlock(inChannels=320,outChannels= 320,timeEmbeddingDimension=1280),\
                                            UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension=320,contextEmbeddingDimension=768)),     
                            #diffusion_model.input_blocks.2.0.in_layers.0
                            #diffusion_model.input_blocks.2.0.in_layers.2  320 320 3 3
                            #diffusion_model.input_blocks.2.0.out_layers.3 320 320 3 3
                            #diffusion_model.input_blocks.2.0.emb_layers.1 320 1280
                            #diffusion_model.input_blocks.2.1.transformer_blocks.0.attn1.to_k 320 320
                            #diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k 320 768
                            
                        # stage 2 B 320 64 64 -> B 640 32 32
                            # level 3
                        SequentialAdapter(nn.Conv2d(in_channels=320,out_channels=320,kernel_size=3,stride=2,padding=1)),  
                            # B 320 64 64 -> B 640 32 32  
                            #diffusion_model.input_blocks.3.0.op 320,320,3,3
                        
                            # level 4
                        SequentialAdapter(UnetResidualBlock(inChannels=320,outChannels= 640,timeEmbeddingDimension=1280),\
                                            UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension=640,contextEmbeddingDimension=768)),  
                            #.diffusion_model.input_blocks.4.0.in_layers.2  640 320 3 3
                            # diffusion_model.input_blocks.4.0.emb_layers.1 640 1280
                            #diffusion_model.input_blocks.4.0.out_layers.3.weight 640 640 3 3
                            #diffusion_model.input_blocks.4.0.skip_connection.weight  640 320 1 1
                            #diffusion_model.input_blocks.4.1.proj_in.weight
                            #diffusion_model.input_blocks.4.1.proj_out.weight
                            #diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight 640 640
                            #diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight 640 768
                            #diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight 5120 640
                            #diffusion_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.weight 640 2560
                            
                            
                            
                            # level 5
                        SequentialAdapter(UnetResidualBlock(inChannels=640,outChannels= 640,timeEmbeddingDimension=1280),\
                                            UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension= 640,contextEmbeddingDimension=768)),   
                            #diffusion_model.input_blocks.5.0.emb_layers.1.weight   640 1280
                            #diffusion_model.input_blocks.5.0.out_layers.3.weight   640 640 3 3
                            #diffusion_model.input_blocks.5.1.transformer_blocks.0.attn1.to_k.weight 640 640
                            #diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight 640 768
                                    
                        
                        # stage 3 B 640 32 32 -> B 1280 16 16
                            # level 6
                        SequentialAdapter(nn.Conv2d(in_channels=640,out_channels=640,kernel_size=3,stride=2,padding=1)), 
                            # B 320 64 64 -> B 640 32 32 
                            #diffusion_model.input_blocks.6.0.op 640,640,3,3
                            
                            # level 7
                        SequentialAdapter(UnetResidualBlock(inChannels=640,outChannels= 1280,timeEmbeddingDimension=1280),\
                                            UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension=1280,contextEmbeddingDimension=768)),  
                            #diffusion_model.input_blocks.7.0.emb_layers.1.weight 1280 1280
                            #diffusion_model.input_blocks.7.0.in_layers.2.weight 1280 640
                            #diffusion_model.input_blocks.7.0.out_layers.3.weight 1280 1280
                            #diffusion_model.input_blocks.7.0.skip_connection.weight 1280 640
                            #diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_k.weight 1280 1280
                            #diffusion_model.input_blocks.7.1.transformer_blocks.0.attn1.to_k.weight 1280 768
                            
                            # level 8
                        SequentialAdapter(UnetResidualBlock(inChannels=1280,outChannels= 1280,timeEmbeddingDimension=1280),\
                                            UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension= 1280,contextEmbeddingDimension=768)),  
                            #diffusion_model.input_blocks.8.0.in_layers.2.weight 1280 1280
                            #diffusion_model.input_blocks.8.0.out_layers.3.weight 1280 1280
                            #diffusion_model.input_blocks.8.1.proj_out.weight 1280 1280
                            #diffusion_model.input_blocks.8.1.transformer_blocks.0.attn1.to_k.weight  1280 1280
                            #diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight   1280 768
                            #diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight 10240 1280
                            #diffusion_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.weight 1280 5120
                        
                        
                        # stage 4 B 1280 16 16 -> B 1280 8 8
                            # level 9
                        SequentialAdapter(nn.Conv2d(in_channels=1280,out_channels=1280,kernel_size=3,stride=2,padding=1)),  
                            # B 1280 16 16 -> B 1280 8 8
                            #diffusion_model.input_blocks.9.0.op 1280,1280,3,3
                            #
                               
                            # level 10
                        SequentialAdapter(UnetResidualBlock(inChannels=1280,outChannels= 1280,timeEmbeddingDimension=1280)),
                            #diffusion_model.input_blocks.10.0.in_layers.2.weight 1280 1280
                            #diffusion_model.input_blocks.10.0.out_layers.3.weight 1280 1280
                            
                            # level 11
                        SequentialAdapter(UnetResidualBlock(inChannels=1280,outChannels= 1280,timeEmbeddingDimension=1280))
                            #diffusion_model.input_blocks.11.0.in_layers.2.weight 1280 1280
                            #diffusion_model.input_blocks.11.0.out_layers.3.weight 1280 1280
                        ]) 
        
        self.bottleneck =  SequentialAdapter(
                            #middle 0
                        UnetResidualBlock(inChannels=1280,outChannels= 1280,timeEmbeddingDimension=1280),
                            #diffusion_model.middle_block.0.emb_layers.1.weight 1280 1280
                            #diffusion_model.middle_block.0.in_layers.2.weight 1280 1280
                            #diffusion_model.middle_block.0.out_layers.3.weight 1280 1280                     

                            #middle 1
                        UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension=1280,contextEmbeddingDimension=768),
                            #diffusion_model.middle_block.1.proj_in.weight  1280 1280
                            #diffusion_model.middle_block.1.proj_out.weight  1280 1280
                            #diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight 1280 1280
                            #diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight 1280 768
                            #diffusion_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight 10240 1280
                            #diffusion_model.middle_block.1.transformer_blocks.0.ff.net.2.proj.weight 1280 5120
                        
                            #middle 2
                        UnetResidualBlock(inChannels=1280,outChannels= 1280,timeEmbeddingDimension=1280))
                            #diffusion_model.middle_block.2.emb_layers.1.weight 1280 1280
                            #diffusion_model.middle_block.2.in_layers.2.weight  1280 1280
                            #diffusion_model.middle_block.2.out_layers.3.weight 1280 1280
        
        self.decoders = nn.ModuleList([
                        # stage 4 B 2560 8 8 -> B 1280 16 16
                            # level 11 
                        SequentialAdapter(UnetResidualBlock(inChannels=2560,outChannels= 1280,timeEmbeddingDimension=1280)),
                            #diffusion_model.output_blocks.0.0.in_layers.2.weight 1280 2560
                            #diffusion_model.output_blocks.0.0.out_layers.3.weight 1280 1280
                            #diffusion_model.output_blocks.0.0.skip_connection.weight 1280 2560
                            
                            
                            # level 10
                        SequentialAdapter(UnetResidualBlock(inChannels=2560,outChannels= 1280,timeEmbeddingDimension=1280)), 
                            #diffusion_model.output_blocks.1.0.in_layers.2.weight 1280 2560
                            #diffusion_model.output_blocks.1.0.out_layers.3.weight 1280 1280
                            #diffusion_model.output_blocks.1.0.skip_connection.weight 1280 2560
                            
                            
                            # level 9
                        SequentialAdapter(UnetResidualBlock(inChannels=2560,outChannels= 1280,timeEmbeddingDimension=1280),\
                                        UpScaleTwo(inChannels=1280)),
                            #diffusion_model.output_blocks.2.0.in_layers.2.weight 1280 2560
                            #diffusion_model.output_blocks.2.0.out_layers.3.weight 1280 1280
                            #diffusion_model.output_blocks.2.0.skip_connection.weight 1280 2560
                            #diffusion_model.output_blocks.2.1.conv.weight 1280 1280 3 3  UpScaleTwo
                        
                        # stage 3 B 1280 16 16 -> B 640 32 32
                            # level 8
                        SequentialAdapter(UnetResidualBlock(inChannels=2560,outChannels= 1280,timeEmbeddingDimension=1280),\
                                        UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension=1280,contextEmbeddingDimension=768)),
                            #diffusion_model.output_blocks.3.0.in_layers.2.weight 1280 2560
                            #diffusion_model.output_blocks.3.0.out_layers.3.weight 1280 1280
                            #diffusion_model.output_blocks.3.0.skip_connection.weight 1280 2560
                            #diffusion_model.output_blocks.3.1.transformer_blocks.0.attn1.to_k.weight 1280 1280
                            #diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_k.weight 1280 768  
                            
                            
                            # level 7
                        SequentialAdapter(UnetResidualBlock(inChannels=2560,outChannels= 1280,timeEmbeddingDimension=1280),\
                                        UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension=1280,contextEmbeddingDimension=768)),
                            #diffusion_model.output_blocks.4.0.in_layers.2.weight 1280 2560
                            #diffusion_model.output_blocks.4.0.out_layers.3.weight  1280 1280
                            #diffusion_model.output_blocks.4.0.skip_connection.weight 1280 2560
                            #diffusion_model.output_blocks.4.1.transformer_blocks.0.attn1.to_k.weight 1280 1280
                            #diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_k.weight 1280 768                         
                           
                            # level 6
                        SequentialAdapter(UnetResidualBlock(inChannels=1920,outChannels= 1280,timeEmbeddingDimension=1280),\
                                        UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension=1280,contextEmbeddingDimension=768),\
                                        UpScaleTwo(inChannels=1280)),   
                            #diffusion_model.output_blocks.5.0.in_layers.2.weight 1280 1920
                            #diffusion_model.output_blocks.5.0.out_layers.3.weight 1280 1280
                            #diffusion_model.output_blocks.5.0.skip_connection.weight 1280 1920
                            #diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1.to_k.weight 1280 1280
                            #diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_k.weight 1280 768
                            #diffusion_model.output_blocks.5.2.conv.weight 1280 1280 3 3  UpScaleTwo
                
                        
                        # stage 2 B 640 32 32 -> B 320 64 64                
                            #level 5
                        SequentialAdapter(UnetResidualBlock(inChannels=1920,outChannels= 640,timeEmbeddingDimension=1280),\
                                        UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension=640,contextEmbeddingDimension=768)),                        
                            #diffusion_model.output_blocks.6.0.emb_layers.1.weight 640 1280
                            #diffusion_model.output_blocks.6.0.in_layers.2.weight 640 1920
                            #diffusion_model.output_blocks.6.0.out_layers.3.weight 640 640
                            #diffusion_model.output_blocks.6.0.skip_connection.weight	640, 1920, 1, 1
                            #diffusion_model.output_blocks.6.1.transformer_blocks.0.attn1.to_k 640 640
                            #diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_k.weight   640 768
                            
                            
                                
                            #level 4
                         SequentialAdapter(UnetResidualBlock(inChannels=1280,outChannels= 640,timeEmbeddingDimension=1280),\
                                        UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension=640,contextEmbeddingDimension=768)),             
                            #diffusion_model.output_blocks.7.0.emb_layers.1.weight 640 1280
                            #diffusion_model.output_blocks.7.0.in_layers.2.weight 640 1280
                            #diffusion_model.output_blocks.7.0.out_layers.3.weight 640 640
                            #diffusion_model.output_blocks.7.0.skip_connection.weight 640 1280 3 3
                            #diffusion_model.output_blocks.7.1.transformer_blocks.0.attn1.to_k.weight 640 640
                            #diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_k.weight 640 768
                            
                            #level 3
                         SequentialAdapter(UnetResidualBlock(inChannels=960,outChannels= 640,timeEmbeddingDimension=1280),\
                                        UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension=640,contextEmbeddingDimension=768),
                                        UpScaleTwo(inChannels=640)),
                            #diffusion_model.output_blocks.8.0.in_layers.2.weight 640 960
                            #diffusion_model.output_blocks.8.0.out_layers.3.weight 640 640
                            #diffusion_model.output_blocks.8.0.skip_connection.weight 640 960
                            #diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_k.weight 640 640
                            #diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_k.weight 640 768
                            #diffusion_model.output_blocks.8.2.conv.weight 640 640 3 3  UpScaleTwo
                         
                         # stage 1 B 320 64 64 -> B 320 64 64
                             #level 2
                        SequentialAdapter(UnetResidualBlock(inChannels=960,outChannels= 320,timeEmbeddingDimension=1280),\
                                        UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension=320,contextEmbeddingDimension=768)),  
                            #diffusion_model.output_blocks.9.0.emb_layers.1.weight  320 1280
                            #diffusion_model.output_blocks.9.0.in_layers.2.weight   320 960
                            #diffusion_model.output_blocks.9.0.out_layers.3.weight 320 320
                            #diffusion_model.output_blocks.9.0.skip_connection.weight 320 960
                            #diffusion_model.output_blocks.9.1.transformer_blocks.0.attn1.to_k.weight 320 320
                            #diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_k.weight 320 768
                            
                            
                            
                            #level 1                      
                         SequentialAdapter(UnetResidualBlock(inChannels=640,outChannels= 320,timeEmbeddingDimension=1280),\
                                        UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension=320,contextEmbeddingDimension=768)),             
                            #diffusion_model.output_blocks.10.0.emb_layers.1.weight 320 1280
                            #diffusion_model.output_blocks.10.0.in_layers.2.weight 320 640
                            #diffusion_model.output_blocks.10.0.out_layers.3.weight 320 320
                            #diffusion_model.output_blocks.10.0.skip_connection.weight 320 640
                            #diffusion_model.output_blocks.10.1.transformer_blocks.0.attn1.to_k.weight 320 320
                            #diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_k.weight 320 768
                            
                            #level 0                      
                         SequentialAdapter(UnetResidualBlock(inChannels=640,outChannels= 320,timeEmbeddingDimension=1280),\
                                        UnetGlobalCrossAttentionBlock(numHeads=8,latentEmbeddingDimension=320,contextEmbeddingDimension=768))
                            #diffusion_model.output_blocks.11.0.emb_layers.1.weight 320 1280
                            #diffusion_model.output_blocks.11.0.in_layers.2.weight 320 640
                            #diffusion_model.output_blocks.11.0.out_layers.3.weight 320 320
                            #diffusion_model.output_blocks.11.0.skip_connection.weight  320 640
                            #diffusion_model.output_blocks.11.1.transformer_blocks.0.attn1.to_k.weight 320 320
                            #diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_k.weight 320 768                       
                        ])
        
    def forward(self,latentInput,contextInput,timeStep1280):
        latent =    latentInput  #  B 4 64 64
        context = contextInput   # B 77 768
        time = timeStep1280  # 1 1280
        
        skipConnections = []
        for i,layer in enumerate(self.encoders):
            latent = layer(latent,context,time)
            #print(f'encoder layer {i} ,latent shape {latent.shape} ')       
            skipConnections.append(latent)
        
        latent = self.bottleneck(latent,context,time)
        #print(f'bottle neck layer {12} ,latent shape {latent.shape} ')  
        
        for i,layer in enumerate(self.decoders):
            skipLatent = skipConnections.pop()
            latent = torch.cat([latent,skipLatent],dim=1)
            latent = layer(latent,context,time)
            #print(f'decoder layer {11-i} ,latent shape {latent.shape} ')  
       
        return latent