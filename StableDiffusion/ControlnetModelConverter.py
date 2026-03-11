import torch
import torch

import torch

def convert_ldm_to_diffusers(original_sd):
    new_sd = {}
    
    # Track attention parts to concatenate later
    attn_parts = {}

    for key, value in original_sd.items():
        # Remove the 'control_model.' prefix
        k = key.replace("control_model.", "")
        
        # 1. Time Embeddings
        if k.startswith("time_embed"):
            k = k.replace("time_embed.0", "time_embed.linear_1")\
            .replace("time_embed.2", "time_embed.linear_2")

        # 2. Input Hint Block (Canny To Latent)
        if k.startswith("input_hint_block"):
            k = k.replace("input_hint_block", "input_hint_block.cannyToLatent")
            if ".14." in k:
                k = k.replace(".14.", ".14.zeroConv.")
        if k.startswith("zero_convs"):
            # e.g., zero_convs.0.0.weight -> zero_convs.0.zeroConv.weight
            parts = k.split('.')
            k = f"zero_convs.{parts[1]}.zeroConv.{parts[3]}"

        if k.startswith("middle_block_out"):
            k = k.replace("middle_block_out.0", "middle_block_out.zeroConv")

        # 3. Input Blocks & Middle Blocks (ResNet + Downsample)
        if "input_blocks." in k or "middle_block." in k:
            # ResNet mappings            
            k = k.replace(".in_layers.0", ".groupnorm_feature")
            k = k.replace(".in_layers.2", ".conv_feature")
            k = k.replace(".emb_layers.1", ".linear_time")
            k = k.replace(".out_layers.0", ".groupnorm_merged")
            k = k.replace(".out_layers.3", ".conv_merged")
            k = k.replace(".skip_connection", ".residual_layer")
            # Downsample mapping
        # 4. Zero Convolutions

        new_sd[k] = value

    # Final Step: Perform concatenation for in_proj (Self-Attention)
    for target_key, components in attn_parts.items():
        # Stack in order: Q, K, V
        new_sd[target_key] = torch.cat([components['q'], components['k'], components['v']], dim=0)

    return new_sd
            
'''   
k = k.replace(".op.", ".")
            
            # Spatial Transformer mappings
            if "transformer_blocks" in k:
                k = k.replace(".norm1", ".layernorm_1").replace(".norm2", ".layernorm_2").replace(".norm3", ".layernorm_3")
                k = k.replace(".ff.net.0.proj", ".linear_geglu_1").replace(".ff.net.2", ".linear_geglu_2")
                k = k.replace(".attn2.to_q", ".attention_2.q_proj").replace(".attn2.to_k", ".attention_2.k_proj")
                k = k.replace(".attn2.to_v", ".attention_2.v_proj").replace(".attn2.to_out.0", ".attention_2.out_proj")
                k = k.replace(".attn1.to_out.0", ".attention_1.out_proj")
                
                # Special Case: Handle self-attention concatenation for in_proj
                if ".attn1.to_q" in k or ".attn1.to_k" in k or ".attn1.to_v" in k:
                    base_name = k.rsplit(".attn1.to_", 1)[0]
                    param_type = "weight" if "weight" in k else "bias"
                    comp_type = k.split(".to_")[-1].split(".")[0] # q, k, or v
                    
                    target_key = f"{base_name}.attention_1.in_proj.{param_type}"
                    if target_key not in attn_parts: attn_parts[target_key] = {}
                    attn_parts[target_key][comp_type] = value
                    continue # Do not add to new_sd yet

            # Norms and projections inside blocks
            k = k.replace(".norm.", ".groupnorm.")
            k = k.replace(".proj_in.", ".conv_input.")
            k = k.replace(".proj_out.", ".conv_output.")
            
'''


# Validation Logic
# Assuming 'original_sd' is your input dictionary
# mapped_sd = map_state_dict(original_sd)

def ControlnetModelConverter(filePath:str)->dict[str,torch.Tensor]:
    ldmDict = torch.load("../models/ControlNet-v1-1/control_v11p_sd15_canny.pth", map_location='cpu')
    #for key,value in controlDict.items():
    #    print(key,value.shape)    
    controlDict = convert_ldm_to_diffusers(ldmDict)

    return controlDict


'''
    ctl = {}
    tmb ={}
    outDict ={}
    ctl['cannyToLatent.0.weight'] = controlDict['control_model.input_hint_block.0.weight']
    ctl['cannyToLatent.0.bias'] = controlDict['control_model.input_hint_block.0.bias']
    ctl['cannyToLatent.2.weight'] = controlDict['control_model.input_hint_block.2.weight']
    ctl['cannyToLatent.2.bias'] = controlDict['control_model.input_hint_block.2.bias']
    ctl['cannyToLatent.4.weight'] = controlDict['control_model.input_hint_block.4.weight']
    ctl['cannyToLatent.4.bias'] = controlDict['control_model.input_hint_block.4.bias']
    ctl['cannyToLatent.6.weight'] = controlDict['control_model.input_hint_block.6.weight']
    ctl['cannyToLatent.6.bias'] = controlDict['control_model.input_hint_block.6.bias']
    ctl['cannyToLatent.8.weight'] = controlDict['control_model.input_hint_block.8.weight']
    ctl['cannyToLatent.8.bias'] = controlDict['control_model.input_hint_block.8.bias']
    ctl['cannyToLatent.10.weight'] = controlDict['control_model.input_hint_block.10.weight']
    ctl['cannyToLatent.10.bias'] = controlDict['control_model.input_hint_block.10.bias']
    ctl['cannyToLatent.12.weight'] = controlDict['control_model.input_hint_block.12.weight']
    ctl['cannyToLatent.12.bias'] = controlDict['control_model.input_hint_block.12.bias']
    ctl['cannyToLatent.14.zeroConv.weight'] = controlDict['control_model.input_hint_block.14.weight']
    ctl['cannyToLatent.14.zeroConv.bias'] = controlDict['control_model.input_hint_block.14.bias']
    tmb['linear_1.weight']=controlDict['control_model.time_embed.0.weight']
    tmb['linear_1.bias']=controlDict['control_model.time_embed.0.bias']
    tmb['linear_2.weight']=controlDict['control_model.time_embed.2.weight']
    tmb['linear_2.bias']=controlDict['control_model.time_embed.2.bias']
    
    for key,value in controlDict.items():
        if key.startswith('control_model.input_blocks.'):
            newKey = key.replace('control_model.input_blocks.','input_blocks.')
            outDict[newKey]=controlDict[key]
        
        if key.startswith('control_model.middle_block.'):
            newKey = key.replace('control_model.middle_block.','middle_block.')
            outDict[newKey]=controlDict[key]
        
        if key.startswith('control_model.middle_block_out.0.'):
            newKey = key.replace('control_model.middle_block_out.0.','middle_block_out.zeroConv.')
            outDict[newKey]=controlDict[key]
        
        if key.startswith('control_model.time_embed.'):
            pass 
        
        if key.startswith('control_model.zero_convs.'):
            newKey = key.replace('control_model.zero_convs.','zero_convs.')
        

    return ctl,tmb


'''

    
'''
control_model.input_hint_block.0.weight torch.Size([16, 3, 3, 3])
control_model.input_hint_block.0.bias torch.Size([16])
control_model.input_hint_block.2.weight torch.Size([16, 16, 3, 3])
control_model.input_hint_block.2.bias torch.Size([16])
control_model.input_hint_block.4.weight torch.Size([32, 16, 3, 3])
control_model.input_hint_block.4.bias torch.Size([32])
control_model.input_hint_block.6.weight torch.Size([32, 32, 3, 3])
control_model.input_hint_block.6.bias torch.Size([32])
control_model.input_hint_block.8.weight torch.Size([96, 32, 3, 3])
control_model.input_hint_block.8.bias torch.Size([96])
control_model.input_hint_block.10.weight torch.Size([96, 96, 3, 3])
control_model.input_hint_block.10.bias torch.Size([96])
control_model.input_hint_block.12.weight torch.Size([256, 96, 3, 3])
control_model.input_hint_block.12.bias torch.Size([256])
control_model.input_hint_block.14.weight torch.Size([320, 256, 3, 3])
control_model.input_hint_block.14.bias torch.Size([320])

control_model.time_embed.0.weight torch.Size([1280, 320])
control_model.time_embed.0.bias torch.Size([1280])
control_model.time_embed.2.weight torch.Size([1280, 1280])
control_model.time_embed.2.bias torch.Size([1280])    
    '''
    
'''
control_model.time_embed.0.weight torch.Size([1280, 320])
control_model.time_embed.0.bias torch.Size([1280])
control_model.time_embed.2.weight torch.Size([1280, 1280])
control_model.time_embed.2.bias torch.Size([1280])
control_model.input_blocks.0.0.weight torch.Size([320, 4, 3, 3])
control_model.input_blocks.0.0.bias torch.Size([320])
control_model.input_blocks.1.0.in_layers.0.weight torch.Size([320])
control_model.input_blocks.1.0.in_layers.0.bias torch.Size([320])
control_model.input_blocks.1.0.in_layers.2.weight torch.Size([320, 320, 3, 3])
control_model.input_blocks.1.0.in_layers.2.bias torch.Size([320])
control_model.input_blocks.1.0.emb_layers.1.weight torch.Size([320, 1280])
control_model.input_blocks.1.0.emb_layers.1.bias torch.Size([320])
control_model.input_blocks.1.0.out_layers.0.weight torch.Size([320])
control_model.input_blocks.1.0.out_layers.0.bias torch.Size([320])
control_model.input_blocks.1.0.out_layers.3.weight torch.Size([320, 320, 3, 3])
control_model.input_blocks.1.0.out_layers.3.bias torch.Size([320])
control_model.input_blocks.1.1.norm.weight torch.Size([320])
control_model.input_blocks.1.1.norm.bias torch.Size([320])
control_model.input_blocks.1.1.proj_in.weight torch.Size([320, 320, 1, 1])
control_model.input_blocks.1.1.proj_in.bias torch.Size([320])
control_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight torch.Size([320, 320])
control_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight torch.Size([320, 320])
control_model.input_blocks.1.1.transformer_blocks.0.attn1.to_v.weight torch.Size([320, 320])
control_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.weight torch.Size([320, 320])
control_model.input_blocks.1.1.transformer_blocks.0.attn1.to_out.0.bias torch.Size([320])
control_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.weight torch.Size([2560, 320])
control_model.input_blocks.1.1.transformer_blocks.0.ff.net.0.proj.bias torch.Size([2560])
control_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.weight torch.Size([320, 1280])
control_model.input_blocks.1.1.transformer_blocks.0.ff.net.2.bias torch.Size([320])
control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight torch.Size([320, 320])
control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight torch.Size([320, 768])
control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight torch.Size([320, 768])
control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight torch.Size([320, 320])
control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.bias torch.Size([320])
control_model.input_blocks.1.1.transformer_blocks.0.norm1.weight torch.Size([320])
control_model.input_blocks.1.1.transformer_blocks.0.norm1.bias torch.Size([320])
control_model.input_blocks.1.1.transformer_blocks.0.norm2.weight torch.Size([320])
control_model.input_blocks.1.1.transformer_blocks.0.norm2.bias torch.Size([320])
control_model.input_blocks.1.1.transformer_blocks.0.norm3.weight torch.Size([320])
control_model.input_blocks.1.1.transformer_blocks.0.norm3.bias torch.Size([320])
control_model.input_blocks.1.1.proj_out.weight torch.Size([320, 320, 1, 1])
control_model.input_blocks.1.1.proj_out.bias torch.Size([320])
control_model.input_blocks.2.0.in_layers.0.weight torch.Size([320])
control_model.input_blocks.2.0.in_layers.0.bias torch.Size([320])
control_model.input_blocks.2.0.in_layers.2.weight torch.Size([320, 320, 3, 3])
control_model.input_blocks.2.0.in_layers.2.bias torch.Size([320])
control_model.input_blocks.2.0.emb_layers.1.weight torch.Size([320, 1280])
control_model.input_blocks.2.0.emb_layers.1.bias torch.Size([320])
control_model.input_blocks.2.0.out_layers.0.weight torch.Size([320])
control_model.input_blocks.2.0.out_layers.0.bias torch.Size([320])
control_model.input_blocks.2.0.out_layers.3.weight torch.Size([320, 320, 3, 3])
control_model.input_blocks.2.0.out_layers.3.bias torch.Size([320])
control_model.input_blocks.2.1.norm.weight torch.Size([320])
control_model.input_blocks.2.1.norm.bias torch.Size([320])
control_model.input_blocks.2.1.proj_in.weight torch.Size([320, 320, 1, 1])
control_model.input_blocks.2.1.proj_in.bias torch.Size([320])
control_model.input_blocks.2.1.transformer_blocks.0.attn1.to_q.weight torch.Size([320, 320])
control_model.input_blocks.2.1.transformer_blocks.0.attn1.to_k.weight torch.Size([320, 320])
control_model.input_blocks.2.1.transformer_blocks.0.attn1.to_v.weight torch.Size([320, 320])
control_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.weight torch.Size([320, 320])
control_model.input_blocks.2.1.transformer_blocks.0.attn1.to_out.0.bias torch.Size([320])
control_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.weight torch.Size([2560, 320])
control_model.input_blocks.2.1.transformer_blocks.0.ff.net.0.proj.bias torch.Size([2560])
control_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.weight torch.Size([320, 1280])
control_model.input_blocks.2.1.transformer_blocks.0.ff.net.2.bias torch.Size([320])
control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_q.weight torch.Size([320, 320])
control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight torch.Size([320, 768])
control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_v.weight torch.Size([320, 768])
control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.weight torch.Size([320, 320])
control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_out.0.bias torch.Size([320])
control_model.input_blocks.2.1.transformer_blocks.0.norm1.weight torch.Size([320])
control_model.input_blocks.2.1.transformer_blocks.0.norm1.bias torch.Size([320])
control_model.input_blocks.2.1.transformer_blocks.0.norm2.weight torch.Size([320])
control_model.input_blocks.2.1.transformer_blocks.0.norm2.bias torch.Size([320])
control_model.input_blocks.2.1.transformer_blocks.0.norm3.weight torch.Size([320])
control_model.input_blocks.2.1.transformer_blocks.0.norm3.bias torch.Size([320])
control_model.input_blocks.2.1.proj_out.weight torch.Size([320, 320, 1, 1])
control_model.input_blocks.2.1.proj_out.bias torch.Size([320])
control_model.input_blocks.3.0.op.weight torch.Size([320, 320, 3, 3])
control_model.input_blocks.3.0.op.bias torch.Size([320])
control_model.input_blocks.4.0.in_layers.0.weight torch.Size([320])
control_model.input_blocks.4.0.in_layers.0.bias torch.Size([320])
control_model.input_blocks.4.0.in_layers.2.weight torch.Size([640, 320, 3, 3])
control_model.input_blocks.4.0.in_layers.2.bias torch.Size([640])
control_model.input_blocks.4.0.emb_layers.1.weight torch.Size([640, 1280])
control_model.input_blocks.4.0.emb_layers.1.bias torch.Size([640])
control_model.input_blocks.4.0.out_layers.0.weight torch.Size([640])
control_model.input_blocks.4.0.out_layers.0.bias torch.Size([640])
control_model.input_blocks.4.0.out_layers.3.weight torch.Size([640, 640, 3, 3])
control_model.input_blocks.4.0.out_layers.3.bias torch.Size([640])
control_model.input_blocks.4.0.skip_connection.weight torch.Size([640, 320, 1, 1])
control_model.input_blocks.4.0.skip_connection.bias torch.Size([640])
control_model.input_blocks.4.1.norm.weight torch.Size([640])
control_model.input_blocks.4.1.norm.bias torch.Size([640])
control_model.input_blocks.4.1.proj_in.weight torch.Size([640, 640, 1, 1])
control_model.input_blocks.4.1.proj_in.bias torch.Size([640])
control_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight torch.Size([640, 640])
control_model.input_blocks.4.1.transformer_blocks.0.attn1.to_k.weight torch.Size([640, 640])
control_model.input_blocks.4.1.transformer_blocks.0.attn1.to_v.weight torch.Size([640, 640])
control_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.weight torch.Size([640, 640])
control_model.input_blocks.4.1.transformer_blocks.0.attn1.to_out.0.bias torch.Size([640])
control_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.weight torch.Size([5120, 640])
control_model.input_blocks.4.1.transformer_blocks.0.ff.net.0.proj.bias torch.Size([5120])
control_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.weight torch.Size([640, 2560])
control_model.input_blocks.4.1.transformer_blocks.0.ff.net.2.bias torch.Size([640])
control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_q.weight torch.Size([640, 640])
control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight torch.Size([640, 768])
control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v.weight torch.Size([640, 768])
control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight torch.Size([640, 640])
control_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.bias torch.Size([640])
control_model.input_blocks.4.1.transformer_blocks.0.norm1.weight torch.Size([640])
control_model.input_blocks.4.1.transformer_blocks.0.norm1.bias torch.Size([640])
control_model.input_blocks.4.1.transformer_blocks.0.norm2.weight torch.Size([640])
control_model.input_blocks.4.1.transformer_blocks.0.norm2.bias torch.Size([640])
control_model.input_blocks.4.1.transformer_blocks.0.norm3.weight torch.Size([640])
control_model.input_blocks.4.1.transformer_blocks.0.norm3.bias torch.Size([640])
control_model.input_blocks.4.1.proj_out.weight torch.Size([640, 640, 1, 1])
control_model.input_blocks.4.1.proj_out.bias torch.Size([640])
control_model.input_blocks.5.0.in_layers.0.weight torch.Size([640])
control_model.input_blocks.5.0.in_layers.0.bias torch.Size([640])
control_model.input_blocks.5.0.in_layers.2.weight torch.Size([640, 640, 3, 3])
control_model.input_blocks.5.0.in_layers.2.bias torch.Size([640])
control_model.input_blocks.5.0.emb_layers.1.weight torch.Size([640, 1280])
control_model.input_blocks.5.0.emb_layers.1.bias torch.Size([640])
control_model.input_blocks.5.0.out_layers.0.weight torch.Size([640])
control_model.input_blocks.5.0.out_layers.0.bias torch.Size([640])
control_model.input_blocks.5.0.out_layers.3.weight torch.Size([640, 640, 3, 3])
control_model.input_blocks.5.0.out_layers.3.bias torch.Size([640])
control_model.input_blocks.5.1.norm.weight torch.Size([640])
control_model.input_blocks.5.1.norm.bias torch.Size([640])
control_model.input_blocks.5.1.proj_in.weight torch.Size([640, 640, 1, 1])
control_model.input_blocks.5.1.proj_in.bias torch.Size([640])
control_model.input_blocks.5.1.transformer_blocks.0.attn1.to_q.weight torch.Size([640, 640])
control_model.input_blocks.5.1.transformer_blocks.0.attn1.to_k.weight torch.Size([640, 640])
control_model.input_blocks.5.1.transformer_blocks.0.attn1.to_v.weight torch.Size([640, 640])
control_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.weight torch.Size([640, 640])
control_model.input_blocks.5.1.transformer_blocks.0.attn1.to_out.0.bias torch.Size([640])
control_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.weight torch.Size([5120, 640])
control_model.input_blocks.5.1.transformer_blocks.0.ff.net.0.proj.bias torch.Size([5120])
control_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.weight torch.Size([640, 2560])
control_model.input_blocks.5.1.transformer_blocks.0.ff.net.2.bias torch.Size([640])
control_model.input_blocks.5.1.transformer_blocks.0.attn2.to_q.weight torch.Size([640, 640])
control_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k.weight torch.Size([640, 768])
control_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v.weight torch.Size([640, 768])
control_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.weight torch.Size([640, 640])
control_model.input_blocks.5.1.transformer_blocks.0.attn2.to_out.0.bias torch.Size([640])
control_model.input_blocks.5.1.transformer_blocks.0.norm1.weight torch.Size([640])
control_model.input_blocks.5.1.transformer_blocks.0.norm1.bias torch.Size([640])
control_model.input_blocks.5.1.transformer_blocks.0.norm2.weight torch.Size([640])
control_model.input_blocks.5.1.transformer_blocks.0.norm2.bias torch.Size([640])
control_model.input_blocks.5.1.transformer_blocks.0.norm3.weight torch.Size([640])
control_model.input_blocks.5.1.transformer_blocks.0.norm3.bias torch.Size([640])
control_model.input_blocks.5.1.proj_out.weight torch.Size([640, 640, 1, 1])
control_model.input_blocks.5.1.proj_out.bias torch.Size([640])
control_model.input_blocks.6.0.op.weight torch.Size([640, 640, 3, 3])
control_model.input_blocks.6.0.op.bias torch.Size([640])
control_model.input_blocks.7.0.in_layers.0.weight torch.Size([640])
control_model.input_blocks.7.0.in_layers.0.bias torch.Size([640])
control_model.input_blocks.7.0.in_layers.2.weight torch.Size([1280, 640, 3, 3])
control_model.input_blocks.7.0.in_layers.2.bias torch.Size([1280])
control_model.input_blocks.7.0.emb_layers.1.weight torch.Size([1280, 1280])
control_model.input_blocks.7.0.emb_layers.1.bias torch.Size([1280])
control_model.input_blocks.7.0.out_layers.0.weight torch.Size([1280])
control_model.input_blocks.7.0.out_layers.0.bias torch.Size([1280])
control_model.input_blocks.7.0.out_layers.3.weight torch.Size([1280, 1280, 3, 3])
control_model.input_blocks.7.0.out_layers.3.bias torch.Size([1280])
control_model.input_blocks.7.0.skip_connection.weight torch.Size([1280, 640, 1, 1])
control_model.input_blocks.7.0.skip_connection.bias torch.Size([1280])
control_model.input_blocks.7.1.norm.weight torch.Size([1280])
control_model.input_blocks.7.1.norm.bias torch.Size([1280])
control_model.input_blocks.7.1.proj_in.weight torch.Size([1280, 1280, 1, 1])
control_model.input_blocks.7.1.proj_in.bias torch.Size([1280])
control_model.input_blocks.7.1.transformer_blocks.0.attn1.to_q.weight torch.Size([1280, 1280])
control_model.input_blocks.7.1.transformer_blocks.0.attn1.to_k.weight torch.Size([1280, 1280])
control_model.input_blocks.7.1.transformer_blocks.0.attn1.to_v.weight torch.Size([1280, 1280])
control_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.weight torch.Size([1280, 1280])
control_model.input_blocks.7.1.transformer_blocks.0.attn1.to_out.0.bias torch.Size([1280])
control_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.weight torch.Size([10240, 1280])
control_model.input_blocks.7.1.transformer_blocks.0.ff.net.0.proj.bias torch.Size([10240])
control_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.weight torch.Size([1280, 5120])
control_model.input_blocks.7.1.transformer_blocks.0.ff.net.2.bias torch.Size([1280])
control_model.input_blocks.7.1.transformer_blocks.0.attn2.to_q.weight torch.Size([1280, 1280])
control_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k.weight torch.Size([1280, 768])
control_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v.weight torch.Size([1280, 768])
control_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.weight torch.Size([1280, 1280])
control_model.input_blocks.7.1.transformer_blocks.0.attn2.to_out.0.bias torch.Size([1280])
control_model.input_blocks.7.1.transformer_blocks.0.norm1.weight torch.Size([1280])
control_model.input_blocks.7.1.transformer_blocks.0.norm1.bias torch.Size([1280])
control_model.input_blocks.7.1.transformer_blocks.0.norm2.weight torch.Size([1280])
control_model.input_blocks.7.1.transformer_blocks.0.norm2.bias torch.Size([1280])
control_model.input_blocks.7.1.transformer_blocks.0.norm3.weight torch.Size([1280])
control_model.input_blocks.7.1.transformer_blocks.0.norm3.bias torch.Size([1280])
control_model.input_blocks.7.1.proj_out.weight torch.Size([1280, 1280, 1, 1])
control_model.input_blocks.7.1.proj_out.bias torch.Size([1280])
control_model.input_blocks.8.0.in_layers.0.weight torch.Size([1280])
control_model.input_blocks.8.0.in_layers.0.bias torch.Size([1280])
control_model.input_blocks.8.0.in_layers.2.weight torch.Size([1280, 1280, 3, 3])
control_model.input_blocks.8.0.in_layers.2.bias torch.Size([1280])
control_model.input_blocks.8.0.emb_layers.1.weight torch.Size([1280, 1280])
control_model.input_blocks.8.0.emb_layers.1.bias torch.Size([1280])
control_model.input_blocks.8.0.out_layers.0.weight torch.Size([1280])
control_model.input_blocks.8.0.out_layers.0.bias torch.Size([1280])
control_model.input_blocks.8.0.out_layers.3.weight torch.Size([1280, 1280, 3, 3])
control_model.input_blocks.8.0.out_layers.3.bias torch.Size([1280])
control_model.input_blocks.8.1.norm.weight torch.Size([1280])
control_model.input_blocks.8.1.norm.bias torch.Size([1280])
control_model.input_blocks.8.1.proj_in.weight torch.Size([1280, 1280, 1, 1])
control_model.input_blocks.8.1.proj_in.bias torch.Size([1280])
control_model.input_blocks.8.1.transformer_blocks.0.attn1.to_q.weight torch.Size([1280, 1280])
control_model.input_blocks.8.1.transformer_blocks.0.attn1.to_k.weight torch.Size([1280, 1280])
control_model.input_blocks.8.1.transformer_blocks.0.attn1.to_v.weight torch.Size([1280, 1280])
control_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.weight torch.Size([1280, 1280])
control_model.input_blocks.8.1.transformer_blocks.0.attn1.to_out.0.bias torch.Size([1280])
control_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight torch.Size([10240, 1280])
control_model.input_blocks.8.1.transformer_blocks.0.ff.net.0.proj.bias torch.Size([10240])
control_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.weight torch.Size([1280, 5120])
control_model.input_blocks.8.1.transformer_blocks.0.ff.net.2.bias torch.Size([1280])
control_model.input_blocks.8.1.transformer_blocks.0.attn2.to_q.weight torch.Size([1280, 1280])
control_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k.weight torch.Size([1280, 768])
control_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v.weight torch.Size([1280, 768])
control_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.weight torch.Size([1280, 1280])
control_model.input_blocks.8.1.transformer_blocks.0.attn2.to_out.0.bias torch.Size([1280])
control_model.input_blocks.8.1.transformer_blocks.0.norm1.weight torch.Size([1280])
control_model.input_blocks.8.1.transformer_blocks.0.norm1.bias torch.Size([1280])
control_model.input_blocks.8.1.transformer_blocks.0.norm2.weight torch.Size([1280])
control_model.input_blocks.8.1.transformer_blocks.0.norm2.bias torch.Size([1280])
control_model.input_blocks.8.1.transformer_blocks.0.norm3.weight torch.Size([1280])
control_model.input_blocks.8.1.transformer_blocks.0.norm3.bias torch.Size([1280])
control_model.input_blocks.8.1.proj_out.weight torch.Size([1280, 1280, 1, 1])
control_model.input_blocks.8.1.proj_out.bias torch.Size([1280])
control_model.input_blocks.9.0.op.weight torch.Size([1280, 1280, 3, 3])
control_model.input_blocks.9.0.op.bias torch.Size([1280])
control_model.input_blocks.10.0.in_layers.0.weight torch.Size([1280])
control_model.input_blocks.10.0.in_layers.0.bias torch.Size([1280])
control_model.input_blocks.10.0.in_layers.2.weight torch.Size([1280, 1280, 3, 3])
control_model.input_blocks.10.0.in_layers.2.bias torch.Size([1280])
control_model.input_blocks.10.0.emb_layers.1.weight torch.Size([1280, 1280])
control_model.input_blocks.10.0.emb_layers.1.bias torch.Size([1280])
control_model.input_blocks.10.0.out_layers.0.weight torch.Size([1280])
control_model.input_blocks.10.0.out_layers.0.bias torch.Size([1280])
control_model.input_blocks.10.0.out_layers.3.weight torch.Size([1280, 1280, 3, 3])
control_model.input_blocks.10.0.out_layers.3.bias torch.Size([1280])
control_model.input_blocks.11.0.in_layers.0.weight torch.Size([1280])
control_model.input_blocks.11.0.in_layers.0.bias torch.Size([1280])
control_model.input_blocks.11.0.in_layers.2.weight torch.Size([1280, 1280, 3, 3])
control_model.input_blocks.11.0.in_layers.2.bias torch.Size([1280])
control_model.input_blocks.11.0.emb_layers.1.weight torch.Size([1280, 1280])
control_model.input_blocks.11.0.emb_layers.1.bias torch.Size([1280])
control_model.input_blocks.11.0.out_layers.0.weight torch.Size([1280])
control_model.input_blocks.11.0.out_layers.0.bias torch.Size([1280])
control_model.input_blocks.11.0.out_layers.3.weight torch.Size([1280, 1280, 3, 3])
control_model.input_blocks.11.0.out_layers.3.bias torch.Size([1280])
control_model.zero_convs.0.0.weight torch.Size([320, 320, 1, 1])
control_model.zero_convs.0.0.bias torch.Size([320])
control_model.zero_convs.1.0.weight torch.Size([320, 320, 1, 1])
control_model.zero_convs.1.0.bias torch.Size([320])
control_model.zero_convs.2.0.weight torch.Size([320, 320, 1, 1])
control_model.zero_convs.2.0.bias torch.Size([320])
control_model.zero_convs.3.0.weight torch.Size([320, 320, 1, 1])
control_model.zero_convs.3.0.bias torch.Size([320])
control_model.zero_convs.4.0.weight torch.Size([640, 640, 1, 1])
control_model.zero_convs.4.0.bias torch.Size([640])
control_model.zero_convs.5.0.weight torch.Size([640, 640, 1, 1])
control_model.zero_convs.5.0.bias torch.Size([640])
control_model.zero_convs.6.0.weight torch.Size([640, 640, 1, 1])
control_model.zero_convs.6.0.bias torch.Size([640])
control_model.zero_convs.7.0.weight torch.Size([1280, 1280, 1, 1])
control_model.zero_convs.7.0.bias torch.Size([1280])
control_model.zero_convs.8.0.weight torch.Size([1280, 1280, 1, 1])
control_model.zero_convs.8.0.bias torch.Size([1280])
control_model.zero_convs.9.0.weight torch.Size([1280, 1280, 1, 1])
control_model.zero_convs.9.0.bias torch.Size([1280])
control_model.zero_convs.10.0.weight torch.Size([1280, 1280, 1, 1])
control_model.zero_convs.10.0.bias torch.Size([1280])
control_model.zero_convs.11.0.weight torch.Size([1280, 1280, 1, 1])
control_model.zero_convs.11.0.bias torch.Size([1280])
control_model.input_hint_block.0.weight torch.Size([16, 3, 3, 3])
control_model.input_hint_block.0.bias torch.Size([16])
control_model.input_hint_block.2.weight torch.Size([16, 16, 3, 3])
control_model.input_hint_block.2.bias torch.Size([16])
control_model.input_hint_block.4.weight torch.Size([32, 16, 3, 3])
control_model.input_hint_block.4.bias torch.Size([32])
control_model.input_hint_block.6.weight torch.Size([32, 32, 3, 3])
control_model.input_hint_block.6.bias torch.Size([32])
control_model.input_hint_block.8.weight torch.Size([96, 32, 3, 3])
control_model.input_hint_block.8.bias torch.Size([96])
control_model.input_hint_block.10.weight torch.Size([96, 96, 3, 3])
control_model.input_hint_block.10.bias torch.Size([96])
control_model.input_hint_block.12.weight torch.Size([256, 96, 3, 3])
control_model.input_hint_block.12.bias torch.Size([256])
control_model.input_hint_block.14.weight torch.Size([320, 256, 3, 3])
control_model.input_hint_block.14.bias torch.Size([320])
control_model.middle_block.0.in_layers.0.weight torch.Size([1280])
control_model.middle_block.0.in_layers.0.bias torch.Size([1280])
control_model.middle_block.0.in_layers.2.weight torch.Size([1280, 1280, 3, 3])
control_model.middle_block.0.in_layers.2.bias torch.Size([1280])
control_model.middle_block.0.emb_layers.1.weight torch.Size([1280, 1280])
control_model.middle_block.0.emb_layers.1.bias torch.Size([1280])
control_model.middle_block.0.out_layers.0.weight torch.Size([1280])
control_model.middle_block.0.out_layers.0.bias torch.Size([1280])
control_model.middle_block.0.out_layers.3.weight torch.Size([1280, 1280, 3, 3])
control_model.middle_block.0.out_layers.3.bias torch.Size([1280])
control_model.middle_block.1.norm.weight torch.Size([1280])
control_model.middle_block.1.norm.bias torch.Size([1280])
control_model.middle_block.1.proj_in.weight torch.Size([1280, 1280, 1, 1])
control_model.middle_block.1.proj_in.bias torch.Size([1280])
control_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight torch.Size([1280, 1280])
control_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight torch.Size([1280, 1280])
control_model.middle_block.1.transformer_blocks.0.attn1.to_v.weight torch.Size([1280, 1280])
control_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.weight torch.Size([1280, 1280])
control_model.middle_block.1.transformer_blocks.0.attn1.to_out.0.bias torch.Size([1280])
control_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.weight torch.Size([10240, 1280])
control_model.middle_block.1.transformer_blocks.0.ff.net.0.proj.bias torch.Size([10240])
control_model.middle_block.1.transformer_blocks.0.ff.net.2.weight torch.Size([1280, 5120])
control_model.middle_block.1.transformer_blocks.0.ff.net.2.bias torch.Size([1280])
control_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight torch.Size([1280, 1280])
control_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight torch.Size([1280, 768])
control_model.middle_block.1.transformer_blocks.0.attn2.to_v.weight torch.Size([1280, 768])
control_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.weight torch.Size([1280, 1280])
control_model.middle_block.1.transformer_blocks.0.attn2.to_out.0.bias torch.Size([1280])
control_model.middle_block.1.transformer_blocks.0.norm1.weight torch.Size([1280])
control_model.middle_block.1.transformer_blocks.0.norm1.bias torch.Size([1280])
control_model.middle_block.1.transformer_blocks.0.norm2.weight torch.Size([1280])
control_model.middle_block.1.transformer_blocks.0.norm2.bias torch.Size([1280])
control_model.middle_block.1.transformer_blocks.0.norm3.weight torch.Size([1280])
control_model.middle_block.1.transformer_blocks.0.norm3.bias torch.Size([1280])
control_model.middle_block.1.proj_out.weight torch.Size([1280, 1280, 1, 1])
control_model.middle_block.1.proj_out.bias torch.Size([1280])
control_model.middle_block.2.in_layers.0.weight torch.Size([1280])
control_model.middle_block.2.in_layers.0.bias torch.Size([1280])
control_model.middle_block.2.in_layers.2.weight torch.Size([1280, 1280, 3, 3])
control_model.middle_block.2.in_layers.2.bias torch.Size([1280])
control_model.middle_block.2.emb_layers.1.weight torch.Size([1280, 1280])
control_model.middle_block.2.emb_layers.1.bias torch.Size([1280])
control_model.middle_block.2.out_layers.0.weight torch.Size([1280])
control_model.middle_block.2.out_layers.0.bias torch.Size([1280])
control_model.middle_block.2.out_layers.3.weight torch.Size([1280, 1280, 3, 3])
control_model.middle_block.2.out_layers.3.bias torch.Size([1280])
control_model.middle_block_out.0.weight torch.Size([1280, 1280, 1, 1])
control_model.middle_block_out.0.bias torch.Size([1280])

'''
    
    