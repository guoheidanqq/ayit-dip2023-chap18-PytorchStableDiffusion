import torch

def ControlnetModelConverter(filePath:str)->dict[str,torch.Tensor]:
    controlDict = torch.load("../models/ControlNet-v1-1/control_v11p_sd15_canny.pth", map_location='cpu')
    #for key,value in controlDict.items():
    #    print(key,value.shape)    
    ctl = {}
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
    return ctl
    
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
    
    '''
    
    