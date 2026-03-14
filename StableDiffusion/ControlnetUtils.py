import os    
import torch 
import torch.nn as nn
class ControlnetUtils:
        
    @staticmethod
    def get_readable_size(file_path)->str:
        size_bytes = os.path.getsize(file_path)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
    
    @staticmethod
    def writeControlnetToFile(model:nn.Module)->None:
        
        #for name,param in model.named_parameters():
        #    print(name,param.shape)
        loraDict={}
        for name,param in model.named_parameters():
            if 'controlnetOutput.'in name and param.requires_grad==True :
                loraDict[name]=param
                print(name,param.shape)
                
        for name,param in loraDict.items():
            print(name,param.shape)    
        path = '/home/aistudio/models/training'
        tmpPath = os.path.join(path, 'controlnetCanny.ckpt.tmp')
        finalPath = os.path.join(path, 'controlnetCanny.ckpt')
        torch.save(loraDict,tmpPath)
        os.replace(tmpPath,finalPath) 
        print(f'model size on disk : {ControlnetUtils.get_readable_size(finalPath)}')
    

  
        


        