import torch
import os
from datetime import datetime
import shutil


def prepare_folder(name, model_name):
    model_dir = f'./model_results/{name}/{model_name}'
   
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir

def prepare_tune_folder(name, model_name):
    str_time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    tune_model_dir = f'./tune_results/{name}/{model_name}/{str_time}/'
   
    if os.path.exists(tune_model_dir):
        print(f'rm tune_model_dir {tune_model_dir}')
        shutil.rmtree(tune_model_dir)
    os.makedirs(tune_model_dir)
    print(f'make tune_model_dir {tune_model_dir}')
    return tune_model_dir

def save_preds_and_params(parameters, preds, model, file):
    save_dict = {'parameters':parameters, 'preds': preds, 'params': model.state_dict()
           , 'nparams': sum(p.numel() for p in model.parameters())}
    torch.save(save_dict, file)
    return 
    
    


