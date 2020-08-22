
import os
import torch
from dotmap import DotMap


def get_path(name, sparse, rrank, window, penalty, opt_scheme, lr, count):
    
    path = f'../out/{opt_scheme}/{name}/training/'
    path1= f'../out/{opt_scheme}/{name}/model/'

    if not os.path.isdir(path):
        os.makedirs(path)
    if not os.path.isdir(path1):
        os.makedirs(path1)
    
    info = f'{sparse}_r_{rrank}_w_{window}_p_{penalty}_lr_{lr}_{count}'
    
    loss_path = os.path.join(path, info + '.txt')

    model_path = os.path.join(path1, info)
        
    total_loss = f'../out/{opt_scheme}/{name}/best.txt'
    
    if not os.path.exists(total_loss):
        with open(total_loss, 'w') as f:
            f.write('No.\titers\ttime\tsparse\trank\twindow\tpenalty\tscheme\tlr\trmse\tmae\n')
        f.close()

    return loss_path, model_path, total_loss


def get_device(gpu = None):
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.cuda.is_avaliable and gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')
    return device

def save_checkpoints(model, path):
    """
    Save a trained model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(model_state=model.state_dict()), path)
