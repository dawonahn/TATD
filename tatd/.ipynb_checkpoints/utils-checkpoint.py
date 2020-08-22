
import os
import torch

def get_path(name, ssmooth, ssparse, actF, rrank, window, penalty, count):
    
    path = f'../out/{name}/{ssmooth}/'
    path1= f'../out/{name}/{ssmooth}_model/'
    if not os.path.isdir(path):
        os.makedirs(path)
    if not os.path.isdir(path1):
        os.makedirs(path1)
    
    info = f'{ssparse}_r_{rrank}_w_{window}_p_{penalty}_a_{actF}_{count}'
    
    loss_path = os.path.join(path, info + '.txt')

    model_path = os.path.join(path1, info)
        
    total_loss = f'../out/{name}/{ssmooth}_ActF.txt'        
    
    if not os.path.exists(total_loss):
        with open(total_loss, 'w') as f:
            f.write('No.\tType1\tType2\tActF\tRank\tWindow\tPenalty\tStop\tRMSE\tNorm\tMAE\n')
        f.close()

    return loss_path, model_path, total_loss

#def make_sparsity(data, tmode, device):
#    npy = data.indices().cpu().numpy()
#    df = pd.DataFrame(npy).groupby(tmode).count()[1]
#    max_ = df.max()
#    return 1- torch.FloatTensor(list(df/max_)).to(device)

def get_device(gpu = None):
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.cuda.is_avaliable and gpu is not None:
        device = torch.device('cuda:{}'.format(GPU))
    else:
        device = torch.device('cpu')
    return device

def save_checkpoints(model, path):
    """
    Save a trained model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(model_state=model.state_dict()), path)
