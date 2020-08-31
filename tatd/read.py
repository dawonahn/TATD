
import os
import torch
import pandas as pd
import numpy as np



def make_min_max_sparsity(data, tmode, device):

    ''' Calculate time sparsity '''

    npy = data.indices().cpu().numpy()
    df = pd.DataFrame(npy.T)
    r_index = df[tmode].max()
    r_index = set( np.arange(0, r_index + 1 ))
    df = df.groupby(tmode).count()

    df = pd.DataFrame(df.iloc[:,1])

    df = df.reset_index()
    
    df.columns = [0,1]
    subset = set(df[0])
    diff = r_index - subset
    if len(diff) != 0 :
        lst = [ [i, 1] for i in diff]
        df = df.append(pd.DataFrame(lst))


    df = df.sort_values(by = 0)
    
    dff = df[1]
    max_ = dff.max()
    min_  = dff.min()
    min_max = (0.999 - 0.001) * (dff - min_)/(max_ - min_) 
    return 1 - torch.FloatTensor(list(min_max+ 0.001)).to(device)

def read_tensor(path, name, dtype):

    ''' Read COO format tensor (sparse format) '''

    i =  torch.LongTensor(np.load(os.path.join(path, name, dtype + '_idxs.npy')))
    v = torch.FloatTensor(np.load(os.path.join(path, name, dtype +'_vals.npy')))
    stensor = torch.sparse.FloatTensor(i.t(), v).coalesce()
    return stensor

def read_dataset(name, device, path = '../data'):
    
    ''' Read data and make metadata '''
    
    dct = {}
    dct['name'] = name
    for dtype in ['train', 'valid', 'test']:
        dct[dtype] = {}
        stensor = read_tensor(path, name, dtype)
        dct[dtype] = stensor.to(device)
    
    dct['tmode'] = 0 ### Default value
    dct['nmode'] = len(stensor.shape)
    dct['ndim'] = max(dct['train'].shape, dct['valid'].shape, dct['test'].shape)
    dct['ts_beta'] = make_min_max_sparsity(dct['train'], dct['tmode'], device)

    return dct                                  
    
