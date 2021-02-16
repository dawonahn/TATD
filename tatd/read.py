
import os
import torch
import pandas as pd
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_index(length, w1, w2):

    return np.arange(-w2, w2+1)[None, :] + np.arange(length+1)[:, None]

def empty_slice(df, tmode):
    
    e_idx = set(np.arange(0, df[tmode].max()+1))
    
    df = df.groupby(tmode).count()
    df = pd.DataFrame(df.iloc[:, 1]).reset_index()
    df.columns = [0,1]
    subset = set(df[0])
    diff = e_idx - subset
    if len(diff) != 0:
        lst = [[i, 0] for i in diff]
        df = df.append(pd.DataFrame(lst))
    df = df.sort_values(by = 0)
    df = df.reset_index(drop = True)
    return df

def local_sparsity(data, tmode, w, mmin = 0.1, mmax = 0.9):
    
    npy = data.indices().cpu().numpy()
    df = pd.DataFrame(npy.T)
    
    tmp = empty_slice(df, tmode)
    tmp['max_'] = 0
    
    max_ = tmp[[0]].max()[0]
    w1 = w
    w2 = int((w1-1)/2)
    idxs = make_index(max_, w1, w2)

    idxs = np.where(idxs > 0, idxs, 0)
    idxs = np.where(idxs < max_, idxs, max_)

    for i in range(len(idxs)):
        tmp.at[i, 'max_'] = list(tmp.iloc[idxs[i]].max())[-2]

    ls = tmp[1]/tmp['max_']
    ls2 = (mmax - mmin) * (ls - ls.min()) / (ls.max() - ls.min())
    ls2 = np.where(np.isnan(ls2), 1, ls2)
    density = torch.FloatTensor(1 -ls2).to(DEVICE)
    return density

def global_sparsity(data, tmode):

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
    max_, min_ = dff.max(), dff.min()

    min_max = (0.999 - 0.001) * (dff - min_)/(max_ - min_) 
    min_max = np.where(np.isnan(min_max), 1, min_max)
    return 1 - torch.FloatTensor(list(min_max+ 0.001)).to(DEVICE)
#    min_max = (0.9 - 0.1) * (dff - min_)/(max_ - min_) 
#    return 1 - torch.FloatTensor(list(min_max+ 0.1)).to(DEVICE)


def read_tensor(path, name, dtype):

    ''' Read COO format tensor (sparse format) '''

    i =  torch.LongTensor(np.load(os.path.join(path, name, dtype + '_idxs.npy')))
    
    v = torch.FloatTensor(np.load(os.path.join(path, name, dtype +'_vals.npy')))
    stensor = torch.sparse.FloatTensor(i.t(), v).coalesce()
    return stensor

def read_dataset(name, window, path = '../data'):
    
    ''' Read data and make metadata '''
    
    dct = {}
    dct['name'] = name

    for dtype in ['train', 'valid', 'test']:
        dct[dtype] = {}
        stensor = read_tensor(path, name, dtype)
        dct[dtype] = stensor.to(DEVICE)
    
    dct['tmode'] = 0 ### Default value
    dct['nmode'] = len(stensor.shape)
    dct['ndim'] = max(dct['train'].shape, dct['valid'].shape, dct['test'].shape)
    dct['g_beta'] = global_sparsity(dct['train'], dct['tmode'])
    #dct['l_beta'] = local_sparsity(dct['train'], dct['tmode'], window,)

    return dct

    
    
