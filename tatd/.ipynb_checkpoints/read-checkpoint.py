
import os
import torch
import pandas as pd
import numpy as np



def make_sparse_info(data, nmode , tmode, device):
    
    mode_idx = [i for i in range(nmode) if i!= tmode]
    
    npy = data.indices().cpu().numpy()
    
    df = pd.DataFrame(npy.T)
    
    tmp = df.groupby(tmode).count()
    
    tmp = tmp.iloc[:, 1]    
    max_ = tmp.max()
    min_  = tmp.min()
    
    times = (tmp - min_)/max_
    
    df = df.groupby(tmode).nunique().iloc[:, 1:]    
    df = df.values
    
    max_ = df.max()
    min_  = df.min()
    
    ntimes = (df - min_)/max_
    
    final = pd.DataFrame(ntimes)
    final.columns = mode_idx 
    final[tmode] = times

    vals =  final[[0, 1, 2]].values

    return 1 - torch.FloatTensor(vals).to(device)



def make_sparsity(data, tmode, device):
    npy = data.indices().cpu().numpy()
    df = pd.DataFrame(npy.T)
    df = df.groupby(tmode).count().iloc[:, 1]    
    max_ = df.max()
    min_  = df.min()
    return 1- torch.FloatTensor(list((df - min_)/max_)).to(device)

def make_log_sparse(data, tmode, device):
    npy = data.indices().cpu().numpy()
    df = pd.DataFrame(npy.T)
    df = df.groupby(tmode).count().iloc[:, 1]    
    max_ = df.max()
    min_  = df.min()
    return torch.abs(torch.log(torch.FloatTensor(list(df/max_)))).to(device)

def make_sparsity1(data, tmode, device):
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

        lst = [ [i, 0] for i in diff]
        df = df.append(pd.DataFrame(lst))


    df = df.sort_values(by = 0)
    
    df = df[1]
    max_ = df.max()
    min_  = df.min()
    return 1- torch.FloatTensor(list((df - min_)/max_)).to(device)

def make_inverse_sparse(data, tmode, device):
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
    try:
        inverse = (max_ / dff)
        inverse = (max_/dff).replace(np.inf, 0)
        inverse = (max_/dff).replace(-np.inf, 0)
        inverse = torch.FloatTensor(list(inverse)).to(device)

    except ZeroDivisionError:
        print("Zero detected")
        inverse = (max_/df).replace(np.inf, 0)
        inverse = torch.FloatTensor(list(inverse)).to(device)
    return inverse

def make_max_sparse(data, tmode, device):
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
    
    max_df = dff/max_
    max_df = max_df/ max_df.max()
    max_df = torch.FloatTensor(list(max_df)).to(device)
    return max_df 

def make_min_max_sparsity(data, tmode, device):
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

    return 1 - torch.FloatTensor(list((dff - min_)/(max_ - min_) + 0.001)).to(device)
def read_tensors(path, name, device):
    
    dct = {}
    dct['name'] = name
    for dtype in ['train', 'valid', 'test']:
        dct[dtype] = {}
        i =  torch.LongTensor(np.load(os.path.join(path, name, dtype + '_idxs.npy')))
        v = torch.FloatTensor(np.load(os.path.join(path, name, dtype +'_vals.npy')))
        stensor = torch.sparse.FloatTensor(i.t(), v).coalesce()
        dct[dtype] = stensor.to(device)
        
    dct['nmode'] = i.shape[1]
    dct['ndim'] = max(dct['train'].shape, dct['valid'].shape, dct['test'].shape)
    return dct                                  
    
def read_dataset(path = '../data', name = 'air10', device = 'cpu', **kargs):
    
    tensors = read_tensors(path, name, device)
    if name == 'beijing':
        tmode = 0
        lr = 0.001
    elif name == 'airq_b_7' or name == 'airq_b':
        tmode = 0
        lr = 0.00001
    elif name == 'airquality' or name == 'nairquality':
        tmode = 0
        lr = 0.001 
    elif name == 'ddos60':
        tmode = 2
        lr = 0.001 
    elif name == 'energy' or name == 'energy1':
        tmode = 0
        lr = 0.0001
    elif name == 'energy_9' or name == 'energy_1' or name == 'energy_3' or name == 'energy_5':
        tmode = 0
        lr = 0.0001
    elif name == 'ml':       
        tmode = 2
        lr = 0.001
    elif name == 'mad':       
        tmode = 0
        lr = 0.001
    elif name == 'radar':       
        tmode = 0
        lr = 0.001
    elif name == 'air_seoul':       
        tmode = 0
        lr = 0.001
    elif name.startswith('beijing_'):       
        tmode = 0
        lr = 0.0001
    elif name == 'traffic':
        tmode = 0
        lr = 0.0001
    elif name == 'ddos60':
        tmode = 2
        lr = 0.005
    elif name == 'lbnl':
        tmode = 2
        lr = 0.0001
    elif name == 'mit':
        tmode = 2
        lr = 0.001
    tensors['tmode'] = tmode
    tensors['lr'] = lr
    tensors['sparse'] = make_sparsity1(tensors['train'], tmode, device)
    tensors['inverse'] = make_inverse_sparse(tensors['train'], tmode, device)
    tensors['sparse_min_max'] = make_min_max_sparsity(tensors['train'], tmode, device)
   # tensors['log_sparse'] = make_log_sparse(tensors['train'], tmode, device)

    return tensors                    
    
