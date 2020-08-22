
import time
import os
from tqdm import tqdm

import torch

import tensorly as tl
tl.set_backend('pytorch')

from tdtd.model.cp import *
from tdtd.utils import *

def norm(val, rec):
    return tl.norm(val - rec, 2)/tl.norm(val, 2)

def mse(val, rec):
    return F.mse_loss(val, rec)

def rmse(val, rec):
    return torch.sqrt(F.mse_loss(val, rec))

def mae(val, rec):
    return torch.mean(torch.abs(val-rec))

def evaluate(model, data):
    with torch.no_grad():
        rec = krprod(model.factors, data.indices())
        vals = data.values()
        e1, e2, e3 = rmse(vals, rec), norm(vals,  rec), mae(vals, rec)
    return e1.cpu().item(), e2.cpu().item(), e3.cpu().item()

def train_model(model, optimizer, dataset, penalty, path_loss, path_model):
    
    count = 0
    error_list = []
    with open(path_loss, 'w') as f:
        head ='Epoch\tTrnRMSE\tTrnNorm\tTrMAE\tValRMSE\tValNorm\tValMAE\t''TestRMSE\tTestNorm\tTestMAE\tIsStop\n'
        f.write(head)
    error_list.append(head)
    print(head)
        
    train = dataset['train']
    valid = dataset['valid']
    test = dataset['test']
    
    nmode = dataset['nmode']
    tmode = dataset['tmode']
    n_iters = dataset['iters']
    
    start_time = time.time()
    
    old_rmse, stop_iter = 1e+5, -1
    
    for n_iter in range(1,n_iters + 1):
    
        for mode in range(nmode):
            
            optimizer.zero_grad()
            
            model.turn_on_grad(mode)
            
            rec = model(train.indices())

            loss = (rec - train.values()).pow(2).sum()
            
            if mode == tmode:
                loss = loss + penalty * model.smooth_reg(mode)

            loss.backward()
            
            optimizer.step()
            
            model.turn_off_grad(mode)
                
        trn_rmse, trn_norm, trn_mae = evaluate(model, train)
        val_rmse, val_norm, val_mae = evaluate(model, valid)
        test_rmse, test_norm, test_mae = evaluate(model, test)
        
        error_list.append([trn_rmse, trn_norm, trn_mae,
                           val_rmse, val_norm, val_mae, 
                           test_rmse, test_norm, test_mae ])
        
        if val_rmse > old_rmse:
            stop_iter = n_iter
            count +=1
        old_rmse = val_rmse 
        
        if (n_iter % 1000) == 0:
            print(f"{n_iter}\t{trn_rmse:.4f}\t{trn_norm:.4f}\t{trn_mae:.4f}\t"
                  f"\t{val_rmse:.4f}\t{val_norm:.4f}\t\t{val_mae:.4f}\t"
                  f"\t{test_rmse:.4f}\t{test_norm:.4f}\t{test_mae:.4f}")
        if stop_iter == n_iter and count == 1:
            print('>>>>>>>>>>>>>>>>>>>>>>>>>STOP')
        with open(path_loss, 'a') as f:
            f.write(f'{n_iter:3d}\t')
            f.write(f'{trn_rmse:.8f}\t{trn_norm:.8f}\t{trn_mae:.8f}\t')
            f.write(f'{val_rmse:.8f}\t{val_norm:.8f}\t{val_mae:.8f}\t')
            f.write(f'{test_rmse:.8f}\t{test_norm:.8f}\t{test_mae:.8f}')
            if n_iter == stop_iter:
                f.write('\tSTOP')
            f.write('\n')
        if n_iter == stop_iter and n_iter > 1000 and count < 10:
            p = f'{path_model}-best.pth.tar'
            save_checkpoints(model, p)
            
        
    end_time = time.time()
    total_time = end_time - start_time
    print('Took %d mins %d secs' % (divmod(total_time, 60)))
    return error_list    
