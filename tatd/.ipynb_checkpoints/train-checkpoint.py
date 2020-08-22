
import time
import torch
import torch.functional as F
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from tqdm import trange
from tdtd.model.cp import *
from tdtd.utils import *

def isNaN(num):
    return num != num

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

def als_train(model, opt, train, penalty, tmode, nmode):
    'als update'
    for mode in range(nmode):

        opt.zero_grad()

        model.turn_on_grad(mode)

        rec = model(train.indices())

        loss = (rec - train.values()).pow(2).sum()
        
        if mode == tmode:
            loss = loss + penalty * model.smooth_reg(mode)
        else:
            loss = loss + penalty * torch.norm(model.factors[mode])

        loss.backward()

        opt.step()

        model.turn_off_grad(mode)

    return loss
        

def train_model(model, dataset, window, penalty, path_loss, path_model, total_loss_path, count):
    
    tr = dataset['train']
    val = dataset['valid']
    test = dataset['test']
    num = 0
    if dataset['name'] == 'energy1':
        min_iters = 500
    else:
        min_iters = 1000
    nmode,tmode = dataset['nmode'], dataset['tmode']
    n_iters = dataset['iters']
    llr = dataset['lr']
    head ='Iters\tTrnRMSE\tTrnNorm\tTrMAE\tValRMSE\tValNorm\tValMAE\n'
    #print('Iters\tTrnRMSE\tValRMSE\n')
    
    with open(path_loss, 'w') as f:
        f.write(head)
    
    opt = optim.Adam(model.parameters(), lr = llr)
    
    start_time = time.time()
    
    old_rmse, stop_iter = 1e+5, 1
    
    for n_iter in range(1, n_iters + 1):
        
        loss = als_train(model, opt, tr, penalty, tmode, nmode)
        
        trn_rmse, trn_norm , trn_mae = evaluate(model, tr) 
        val_rmse, val_norm, val_mae = evaluate(model, val)
        
        if isNaN(trn_rmse):
            break
        if val_rmse > old_rmse:
            stop_iter = n_iter
            te_rmse, te_norm, te_mae = evaluate(model, test)
            if n_iter > min_iters:
                num +=1
            old_rmse = val_rmse 
        
#        if (n_iter % 1000) == 0:
#            print(f"{n_iter:4d}\t{trn_rmse:.4f}\t{val_rmse:.4f}\t")

        with open(path_loss, 'a') as f:
            f.write(f'{n_iter:4d}\t')
            f.write(f'{trn_rmse:.5f}\t{trn_norm:.5f}\t{trn_mae:.5f}\t')
            f.write(f'{val_rmse:.5f}\t{val_norm:.5f}\t{val_mae:.5f}\t')
            f.write('\n')
        if n_iter == stop_iter and n_iter > min_iters and num == 3:
            with open(total_loss_path, 'a') as f1:
                f1.write(f'{count:2d}\t{model.smooth}\t{model.sparse}\t')
                f1.write(f'{model.factors[0].shape[1]:2d}\t')
                f1.write(f'{window:2d}\t{penalty:.3f}\t{n_iter:4d}\t')
                f1.write(f'{te_rmse:.5f}\t{te_norm:.5f}\t{te_mae:.5f}\n')
                
            p = f'{path_model}-{n_iter}.pth.tar'
            save_checkpoints(model, p)
            if num == 3:
                break
    if n_iter == n_iters:
        
        te_rmse, te_norm, te_mae = evaluate(model, test)
        with open(total_loss_path, 'a') as f1:
            f1.write(f'{count:2d}\t{model.stype}\t{model.sparse}\t')
            f1.write(f'{model.factors[0].shape[1]:2d}\t')
            f1.write(f'{window:2d}\t{penalty:.3f}\t{n_iter:4d}\t')
            f1.write(f'{te_rmse:.5f}\t{te_norm:.5f}\t{te_mae:.5f}\n')

        p = f'{path_model}-{n_iter}.pth.tar'
        save_checkpoints(model, p)
    end_time = time.time()
    total_time = end_time - start_time
    print('Took %d mins %d secs' % (divmod(total_time, 60)))                   
