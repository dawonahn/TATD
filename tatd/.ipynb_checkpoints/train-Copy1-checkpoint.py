
import time
import torch
import torch.functional as F
import torch.optim as optim
import torch.nn as nn

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

def als_train(model, opt, train, penalty, tmode, nmode, reg):
    'als update'
    for mode in range(nmode):

        opt.zero_grad()

        model.turn_on_grad(mode)

        rec = model(train.indices())

        loss = (rec - train.values()).pow(2).sum()
        
        if reg == True:
        
            if mode == tmode:
                loss = loss + penalty * model.smooth_reg(mode)
            else:
                loss = loss + 0.001 * torch.norm(model.factors[mode])

        loss.backward()

        opt.step()

        model.turn_off_grad(mode)

    return loss
        
def sgd_train(model, opt, train, penalty, tmode, nmode, reg):
    'sgd update'
    
    opt.zero_grad()

    rec = model(train.indices())

    loss = (rec - train.values()).pow(2).sum()

    if reg == True:
        for mode in range(nmode):
            if mode == tmode:
                loss = loss + 0.001 * model.smooth_reg(mode)
            else:
                loss = loss + 0.001 * torch.norm(model.factors[mode])

    loss.backward()

    opt.step()

    return loss
             
def train_model(model, dataset, penalty, ttype, path_loss,path_loss1, path_model,reg, count):
    
    if reg == False:
        reg1 = 'x'
    else:
        reg1 = 'o'
    tr = dataset['train']
    val = dataset['valid']
    test = dataset['test']

    nmode,tmode = dataset['nmode'], dataset['tmode']
    n_iters = dataset['iters']
    
    if ttype == 'sgd':
        train = sgd_train
    else:
        train = als_train
    
    head ='Iters\tTrnRMSE\tTrnNorm\tTrMAE\tValRMSE\tValNorm\tValMAE\n'
    print(head)
    
    with open(path_loss, 'w') as f:
        f.write(head)
    
    opt = optim.Adam(model.parameters(), lr = 0.001)
    
    start_time = time.time()
    
    old_rmse, stop_iter = 1e+5, 1
    
    for n_iter in range(1, n_iters +1):
        
        loss = train(model, opt, tr, penalty, tmode, nmode, reg)
        
        trn_rmse, trn_norm , trn_mae = evaluate(model, tr) 
        val_rmse, val_norm, val_mae = evaluate(model, val)
        
        
        if val_rmse > old_rmse:
            stop_iter = n_iter
            te_rmse, te_norm, te_mae = evaluate(model, test)
        old_rmse = val_rmse 
        
        if (n_iter % 1000) == 0:
            print(f"{n_iter:4d}\t{trn_rmse:.4f}\t{val_rmse:.4f}\t")
                  
        with open(path_loss, 'a') as f:
            f.write(f'{n_iter:4d}\t')
            f.write(f'{trn_rmse:.5f}\t{trn_norm:.5f}\t{trn_mae:.5f}\t')
            f.write(f'{val_rmse:.5f}\t{val_norm:.5f}\t{val_mae:.5f}\t')
            if n_iter == stop_iter:
                f.write('\tSTOP')
            f.write('\n')
        if n_iter == stop_iter and n_iter > 1000 :
            
            with open(path_loss1, 'a') as f1:
                f1.write(f'count\t{count:2d}\t{ttype}\trrank\t{model.factors[0].shape[1]}\t')
                f1.write(f'reg\t{reg1}\t{n_iter:4d}\t')
                f1.write(f'{te_rmse:.5f}\t{te_norm:.5f}\t{te_mae:.5f}\t')
    
            p = f'{path_model}-{n_iter}.pth.tar'
            save_checkpoints(model, p)
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    print('Took %d mins %d secs' % (divmod(total_time, 60)))                   
