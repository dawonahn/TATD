import time
import torch
import torch.functional as F
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from tqdm import trange
from tatd.tatd import *
from tatd.utils import *
from tatd.train import *

def gradient(model, opt, train, penalty, tmode):

    opt.zero_grad()
    rec = model(train.indices())
    loss = (rec - train.values()).pow(2).sum()
    loss = loss + penalty * model.smooth_reg(tmode)
#     print(f"Loss :{model.smooth_reg(tmode).data}, Penalty :{penalty}")
    loss.backward()
    opt.step()

    return loss

def metadata(train):
    indices = train.indices().cpu().numpy()
    vals = train.values().cpu().numpy()
    temp = np.vstack([indices, vals]).T

    dct = {}
    for mode, index in enumerate(indices):
        dct[mode] = {}
        unique, counts = np.unique(index, return_counts=True)
        for a, b in zip(* [unique, counts]):
            dct[mode][a] = b

    return dct, temp

def least_square(model, train, penalty, mode, dct, temp, rank, device):

    nnz_count = dct[mode]
    for idx, nnz in nnz_count.items():
        with torch.no_grad():
            nnz_entries = temp[temp[:, mode] == idx]
            nnz_idxs = nnz_entries[:, :-1]
            nnz_vals = nnz_entries[:, -1]

            ones = torch.ones((nnz, rank)).to(device)
            for i, item in enumerate(zip(nnz_idxs.T, model.factors.cpu())):
                if i != mode:
                    index, factor = item
                    ones = ones * factor[torch.LongTensor(index), :]

            mat_b = torch.bmm(ones.view(nnz, rank, 1), ones.view(nnz, 1, rank))
            mat_b = torch.sum(mat_b, dim=0)
            mat_b2 = mat_b + torch.eye(rank)* penalty

            vect_c = ones * torch.FloatTensor(nnz_vals).view(-1, 1)
            vect_c = torch.sum(vect_c, dim=0)

            update = torch.matmul(torch.inverse(mat_b2), vect_c)
            update = torch.where(torch.abs(update) < 0.000001, torch.zeros_like(update), update)
            model.factors[mode][idx] = update.to(device)

def als_train_model(dataset, model, penalty, opt_scheme, lr, rank, device,
                    loss_path, model_path, total_path):

    train, valid, test = dataset['train'], dataset['valid'], dataset['test']
    nmode, tmode = dataset['nmode'], dataset['tmode']
    window, count = dataset['window'], dataset['count']

    head = 'Iters\tTime\tTrnRMSE\tTrMAE\tValRMSE\tValMAE\n'
    with open(loss_path, 'w') as f:
        f.write(head)

    dct, temp = metadata(train)
    if opt_scheme == 'als_adam':
        opt = optim.Adam([list(model.parameters())[tmode]], lr = lr)
    else:
        opt = optim.SGD([list(model.parameters())[tmode]], lr=lr)

    start_time = time.time()
    old_rmse, inner_rmse, stop_iter, = 1e+5, 1e+5, 0
    for n_iter in trange(1, 1000):
        inner_num = 0
        stop = True
        for mode in range(nmode):
            if mode != tmode:
                least_square(model, train, penalty, mode, dct, temp, rank, device)
                trn_rmse, trn_mae = evaluate(model, train)
                val_rmse, val_mae = evaluate(model, valid)
                print('Iter', n_iter, 'Mode', mode, 'Train', trn_rmse, 'Valid', val_rmse)
            else:
                model.turn_on_grad(mode)
                while(stop):
                    stop = gradient(model, opt, train, penalty, tmode)
                    trn_rmse, trn_mae = evaluate(model, train)
                    val_rmse, val_mae = evaluate(model, valid)
                    #print('Iter', n_iter, 'Mode', mode, 'Train', trn_rmse, 'Valid', val_rmse)
                    if inner_num > 10:
                        stop = False
                    if val_rmse > inner_rmse:
                        # break
                        inner_num += 1
                    inner_rmse = val_rmse
                    if isNaN(trn_rmse):
                        print("Nan break")
                        break
            trn_rmse, trn_mae = evaluate(model, train)
            val_rmse, val_mae = evaluate(model, valid)

        if val_rmse > old_rmse and n_iter >1:
            stop_iter += 1
        old_rmse = val_rmse

        with open(loss_path, 'a') as f:
            elapsed = time.time() - start_time
            f.write(f'{n_iter:5d}\t{elapsed:.5f}\t')
            f.write(f'{trn_rmse:.5f}\t{trn_mae:.5f}\t')
            f.write(f'{val_rmse:.5f}\t{val_mae:.5f}\n')
        if stop_iter == 1 or n_iter == 999:
            te_rmse, te_mae = evaluate(model, test)
            with open(total_path, 'a') as f1:
                f1.write(f'{count}\t{n_iter:5d}\t{elapsed:.3f}\t{model.sparse}\t')
                f1.write(f'{model.factors[0].shape[1]:2d}\t')
                f1.write(f'{window:2d}\t{penalty:.3f}\t')
                f1.write(f'{opt_scheme}\t{lr:5f}\t')
                f1.write(f'{te_rmse:.5f}\t{te_mae:.5f}\n')

            p = f'{model_path}-{n_iter}.pth.tar'
            save_checkpoints(model, p)
            break
