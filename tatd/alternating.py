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

def alternating_training(model, opt, train, penalty, tmode, mode):

    opt.zero_grad()

    rec = model(train.indices())

    loss = (rec - train.values()).pow(2).sum()

    if mode == tmode:
        loss = loss + penalty * model.smooth_reg(mode)
    else:
        loss = loss + penalty * model.l2_reg(mode)

    loss.backward()

    opt.step()

    return loss


def alternating_train_model(dataset, model, penalty, opt_scheme, lr, loss_path, model_path, total_path):

    train, valid, test = dataset['train'], dataset['valid'], dataset['test']
    nmode, tmode = dataset['nmode'], dataset['tmode']
    window, count = dataset['window'], dataset['count']

    head = 'Iters\tTime\tTrnRMSE\tTrMAE\tValRMSE\tValMAE\n'

    with open(loss_path, 'w') as f:
        f.write(head)

    opt_dct = {}
    if opt_scheme == 'alternating_adam':
        for mode in range(nmode):
            opt_dct[mode] = optim.Adam([list(model.parameters())[mode]], lr = lr)
    else:
        for mode in range(nmode):
            opt_dct[mode] = optim.SGD([list(model.parameters())[mode]], lr=lr)
    start_time = time.time()
    old_rmse, inner_rmse, stop_iter, = 1e+5, 1e+5, 0

    for n_iter in trange(1, 35000):
        for mode in range(nmode):
            model.turn_on_grad(mode)
            inner_num = 0
            inner_num1 = 0
            stop = True
            while(stop):
                loss = alternating_training(model, opt_dct[mode], train, penalty, tmode, mode)
                inner_num1 += 1
                trn_rmse, trn_mae = evaluate(model, train)
                val_rmse, val_mae = evaluate(model, valid)
                if opt_scheme == 'alternating_sgd':
                    print('Iter', n_iter, 'Mode', mode, 'Train', trn_rmse, 'Valid', val_rmse)
                # if np.abs(val_rmse - inner_rmse) < 0.000001:
                #     stop = False
                if inner_num > 3:
                    stop = False
                if val_rmse >= inner_rmse:
                    inner_num += 1
                inner_rmse = val_rmse
                if isNaN(trn_rmse):
                    print("Nan break")
                    break
            if isNaN(trn_rmse):
                return
        if val_rmse > old_rmse:
            stop_iter += 1
        old_rmse = val_rmse

        with open(loss_path, 'a') as f:
            elapsed = time.time() - start_time
            f.write(f'{n_iter:5d}\t{elapsed:.5f}\t')
            f.write(f'{trn_rmse:.5f}\t{trn_mae:.5f}\t')
            f.write(f'{val_rmse:.5f}\t{val_mae:.5f}\n')
        if stop_iter == 5 or n_iter > 30000:
            te_rmse, te_mae = evaluate(model, test)
            with open(total_path, 'a') as f1:
                f1.write(f'{count:5d}\t{n_iter:5d}\t{elapsed}\t{model.sparse}\t')
                f1.write(f'{model.factors[0].shape[1]:2d}\t')
                f1.write(f'{window:2d}\t{penalty:.3f}\t')
                f1.write(f'{opt_scheme}\t{lr:5f}\t')
                f1.write(f'{te_rmse:.5f}\t{te_mae:.5f}\n')

            p = f'{model_path}-{n_iter}.pth.tar'
            save_checkpoints(model, p)
            break
