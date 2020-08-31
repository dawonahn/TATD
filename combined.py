

from tdtd.utils import *
from tdtd.read import *
from tdtd.train import *
from tdtd.model.cp import *


import tensorly as tl
tl.set_backend('pytorch')

import sys

name = sys.argv[1]
penalty = float(sys.argv[2])
smodel = sys.argv[3]

def eval_(model, data):
    with torch.no_grad():
        rec = krprod(model.factors, data.indices())
        vals = data.values()
    return rmse(vals, rec).cpu().item()

def eval2_(model, data):
    with torch.no_grad():
        rec = krprod(model.factors, data.indices())
        vals = data.values()
    return mae(vals, rec).cpu().item()

def main():

    device = 'cuda:3'
    ssmooth = smodel 
    ssparse = None
    window = 7
    rank = 10

    dataset = read_dataset(name = name, device = device)
    nmode, ndim, tmode =  dataset['nmode'], dataset['ndim'], dataset['tmode']
    density = dataset['inverse']
    

    indices = dataset['train'].indices().cpu().numpy()
    vals = dataset['train'].values().cpu().numpy()
    temp = np.vstack([indices, vals]).T
    
    trn = dataset['train']
    valid = dataset['valid']
    test = dataset['test']
    
    dct = {}
    for mode, index in enumerate(indices):
        dct[mode] = {}
        unique, counts = np.unique(index, return_counts=True)
        for a, b in zip(* [unique, counts]):
            dct[mode][a] = b
            
    model = get_model(device, nmode, ndim , tmode, density, 
                  rank, window, ssmooth,ssparse)
        
    old_rmse = 1e+10
    old_rmse2 = 1e+10

    start_time1 = time.time()
    
    result_lst = []

    opt = optim.Adam(model.parameters(), lr = 1e-2)
    for n_iter in range(10000):

        start_time = time.time()

        for mode in range(nmode):   

            if mode != tmode:
                nnz_count = dct[mode]

                for idx, nnz in nnz_count.items() :

                    nnz_entries = temp[temp[:, mode] == idx]
                    nnz_idxs = nnz_entries[:,:-1]
                    nnz_vals = nnz_entries[:,-1]

                    ones =  torch.ones((nnz, rank)).to(device)

                    for i, item in enumerate(zip(nnz_idxs.T, model.factors)):
                        if i != mode:
                            index, factor = item
                            ones = ones * factor[torch.LongTensor(index).to(device), :]

                    mat_b = torch.bmm(ones.view(nnz, rank, 1), ones.view(nnz, 1, rank))
                    mat_b = torch.sum(mat_b, dim = 0)
                    mat_b2 = mat_b +  torch.eye(rank).to(device) * penalty

                    vect_c = ones * torch.FloatTensor(nnz_vals).view(-1, 1).to(device)
                    vect_c = torch.sum(vect_c, dim = 0)
                    
                    update =  torch.matmul(torch.inverse(mat_b2), vect_c )
                    update = torch.where(torch.abs(update) < 0.000001, torch.zeros_like(update), update) 
                    model.factors[mode][idx].values = update.values

            else:

                stop = True
                while(stop):
                    opt.zero_grad()

                    model.turn_on_grad(mode)

                    rec = model(trn.indices())

                    loss = (rec - trn.values()).pow(2).sum()

                    loss = loss + penalty * model.smooth_reg(mode)

                    loss.backward()

                    opt.step()

                    val_rmse = eval_(model, valid) 

                    if val_rmse > old_rmse2:
                        stop = False
                    old_rmse2 = val_rmse
                    mins, secs = divmod(time.time() - start_time, 60) 
                    print(f'[Iters : {n_iter}] Took {int(mins)}  mins {int(secs)} secs\t'
                    f"Train RMSE : {eval_(model, trn):.4f} Valid RMSE : {val_rmse:.4f}")

        val_rmse = eval_(model, valid) 
        if n_iter % 100 == 0:

            mins, secs = divmod(time.time() - start_time, 60) 
            print(f'[Iters : {n_iter}] Took {int(mins)}  mins {int(secs)} secs\t'
                f"Train RMSE : {eval_(model, trn):.4f} Valid RMSE : {val_rmse:.4f}")
        
        result_lst.append([n_iter, eval_(model, trn), eval_(model, valid)])
        if val_rmse > old_rmse :
            break
        old_rmse = val_rmse 
    end_time = time.time()
    total_time = end_time - start_time1
    print('Took %d mins %d secs' % (divmod(total_time, 60))) 
    print(f"Test RMSE : {eval_(model, test):.4f}")
    te_rmse = eval_(model, test)
    te_mae = eval2_(model, test)
    pd.DataFrame(result_lst).to_csv(f'./combined/{name}_{penalty}_{n_iter}.txt', sep = '\t',  header = False, index =False)
    with open('./combined/als_result.txt',  'a') as f:
        f.write(f'{name:10s}\t{n_iter:3d}\t{penalty:.5f}\t')
        f.write(f'{te_rmse:.5f}\t{te_mae:.5f}\n')
        f.write('\n')
    save_checkpoints(model, f'./combined/{name}_{penalty}_{n_iter}.pth.tar')
    

    
    
if __name__ == '__main__':
    main()
