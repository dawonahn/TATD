
from tatd.utils import *
from tatd.read import *
from tatd.train import *
from tatd.tatd import *
from tatd.alternating import *
from tatd.als import *
# from tatd.als_opt import *

import sys
import argparse
torch.backends.cudnn.deterministic = True
torch.manual_seed(1234)

def tatd_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='google_cluster1', help=u"Dataset")
    parser.add_argument('--sparse', type=int, default=0, help=u"Sparsity penalty")
    parser.add_argument('--rank', type=int, default=10, help=u"Size of factor matrix")
    parser.add_argument('--window', type=int, default=7, help=u"Window size")
    parser.add_argument('--penalty', type=float, default=1, help=u"Strength of penalty")
    parser.add_argument('--opt_scheme', type=str, default='adam', help=u"Alternating Adam")
    parser.add_argument('--lr', type=float, default='0.001', help=u"Learning rate")
    parser.add_argument('--gpu', type=int, default=2, help=u"GPU device")
    parser.add_argument('--count', type=int, default=10, help=u"Experiment for average")
    args = parser.parse_args()

    name = args.name
    ###
    sparse = args.sparse
    rank = args.rank
    window = args.window
    penalty = args.penalty
    ###
    opt_scheme = args.opt_scheme
    lr = args.lr
    ###
    gpu, count = args.gpu
    count = args.count

    return name, sparse, rank, window, penalty, opt_scheme, lr, gpu, count

def main():

    name, sparse, rank, window, penalty, opt_scheme, lr, gpu, count = tatd_parser()

    loss_path, model_path, total_path = get_path(name, sparse, 
                                                rank, window, penalty,
                                                opt_scheme, lr, count)
    device = f'cuda:{gpu}'
            
    dataset = read_dataset(name, device)
    dataset['count'], dataset['window'] = count, window
    nmode, ndim, tmode =  dataset['nmode'], dataset['ndim'], dataset['tmode']
    
    density = dataset['ts_beta']

    model = Tatd(nmode, ndim, tmode, density, rank, window, sparse, device).to(device)

    print(f"START {name} sparse : {bool(sparse)} & {opt_scheme} : {lr} "
          f"Rank : {rank} Window : {window} Penalty : {penalty} \n")

    if opt_scheme.startswith('alternating'):
        alternating_train_model(dataset, model, penalty, opt_scheme, lr,
                    loss_path, model_path, total_path)
    elif opt_scheme.startswith('als'):
        als_train_model(dataset, model, penalty, opt_scheme, lr, rank,  device,
                    loss_path, model_path, total_path)
    else:
        train_model(dataset, model, penalty, opt_scheme, lr, 
                    loss_path, model_path, total_path)

if __name__ == '__main__':
    main()
        
        
