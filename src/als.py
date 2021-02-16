

from .train import *

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gradient(model, opt, train, penalty, tmode):

    opt.zero_grad()
    rec = model(train.indices())
    rloss = (rec - train.values()).pow(2).sum()
    sloss = penalty * model.smooth_reg(tmode)
    loss = rloss + sloss 
    loss.backward()
    opt.step()

    return rloss.detach(), sloss.detach()


def my_khatri_rao(matrices, indices_list, skip_matrix=None):
    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i!=skip_matrix]
        indices_list = indices_list[1:]  # If skip_matrix is time mode

    rank = tl.shape(matrices[0])[1]

    # Compute the Khatri-Rao product for the chosen indices
    sampled_kr = torch.ones((len(indices_list[0]), rank))
    for indices, matrix in zip(indices_list, matrices):
        sampled_kr = sampled_kr * matrix[indices, :]

    return sampled_kr, indices_list


def sparse_least_square(model, train, penalty, mode, rank):
    ''' Update factors considering only nonzeros'''
    num = model.factors[mode].shape[0]
    kr_prod, indices_list = my_khatri_rao(model.factors, train.indices(), skip_matrix=mode)

    indices_list = [i.tolist() for i in indices_list]
    indices_list.insert(mode, slice(None, None, None))
    unfolding = train.to_dense()[indices_list].float()

    inverse = tl.dot(tl.transpose(kr_prod), kr_prod) + torch.eye(rank) * penalty
    factor = tl.dot(tl.transpose(kr_prod), unfolding)

    factor = tl.transpose(tl.solve(inverse, factor))
    factor = torch.where(torch.abs(factor) < 0.000001, torch.zeros_like(factor), factor)
    factor = torch.nn.Parameter(factor)
    factor.require_grad = False

    model.factors[mode] = factor.to(DEVICE)


def als_train_model(dataset, model, rank, penalty, lr):

    name = dataset['name']

    train, valid, test = dataset['train'], dataset['valid'], dataset['test']
    nmode, tmode = dataset['nmode'], dataset['tmode']

    opt = optim.Adam(model.parameters(), lr = lr,)

    start_time = time.time()
    old_rmse, inner_rmse, stop_iter, = 1e+5, 1e+5, 0
    lst =[]
    c = 0
    with trange(10000) as t:
        for n_iter in t:
            inner_num = 0
            stop = True
            for mode in range(nmode):
                if mode != tmode:
                    sparse_least_square(model, train, penalty, mode, rank)
                else:
                    while(stop):
                        rloss, sloss = gradient(model, opt, train, penalty, tmode)
                        trn_rmse, fit = evaluate(model, train)
                        val_rmse, val_fit = evaluate(model, valid)
                        t.set_description(f'Fit: {fit:.5f} trn_rmse : {trn_rmse:.4f} val_rmse : {val_rmse:.4f} rec loss :{rloss:.4f} s loss : {sloss:.4f}')
                        lst.append([n_iter, rloss, sloss])
                        if inner_num > 5:
                            stop = False
                        if val_rmse > inner_rmse:
                            inner_num += 1
                        if val_rmse == inner_rmse:
                            break
                        inner_rmse = val_rmse
                        
                        if isNaN(trn_rmse):
                            print("Nan break")
                            break

            trn_rmse, fit = evaluate(model, train)
            val_rmse, _ = evaluate(model, valid)

            if val_rmse > old_rmse and n_iter > 10:
                stop_iter += 1
            old_rmse = val_rmse

            if stop_iter == 10:
                break
    return trn_rmse, val_rmse, fit, val_fit

