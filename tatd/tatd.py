

import numpy as np
import torch
import torch.nn as nn
import tensorly as tl

from torch.nn import functional as F
tl.set_backend('pytorch')
from tensorly.random import check_random_state
random_state = 1234
rng = check_random_state(random_state)

def gen_random(row, col):
    random_value = torch.FloatTensor(rng.random_sample((row, col)))
    return random_value

def krprod(factors, indices_list):

    ''' Khatri-rao product for given nonzeros' indicies '''

    rank = tl.shape(factors[0])[1]
    nnz = len(indices_list[0])
    nonzeros = tl.ones((nnz, rank), **tl.context(factors[0]))
    
    for indices, factor in zip(indices_list, factors):
        nonzeros = nonzeros * factor[indices, :]

    return torch.sum(nonzeros, dim = 1)

class Kernel(nn.Module):
    ''' Kernel smoothing '''
    
    def __init__(self, window, device, sigma = 0.5):
        super().__init__()
        self.sigma = sigma
        self.window = window
        self.weight = self.gaussian().to(device)
        
    def gaussian(self):
        window = int(self.window-1)/2
        sigma2 = self.sigma * self.sigma
        x = torch.FloatTensor(np.arange(-window, window+1))
        phi_x = torch.exp(-0.5 * abs(x) / sigma2)
        phi_x = phi_x / phi_x.sum()
        return phi_x.view(1, 1, self.window, 1)
    
    def forward(self,factor):
        
        row, col = factor.shape
        conv = F.conv2d(factor.view(1, 1, row, col), self.weight, 
                          padding = (int((self.window-1)/2), 0))
        return conv.view(row, col)
        
class Tatd(nn.Module):
   
    ''' Time-Aware Tensor Decomposition '''

    def __init__(self, nmode, ndim, tmode, density, rank, window, sparse, device):
       
        super().__init__()

        lst = [nn.Parameter(torch.Tensor(rng.random_sample((ndim[mode], rank))))
                for mode in range(nmode)]
       
        self.factors = nn.ParameterList(lst)
        self.smooth  = Kernel(window, device).to(device)
        self.rank = rank
        self.sparse = sparse
        self.density = density
        self.reset_parameters()
#        self.alpha = nn.Parameter(torch.Tensor(rng.random_sample(1)))
#        self.beta = nn.Parameter(torch.Tensor(rng.random_sample(1)))
       
    def reset_parameters(self):
        for f in self.factors:
            nn.init.xavier_normal_(f)
            f.requires_grad = False
           
    def forward(self, indices_list):
        return krprod(self.factors, indices_list)

    def turn_off_grad(self, mode):
        self.factors[mode].requires_grad = False
       
    def turn_on_grad(self, mode):
        self.factors[mode].requires_grad = True

    def l2_reg(self, mode):
        return torch.norm(self.factors[mode]).pow(2)

    def smooth_reg(self, tmode):
        ''' Smoothing regularization on the time factor '''
        smoothed = self.smooth(self.factors[tmode])
         
        smoothing_loss = torch.norm(self.factors[tmode] - smoothed, dim = 1)

        if self.sparse:
            sloss = smoothing_loss.pow(2) * self.density
            return sloss.sum()
        else:
            return smoothing_loss.pow(2).sum()

