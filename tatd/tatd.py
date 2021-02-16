

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import tensorly as tl
from tensorly.random import check_random_state
from tatd.attn_smooth import *
tl.set_backend('pytorch')
random_state = 1234
rng = check_random_state(random_state)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen_random(size):
    #if len(size) >1, size must be tuple
    random_value = torch.FloatTensor(rng.random_sample(size))
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
    
    def __init__(self, window, density, sigma = 0.5):
        super().__init__()
        self.sigma = sigma
        self.window = window
        self.density = density
        self.weight = self.gaussian().to(DEVICE)
        
    def gaussian(self):
        window = int(self.window-1)/2
        sigma2 = self.sigma * self.sigma
        x = torch.FloatTensor(np.arange(-window, window+1))
        phi_x = torch.exp(-0.5 * abs(x) / sigma2)
        phi_x = phi_x / phi_x.sum()
        return phi_x.view(1, 1, self.window, 1)
#        return phi_x

    
    def forward(self,factor):
        
        row, col = factor.shape
        conv = F.conv2d(factor.view(1, 1, row, col), self.weight, 
                          padding = (int((self.window-1)/2), 0))
        return conv.view(row, col)

    
class Tatd(nn.Module):
   
    ''' Time-Aware Tensor Decomposition '''

    def __init__(self, nmode, ndim, tmode, density, rank, window, weightf, sparse, exp):
       
        super().__init__()

        self.nmode =nmode
        self.tmode =tmode
        self.rank = rank
        self.sparse = sparse
        self.density = density
        self.exp = exp
        self.weightf = weightf
        
        lst = [nn.Parameter(gen_random((ndim[mode], rank)))
                for mode in range(nmode)]
        self.factors = nn.ParameterList(lst)
        
        if weightf == 'kernel':   
            self.smooth = Kernel(window, self.density).to(DEVICE)
        else:
            raise NotImplementedError
      
        self.reset_parameters()
       
    def reset_parameters(self):
        for mode in range(self.nmode):
            f = self.factors[mode]
            nn.init.xavier_normal_(f)
            if mode != self.tmode:
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
        sloss = (smoothed - self.factors[tmode]).pow(2)
        if self.sparse == 1:
           sloss = sloss * self.density.view(-1, 1) 
        return sloss.sum()

