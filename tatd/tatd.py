

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
        
class Attn(nn.Module):
    
    def __init__(self, window, rrank, density, exp, layer1, layer2):
    #def __init__(self, window, rrank, density, exp):


        super().__init__()
        
        self.window = window
        self.indexer = self.make_index(density.shape[0])
        self.rank = rrank
        self.density = density
        self.exp = exp
        self.layer1 = layer1 
        self.layer2 = layer2
      
    def make_index(self, length):
        self.length = length
        return np.arange(0, self.window)[None, :] + np.arange(length)[:, None]
    
    def forward(self, time_factor):       
    #def forward(self, time_factor):       
        
        #smoothed = self.smooth(time_factor, mlp)
        smoothed = self.smooth(time_factor)
        
        smoothed = smoothed.view(time_factor.shape)
        
        return smoothed
        
    def smooth(self, data,):
   # def smooth(self, data, mlp2):
        
        row, col = data.shape
        if self.layer2 is not None:
            ldensity = torch.sigmoid(self.layer2(self.density))

        padding = int((self.window-1)/2)
        pad_data = F.pad(data.t(), (padding,padding))
        pad_data = pad_data.t()
        blocks = pad_data[self.indexer, :]
#        with torch.no_grad():
        cos = torch.bmm(blocks, data.view(row, col, 1))
        #cos = cos / torch.sqrt(torch.FloatTensor([self.rank])).to(DEVICE)
        cos = F.softmax(cos, dim = 1)
        result = (blocks *cos).sum(1)
        if self.layer1 is not None:
            cfactor = torch.cat((data, result), dim = 1)
            result = self.layer1(cfactor)
            #result = torch.sigmoid(result)

#        result = (1 - ldensity) * data + ldensity * result
#        result = blocks * cos * ldensity.view(row, -1, 1)

#        if self.exp == 'n_tanh':
#            result = torch.tanh(result)
#        if self.exp == 'n_sig':
#            result = torch.sigmoid(result)
        return result 

#class Kernel(nn.Module):
#    
#    def __init__(self, window, rrank, density, exp, sigma = 0.5):
#
#        super().__init__()
#        self.sigma = sigma
#        self.window = window
#        self.rank = rrank
#        self.density = density
#        self.indexer = self.make_index(density.shape[0])
#        self.weight = self.gaussian().to(DEVICE)
#        self.exp = exp
#        
#    def gaussian(self):
#        window = int(self.window-1)/2
#        sigma2 = self.sigma * self.sigma
#        x = torch.FloatTensor(np.arange(-window, window+1))
#        phi_x = torch.exp(-0.5 * abs(x) / sigma2)
#        phi_x = phi_x / phi_x.sum()
#        return phi_x
#
#    def make_index(self, length):
#        self.length = length
#        return np.arange(0, self.window)[None, :] + np.arange(length)[:, None]
#    
#    def forward(self, time_factor):       
#        
#        smoothed = self.smooth(time_factor)
#        
#        smoothed = smoothed.view(time_factor.shape)
#        
#        return smoothed
#        
#    def smooth(self, data):
#        
#        row, col = data.shape
#
#        ldensity = mlp(self.density)
#        padding = int((self.window-1)/2)
#        pad_data = F.pad(data.t(), (padding,padding))
#        pad_data = pad_data.t()
#        blocks = pad_data[self.indexer, :]
#        with torch.no_grad():
#            cos = ldensity * self.weight
#            cos = torch.sigmoid(cos).view(row, -1, 1)
#
#        result = blocks * cos
#        if self.exp == 'n_tanh':
#            result = torch.tanh(result)
#        if self.exp == 'n_sig':
#            result = torch.sigmoid(result)
#        result = result.sum(1)
#        return result
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
        
        if exp.startswith('cat'):
            self.layer1 = nn.Linear(rank * 2, rank)
        else:
            self.layer1 = None
        if sparse < 2:
            self.layer2 = None
        if sparse == 2:
            self.layer2 = nn.Linear(1, 1)
            self.density = density.view(-1, 1)
        if sparse == 3:
            self.layer2 = nn.Linear(window, 1)
            self.density = self.local_density(density, window) 
        if sparse == 4:
            self.layer2 = nn.Linear(window, 7)
            self.density = self.local_density(density, window) 

        #self.mlp2 = nn.Linear(rank, rank)
        if weightf == 'attn':
            self.smooth  = Attn(window, rank, self.density, exp, 
                    self.layer1, self.layer2).to(DEVICE)
            #self.smooth  = Attn(window, rank, self.density, exp, ).to(DEVICE)
            
        elif weightf == 'attn_w':
            self.layer = nn.Linear(rank, rank)
            self.smooth = Attn_w(window, rank, self.density, exp, self.layer)
        elif weightf == 'attn_w_s':
            self.layer = nn.Parameter(gen_random((rank)))
            self.smooth = Attn_w_s(window, rank, self.density, exp, self.layer)
        else:
            self.smooth = Kernel(window, self.density).to(DEVICE)
            #self.smooth = Kernel(window, rank, self.density, exp).to(DEVICE)

        
        self.reset_parameters()
       
    def reset_parameters(self):
        for mode in range(self.nmode):
            f = self.factors[mode]
            nn.init.xavier_normal_(f)
            if mode != self.tmode:
                f.requires_grad = False
            #nn.init.xavier_normal_(self.layer.weight)
            #nn.init.xavier_normal_(self.layer.bias)

    def local_density(self, density, window):
        length = len(density)
        index = np.arange(0, window)[None, :] + np.arange(length)[:, None]
        padding = int((window -1)/2)
        density = F.pad(density, (padding, padding))
        density = density[index]
        return density

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
#        if self.weightf == 'attn_w':
 #           wdecay = torch.norm(self.layer.weight).pow(2) + torch.norm(self.layer.bias).pow(2)
        if self.sparse == 1:
           sloss = sloss * self.density.view(-1, 1) 
        if self.sparse == 1.5:
           sloss = sloss * (1 / self.density.view(-1, 1) )
        if self.sparse == 1.6:
           sloss = sloss * (1 / (1 - self.density.view(-1, 1) ))
        return sloss.sum()

