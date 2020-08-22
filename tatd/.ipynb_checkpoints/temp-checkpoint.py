import os
import torch
import tensorly as tl
import numpy as np
import pandas as pd
import torch.funtional as F
import torch.nn as nn

def make_sparsity(data, tmode, device):
    npy = data.indices().cpu().numpy()
    columns = list(np.arange(npy.shape[0]))
    columns[tmode] = 'time'
    df = pd.DataFrame(npy.T, columns = columns)
    df = df.groupby('time').count().iloc[:, 0]   
    print(df)
    max_ = df.max()
    return 1- torch.FloatTensor(list(df/max_)).to(device)
   
   
def read_dataset(path = '../data', name = 'air10', device = 'cpu', **kargs):
    
    tensors = read_tensors(path, name, device)


    if name == 'uber':
        tmode = 0
        lr = 0.001
        iters = 3000
        rrank = 5
        window = 9
        sparsity = make_sparsity(tensors['train'], tmode, device)
        
    tensors['tmode'] = tmode
    tensors['lr'] = lr
    tensors['iters'] = iters
    tensors['rrank'] = rrank
    tensors['window'] = window
    tensors['sparse'] = sparsity
    
    return tensors 

class Attention_dot(nn.Module):
    
    def __init__(self, window, rrank ):

        super().__init__()
        
        self.window = window
        self.count = 0
        self.rank = rrank
    def reset_parameters(self):

        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
    
    def make_index(self, length):
        self.length = length
        return np.arange(0, self.window)[None, :] + np.arange(length)[:, None]
    
    def forward(self, time_factor):       
        
        smoothed = self.smooth(time_factor)
        
        return smoothed.view(time_factor)
        
    
    def smooth(self, data):
        
        row, col = data.shape

        if self.count == 0:
            self.indexer = self.make_index(row)
        self.count +=1        
        
        padding = int((self.window-1)/2)

        pad_data = F.pad(data.t(), (padding,padding))
        pad_data = pad_data.t()
        
        blocks = pad_data[self.indexer, :]
        cos = torch.bmm(blocks, data.view(row, col, 1))
        cos = F.softmax(cos, dim = 1)
        result = blocks * cos
        result = result.sum(1)
        return result
class Attention(nn.Module):
    
    def __init__(self, window, rrank ):

        super().__init__()
        
        self.window = window
        self.count = 0
        self.rank = rrank
        self.kernel = nn.Parameter(torch.Tensor(1, rrank, rrank))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.kernel)

    
    def make_index(self, length):
        self.length = length
        return np.arange(0, self.window)[None, :] + np.arange(length)[:, None]
    
    def forward(self, time_factor):       
        
        smoothed = self.smooth(time_factor)

        return smoothed
    
    def smooth(self, data):
        
        row, col = data.shape

        if self.count == 0:
            self.indexer = self.make_index(row)
        self.count +=1        
        
        padding = int((self.window-1)/2)

        pad_data = F.pad(data.t(), (padding,padding))
        pad_data = pad_data.t()
        
        blocks = pad_data[self.indexer, :]
        
        repeat_weight = self.kernel.repeat(row, 1,1)
        cos = torch.bmm(blocks, repeat_weight)
        cos = torch.bmm(cos, data.view(row, col, 1))
        cos = cos / self.window
        result = blocks * cos
        result = result.sum(1)
        return result
  
def get_model(device, nmode, ndim, rrank, window, tmode, smooth, sparse):
    
    if smooth == 'standard':
        smodel = None
    elif smooth == 'kernel':
        smodel = Kernel(window, device).to(device)
    elif smooth == '1d':
        smodel = Smooth1d(window).to(device)
    elif smooth == '2d':
        smodel = Smooth2d(rrank, window).to(device)
    elif smooth == 'attention':
        smodel = Attention(window, rrank).to(device)
    elif smooth == 'dot':
        smodel = Attention_dot(window, rrank).to(device)
    else:
        print("Input the model type..!")
    
    model = TdtdS(nmode, ndim, rrank, window, tmode, 
                 smooth = smodel , sparse = sparse)
    
    return model.to(device)
   
class TdtdS(nn.Module):
    def __init__(self, nmode, ndim, rrank, window, tmode, smooth, sparse):
        super().__init__()
        
        self.rank = rrank
        
        self.factors = nn.ParameterList([nn.Parameter(torch.Tensor(ndim[mode], rrank))
                                         for mode in range(nmode)])
        
#         self.alpha = nn.Parameter(torch.Tensor(1))
#         self.bias = nn.Parameter(torch.Tensor(1))
        
        if sparse is not None:
            self.sparse = sparse
        else:
            self.register_parameter('sparse', None)
            
        if smooth is not None:
            self.smooth = smooth
        else:
            self.register_parameter('smooth', None)
            
        self.reset_parameters()
        

    def reset_parameters(self):
        for f in self.factors:
            nn.init.xavier_normal_(f)
        if self.sparse is not None:
            nn.init.normal_(self.alpha)
            nn.init.normal_(self.bias)            
        
            
    def forward(self, indices_list):
        return krprod(self.factors, indices_list)
    
    def turn_off_grad(self, mode):
        self.factors[mode].requires_grad = False
        
    def turn_on_grad(self, mode):
        self.factors[mode].requires_grad = True
    
    def smooth_reg(self, tmode, sparsity = None):

        if  self.smooth is None:
            return 0
        else:
            loss = (self.factors[tmode] - self.smooth(self.factors[tmode])).pow(2)
            loss = loss.sum(1)
#             loss = torch.sqrt(loss) * (self.sparse * self.alpha + self.bias)
            loss = torch.sqrt(loss)
            loss = loss.sum()
            return loss
        
    def extra_repr(self):
        return 'Rank={}, Smooth={}, Sparse={}'.format(
            self.rank, self.smooth, self.sparse is not None
        )