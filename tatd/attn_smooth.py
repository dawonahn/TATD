
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attn_w(nn.Module):
    
    def __init__(self, window, rrank, density, exp, 
                 layer):

        super().__init__()
        self.window = window
        self.indexer = self.make_index(density.shape[0])
        self.rank = rrank
        self.density = density
        self.exp = exp
        self.layer = layer
      
    def make_index(self, length):
        self.length = length
        return np.arange(0, self.window)[None, :] + np.arange(length)[:, None]
    
    def forward(self, time_factor):       
        smoothed = self.smooth(time_factor)
        smoothed = smoothed.view(time_factor.shape)
        return smoothed
        
    def smooth(self, data,):
        
        row, col = data.shape

        padding = int((self.window-1)/2)
        pad_data = F.pad(data.t(), (padding,padding))
        pad_data = pad_data.t()
        blocks = pad_data[self.indexer, :]
        
        var1 = self.layer(data)
        
        if len(self.exp.split('_')) >= 2:
            if self.exp.split('_')[1] == 'tanh':
                var1 = torch.tanh(var1)
        var2 = torch.bmm(blocks, var1.view(row, -1, 1))
        if self.exp.endswith('scaled'):
            var2 = var2/torch.sqrt(torch.ones(1)*col).to(DEVICE) 
        if self.exp.startswith('softmax'):
            var2 = torch.softmax(var2, dim = 1)
        var3 = torch.bmm(var2.view(row, 1, -1), blocks)

            
        return var3

