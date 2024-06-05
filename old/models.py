import numpy as np
import os, sys
from os import path
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm, trange

from collections import OrderedDict

EPS = 1e-2

### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#models
### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

#utils 
def sigma(t):
    return torch.sigmoid(t)

def build_relu(in_dim, hidden_dim, num_layers):
    v = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for n in range(num_layers):
        v.append(nn.Linear(hidden_dim, hidden_dim))
        v.append(nn.ReLU())
    v.append(nn.Linear(hidden_dim, 1))
    v = OrderedDict([(str(i), v[i]) for i in range(len(v))])
    v = nn.Sequential(v)
    return v

def bERexp(model, x):
    x1, x2 = x[:, :1], x[:, 1:]
    g = model.reluNet(x2)
    y1 = x1*torch.exp(-(model.eps + torch.pow(g, 2)))
    return torch.cat((y1, x2), dim=1)

def bERexpInv(model, b):
    b1, b2 = b[:, :1], b[:, 1:]
    g = model.reluNet(b2)
    ap = b1 *torch.exp(model.eps + torch.pow(g, 2)) 
    return ap

def bER(model, x):
    x1, x2 = x[:, :1], x[:, 1:]
    g = model.reluNet(x2)
    y1 = x1/(model.eps + torch.pow(g, 2))
    return torch.cat((y1, x2), dim=1)

def bERinv(model, b):
    b1, b2 = b[:, :1], b[:, 1:]
    g = model.reluNet(b2)
    ap = b1 * (model.eps + torch.pow(g, 2)) 
    return ap 

#baseline: flowFixie
class flowFixie(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowFixie, self).__init__()
        self.mode = mode
        print(self.mode)
        
    def forward(self, x):
        return x    
   
    def inverse(self, b):
        b1, b2 = b[:, :1], b[:, 1:]
        return b1
########################
#LR fit
class flowLR(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowLR, self).__init__()
        self.add_module('reluNet', build_relu(dim-1, hidden_dim, num_layers))
        self.eps = EPS
        print(mode)

    def forward(self, x):
        return bER(self, x)
    
    def loss(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = self.reluNet(x2)
        s = torch.pow(x1[:, :1] -  torch.pow(g, 2), 2)
        return torch.mean(s)

    def inverse(self, b):
        return bERinv(self, b)
####################
#LR distribution
class flowLRflow(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowLRflow, self).__init__()
        self.add_module('reluNet', build_relu(dim-1, hidden_dim, num_layers))
        self.eps = EPS
        self.ell = nn.Parameter(torch.tensor(1).float())
        self.ell.requires_grad = True
        print(mode)

    def forward(self, x):
        return bER(self, x)
        
    def loss(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = self.reluNet(x2)
        b = bER(self, x)[:, :1]
        s = -torch.log(self.eps + sigma(1 *(1 - b))) + torch.log(self.eps + torch.pow(g, 2))
        return torch.mean(s)

    def inverse(self, b):
        return bERinv(self, b)
########################
#LRexp fit
class flowLRexp(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowLRexp, self).__init__()
        self.add_module('reluNet', build_relu(dim-1, hidden_dim, num_layers))
        self.eps = EPS
        print(mode)

    def forward(self, x):
        return bERexp(self, x)
    
    def loss(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = self.reluNet(x2)
        s = torch.pow(x1[:, :1] - torch.pow(g, 2), 2)
        return torch.mean(s)

    def inverse(self, b):
        return bERexpInv(self, b)
#####################
#LRexp distribution
class flowLRexpFlow(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowLRexpFlow, self).__init__()
        self.add_module('reluNet', build_relu(dim-1, hidden_dim, num_layers))
        self.eps = EPS
        self.ell = nn.Parameter(torch.tensor(1).float())
        self.ell.requires_grad = True
        print(mode)

    def forward(self, x):
        return bERexp(self, x)
        
    def loss(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = self.reluNet(x2)
        b = bERexp(self, x)[:, :1]
        s = -torch.log(self.eps + sigma(1 *(1 - b))) + torch.pow(g, 2)
        return torch.mean(s)

    def inverse(self, b):
        return bERexpInv(self, b)
########################

## ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#general NF class
class flowModel(nn.Module):
    def __init__(self, mode, dim,hidden_dim, num_layers):   
        super(flowModel, self).__init__()
        self.dim, self.mode, self.hidden_dim, self.num_layers = dim, mode, hidden_dim, num_layers
        self.type = mode
        if self.type == 'fixed':
            self.add_module('flow', 
                    flowFixie(dim, hidden_dim, num_layers, mode))
        if self.type == 'LR':
            self.add_module('flow', 
                    flowLR(dim, hidden_dim, num_layers, mode))        
        if self.type == 'LRflow':
            self.add_module('flow', 
                    flowLRflow(dim, hidden_dim, num_layers, mode))
        if self.type == 'LRexp':
            self.add_module('flow', 
                    flowLRexp(dim, hidden_dim, num_layers, mode))
        if self.type == 'LRexpFlow':
            self.add_module('flow', 
                    flowLRexpFlow(dim, hidden_dim, num_layers, mode))
            
    def forward(self, x):
        return self.flow(x)

    def inverse(self, x):
        return self.flow.inverse(x)


