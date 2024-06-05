import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from wscClass import *
##############################
fileNames = {
    "concrete": "Concrete_Data.csv",
    "energy": "ENB2012_data.xlsx",
    "facebook_1":"Features_Variant_1.csv", 
    "CASP": "CASP.csv", 
    "community": "communities_attributes.csv", 
    "bike": "bike_train.csv",
    
    "synth-cos": "nofile",
    "synth-inverse": "nofile",
    "synth-linear": "nofile",
    "synth-squared": "nofile",
}
datasynth = [
        'synth-cos', 
        'synth-inverse',
        'synth-linear', 
        'synth-squared', 
        ]
datareal = [
        'bike', 
        'CASP', 
        'community', 
        'concrete',
        'energy', 
        'facebook_1', 
        ]
##################
# Localizer
def build_relu(in_dim, hidden_dim = 100, num_layers = 5):
    v = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for n in range(num_layers):
        v.append(nn.Linear(hidden_dim, hidden_dim))
        v.append(nn.ReLU())
    v.append(nn.Linear(hidden_dim, 1))
    v = OrderedDict([(str(i), v[i]) for i in range(len(v))])
    v = nn.Sequential(v)
    return v

class localizer(nn.Module):
    
    def __init__(self, dim):   
        super(localizer, self).__init__()
        self.add_module('reluNet', build_relu(dim))
    
    def forward(self, x):
        return abs(self.reluNet(x).squeeze())

###################################
# NF models
gamma = .001
class baseline(nn.Module):
    def __init__(self, dim):   
        super(baseline, self).__init__()
        self.eta = 0

    def forward(self, a, x):
        return a
    
    def inverse(self, b, x):
        return b
    
    def loss(self, a, x):
        return 0

class ER(nn.Module):
    def __init__(self, dim, eps = gamma):   
        super(ER, self).__init__()
        self.dim = dim
        self.eps = eps
        self.eta = 1e-2
        self.add_module('localizer', localizer(dim))
    
    def forward(self, a, x):
        w = self.localizer(x)
        return a/(self.eps + w)
    
    def inverse(self, b, x):
        w = self.localizer(x)
        return b * (self.eps + w)
    
    def loss(self, a, x):
        w = self.localizer(x)
        return torch.mean(torch.pow(a - w, 2))

class ML(nn.Module):
    def __init__(self, dim, eps = gamma):   
        super(ML, self).__init__()
        self.dim = dim
        self.eps = eps
        self.eta = 1e-4
        self.add_module('localizer', localizer(dim))
    
    def forward(self, a, x):
        w = self.localizer(x)
        return torch.log(a/(self.eps + w))
    
    def inverse(self, b, x):
        w = self.localizer(x)
        return torch.exp(b) * (self.eps + w)
    
    def loss(self, a, x):
        b = self.forward(a, x)
        return torch.mean(b**2)

class  Uniform(nn.Module):
    def __init__(self, dim, eps = gamma):   
        super(Uniform, self).__init__()
        self.dim = dim
        self.eps = eps
        self.eta = 1e-5
        self.add_module('localizer', localizer(dim))
    
    def sigma(self, t):
        return 1/(1 + torch.exp(-t))

    def logit(self, t):
        reg = 1e-6
        return torch.log(t/(1 - t) + reg)

    def forward(self, a, x):
        w = self.localizer(x)
        return self.sigma(a/(self.eps + w))
    
    def inverse(self, b, x):
        w = self.localizer(x)
        return self.logit(b) * (self.eps + w)
    
    def loss(self, a, x):
        reg = 1e-6
        w = self.localizer(x)
        b = self.forward(a, x)
        return - torch.mean(torch.log(reg + b * (1 - b)/(self.eps + w)))

class wrapper(nn.Module):
    def __init__(self, dim, name):
        super(wrapper, self).__init__()
        self.add_module('score', name(dim))
    
    def forward(self, a, x):
        return self.score(a, x)
    
    def inverse(self, b, x):
        return self.score.inverse(b, x)

    def loss(self, a, x):
        return self.score.loss(a, x)

nameStrings = ['baseline', 'ER', 'ML', 'Uniform']
names = [baseline, ER, ML, Uniform] 
#############################
# CP functions
def computeQuantiles(z, tryAlpha): 
    z = torch.sort(z, dim = 0).values
    q = [z[int(np.ceil((z.shape[0] + 1) * (1 - alpha))) - 1] for alpha in tryAlpha]
    return q

def evaluation(model, a, X, Y):
    model.eval()
    b = model(a, X)
    ical, itest = torch.arange(0, int(len(a)/2)), torch.arange(int(len(a)/2), len(a))
    tryAlpha = [0.05, 0.1, 0.35]
    q = computeQuantiles(b[ical], tryAlpha)
    s = []
    for iq in range(len(tryAlpha)):
        Q = q[iq]
        x = X[itest]
        sets = torch.tensor([model.inverse(Q, z) for z in x]) 
        y = Y[itest]
        condcov = wsc_unbiased(x, a[itest], sets)
        size = torch.mean(sets).item()
        val = torch.mean(1. * (a[itest] <= sets)).item()
        print(tryAlpha[iq],':', val, size, condcov)
        s.append([val, size, condcov])
    return s
    
#############################
# Optimization function

def initialize(p1, eta):
    k, dim = p1
    nameString = nameStrings[k]
    name = names[k]
    model = wrapper(dim, name)
    model.score.eta = min([eta, model.score.eta])
    print(nameString)
    if name not in [baseline]:
        optimizer = optim.Adam(model.parameters(), model.score.eta)
    else: optimizer = None
    return model, optimizer, nameString

def optimize(k, data, T, nval):
    aAll, xAll = data
    dim = xAll.shape[1]
    train = torch.randperm(len(aAll)) 
    N = len(aAll)
    batch = int(N/5)
    if nval:
        train = torch.arange(len(aAll))[:-batch] 
        val = torch.arange(len(aAll))[-batch:]
    p1 = k, xAll.shape[1]
    model, optimizer, nameString = initialize(p1, 1000)
    if optimizer == None: return model, [0]
    obj, mobj = [], []
    t, s, c, old = 0, 0, 0, 1000
    while s == 0:
        choice = torch.randperm(len(train))
        a, x = aAll[choice[:batch]], xAll[choice[:batch], :]  
        optimizer.zero_grad()
        loss = model.loss(a, x)
        if torch.isnan(loss) or loss.item() > 10:
            print('training nan: restarting eta=', model.score.eta)
            t, s, c = 0, 0, c + 1
            p1 = k, xAll.shape[1]
            model, optimizer, nameString = initialize(p1, model.score.eta/10)
            loss = model.loss(a, x)
            if c > 10: return model, mobj
            else: obj, mobj = [], [] 
        loss.backward()
        optimizer.step()
        if nval:
            av, xv = aAll[val], xAll[val, :]
            loss = model.loss(av, xv)
            if loss.item() > 10:
                model, optimizer, nameString = initialize(p1, model.score.eta/10)
            obj.append(loss.item())
            mobj.append(sum(obj)/(len(obj)))
            if t%100 == 0: print(t, mobj[-1])
        t = t + 1
        if t > T: s = 1
    return model, obj


