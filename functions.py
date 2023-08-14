import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn import ensemble

import os, sys
from os import path
import zipfile
import urllib.request
from os import listdir
from os.path import isfile, join
from collections import OrderedDict

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init

from tqdm import tqdm, trange
##############################################################
fileNames = {
    "housing": "housing.data",
    "concrete": "Concrete_Data.csv",
    "energy": "ENB2012_data.xlsx",
    "homes": "kc-house-data.csv", 
    "facebook_1":"Features_Variant_1.csv", 
    "CASP": "CASP.csv", 
    "blog": "blogData_train", 
    "community": "community.data", 
    "bike": "bike_train.csv",
    
    "synth-cos": "nofile",
    "synth-squared": "nofile",
    "synth-inverse": "nofile",
    "synth-linear": "nofile"
}
EPS = 1e-8

def sigmoid(x, M = 1):
    return 1/(1 + np.exp(- M * x))

def generateInput(N = 1000, d = 3):
    x = -1 + 2 * np.random.rand(N, 1) * 2
    X = []
    for i in range(d):
        q = np.power(x, i)
        X.append(q)
    return np.concatenate(tuple(X), axis=1), x

def generateOutput(X, err):
    w = np.random.randn(len(X[0]), 1)
    return X @ w + err * np.random.randn(len(X), 1)

#data
def getDataset(name):
    base_path='datasets/'
    file_name = base_path + fileNames[name]
    
    if name == 'synth-cos':
        X, x = generateInput()
        err = .1 + 2 * np.cos(3.14/2 * abs(x)) * (abs(x)<.5)
        y = generateOutput(X, err)

    if name == 'synth-squared':
        X, x = generateInput()
        err = .1 + 2 * x * x * (abs(x) >.5) 
        y = generateOutput(X, err)
    
    if name == 'synth-inverse':
        X, x = generateInput()
        err = .1 + 2/ (.1 + abs(x)) * (abs(x) < .5) 
        y = generateOutput(X, err)
    
    if name == 'synth-linear':
        X, x = generateInput()
        err = (.1  + 2 * abs(x) * (abs(x) > .5)) 
        y = generateOutput(X, err)
    
    if name == "housing":
        data = pd.read_csv(file_name, header=0, delimiter="\s+").values
        data = data[np.random.permutation(np.arange(len(data)))]
        X, y = data[:, :-1], data[:, -1]
    
    if name == "concrete":
        data = pd.read_csv(file_name, header=0).values
        data = np.array(data).astype(np.float32)
        X, y = data[:, :-1], data[:, -1]
    
    if name == "energy":
        data = pd.read_excel(file_name, header=0, engine="openpyxl").values
        data = np.array(data).astype(np.float32)
        X, y = data[:, :-1], data[:, -1]
    
    if name=="homes":
        df = pd.read_csv(file_name)
        y = np.array(df['price']).astype(np.float32)
        X = np.matrix(df.drop(['id', 'date', 'price'],axis=1)).astype(np.float32)        
  
    if name=="facebook_1":
        df = pd.read_csv(file_name)        
        y = df.iloc[:,53].values
        X = df.iloc[:,0:53].values
    
    if name=="CASP":
        df = pd.read_csv(file_name)        
        y = df.iloc[:,0].values
        X = df.iloc[:,1:].values        
  
    if name=='blog':
        # https://github.com/xinbinhuang/feature-selection_blogfeedback
        df = pd.read_csv(base_path + 'blogData_train.csv', header=None)
        X = df.iloc[:,0:280].values
        y = df.iloc[:,-1].values

    if name=="bike":
        df=pd.read_csv(base_path + 'bike_train.csv')
        season=pd.get_dummies(df['season'],prefix='season')
        df=pd.concat([df,season],axis=1)
        weather=pd.get_dummies(df['weather'],prefix='weather')
        df=pd.concat([df,weather],axis=1)
        df.drop(['season','weather'],inplace=True,axis=1)
        df.head()
        df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]
        df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
        df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = df['year'].map({2011:0, 2012:1})
        df.drop('datetime',axis=1,inplace=True)
        df.drop(['casual','registered'],axis=1,inplace=True)
        df.columns.to_series().groupby(df.dtypes).groups
        X = df.drop('count',axis=1).values
        y = df['count'].values
    
    if name=="community":
        attrib = pd.read_csv(base_path + 'communities_attributes.csv', delim_whitespace = True)
        data = pd.read_csv(base_path + 'communities.data', names = attrib['attributes'])
        data = data.drop(columns=['state','county',
                          'community','communityname',
                          'fold'], axis=1)
        data = data.replace('?', np.nan)
        imputer = sklearn.impute.SimpleImputer(missing_values = np.nan, strategy = 'mean')
        imputer = imputer.fit(data[['OtherPerCap']])
        data[['OtherPerCap']] = imputer.transform(data[['OtherPerCap']])
        data = data.dropna(axis=1)
        X = data.iloc[:, 0:100].values
        y = data.iloc[:, 100].values

    return X, y

###############################
#sklearn functions

def sklearnSplitter(X, y, seed):
    # ratios = [.4, .4, .2]
    test_size, proper_size = 0.2, 0.5
    X_1, X_test, y_1, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=test_size, random_state=seed)
    X_proper, X_val, y_proper, y_val = sklearn.model_selection.train_test_split(
            X_1, y_1, test_size=proper_size, random_state=seed)
    return (X_proper, y_proper), (X_val, y_val), (X_test, y_test)

def getAndSplit(name, seed):
    # X.shape = N, d
    # y.shape = N, 1
    X, y = getDataset(name)
    allSets = sklearnSplitter(X, y, seed)
    return allSets

def sklearnScaler(allSets):
    # allSets = proper, train, test
    idxTrain = 0 
    scalerX = sklearn.preprocessing.StandardScaler()
    scalerX = scalerX.fit(allSets[idxTrain][0])
    mean_ytrain = np.mean(np.abs(allSets[idxTrain][1]))
    XYsets = []
    for iData in range(len(allSets)): 
        X, y = allSets[iData]
        X, y = np.asarray(X), np.asarray(y)
        X = scalerX.transform(X)
        y = np.squeeze(y)/mean_ytrain
        Z = torch.from_numpy(X).float(), torch.from_numpy(y).float().unsqueeze(1)
        XYsets.append(torch.utils.data.TensorDataset(Z[0], Z[1]))
    return XYsets

#create loaders
def getLoaders(XYsets, batch_size = 16):
    XYloaders =[]
    for iData in range(len(XYsets)): 
        dataset = XYsets[iData]
        XYloaders.append(
                torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, drop_last=True))
    return XYloaders

#get model list
def getModelList(modes):
    models = []
    for iMode in range(len(modes)):
        mode = modes[iMode]
        models.append(mode)
        #if mode =='fixed':
        #    models.append(mode)
            #models.append([mode, 'none', 'none'])
        #else:
        #    if mode == 'LR':
        #        models.append([mode, 'plus', 'LR'])
        #        models.append([mode, 'minus', 'LR'])
        #    models.append([mode, 'plus', 'gauss'])
        #    models.append([mode, 'minus', 'gauss'])
    return models

def split(data, random = 0, cut = 1000):
    X, y = data
    if random:
        idx = torch.randperm(len(X))
    else:
        idx = torch.tensor(range(len(X)))
    X = X[idx[:cut]], X[idx[cut:]]
    y = y[idx[:cut]], y[idx[cut:]]
    return (X[0], y[0]), (X[1], y[1])

def createValidation(loader, ratio = .2, maxVal = 100, random = 1):
    X = torch.cat(tuple([z[0] for z in loader]), dim=0)
    Y = torch.cat(tuple([z[1] for z in loader]), dim=0)
    r = min(maxVal/len(X), ratio) 
    cut = int(len(X) * (1 - r))
    XY = split((X, Y), random, cut) 
    sets = [torch.utils.data.TensorDataset(XY[i][0], XY[i][1]) 
            for i in [0, 1]]
    loaders = [
            torch.utils.data.DataLoader(s, batch_size=loader.batch_size, 
                drop_last=True) for s in sets]
    return loaders

##########################################
######################################################
#############################################################
#optimization
def trainer(mode, loader, netPars, lr = 1e-3):
    #print('training flow:' + mode[0] +'-'+ mode[1])
    print('training flow:' + mode)
    loaderTrain, loaderVal = createValidation(loader)
    x, y = next(iter(loaderVal))
    input_dim = x.shape[1]
    num_epochs, h_dim, n_layers = netPars
    print("input_dim, batch_size, n_batches", input_dim, x.shape[0], len(loaderVal))
    if mode == 'fixed': 
        model = flowModel(mode, input_dim, h_dim, n_layers)
        return model
    
    modelPars = mode, input_dim, h_dim, n_layers
    inputs = torch.cat(tuple([z[0] for z in loaderVal]), dim=0)
    stop = 1
    while stop:
        model = flowModel(mode, input_dim, h_dim, n_layers)
        model.train()
        score1 = model.flow.loss(inputs)
        opt = optim.Adam(model.parameters(), lr)
        opt.zero_grad()
        s = model.flow.loss(inputs)
        s.backward()
        opt.step()
        score2 = model.flow.loss(inputs)
        print('s1, s2:', score1, score2)
        if score2 < score1: 
            stop = 0
            print('restart:', score1, score2)

    print('init:', score1, score2)
    
    #opt = optim.Adam(model.parameters(), lr)

    obj = []
    iEpoch = 0
    old = 100000
    r = 100
    t = 5
    while iEpoch < num_epochs:
        model = trainModel(model, loaderTrain, opt)
        inputs = torch.cat(tuple([z[0] for z in loaderVal]), dim=0)
        score = model.flow.loss(inputs)
        if torch.isnan(score) == False: obj.append(score.item())
        iEpoch = iEpoch + 1
        #t = t + 1
        #if (t * len(loaderTrain) > r):
        #t = 0
        #print(t, score)
        if iEpoch%(t+1)==0: 
            av =sum(obj[-t:])/t 
            print(iEpoch, 'av', av)    
        if iEpoch > t:
            av =sum(obj[-t:])/t 
            if av > old * (1 - lr/10) *.999: 
                print(iEpoch,'--> av=',av) 
                iEpoch = num_epochs
            else: old = av
    print("obj", obj)
    bestEpochs = np.argmin(obj)
    print("retraining with best pars", bestEpochs)
    
    inputs = torch.cat(tuple([z[0] for z in loaderVal]), dim=0)
    stop = 1
    while stop:
        model = flowModel(mode, input_dim, h_dim, n_layers)
        score1 = model.flow.loss(inputs)
        opt = optim.Adam(model.parameters(), lr)
        opt.zero_grad()
        s = model.flow.loss(inputs)
        s.backward()
        opt.step()
        score2 = model.flow.loss(inputs)
        if score2<score1: stop = 0
    print('init:', score1, score2)
    
    epochs = range(bestEpochs)
    for iEpoch in epochs:
        model = trainModel(model, loader, opt)
    return model
    
def trainModel(model, loader, opt):
    model.train()
    dataloader = loader
    for inputs, y in dataloader:
        opt.zero_grad()
        s = model.flow.loss(inputs)
        s.backward()
        opt.step()
    del dataloader
    return model

#############################################################
# random forest
def randomForest(train):
    Xtrain, ytrain = train
    forest = sklearn.ensemble.RandomForestRegressor()
    f = forest.fit(Xtrain, np.ravel(ytrain))
    return f



#######################################
#cp
def A(f, y):
    r = f - y
    a = torch.log(torch.pow(r, 2) + EPS)
    return a

def invA(b):
    return torch.sqrt(torch.exp(b - EPS))

def getRXDatasets(XYloaders):
    #RX = A, X
    #RY = F, Y
    rf= 1
    knn = 0
    RXdatasets = []
    RXloaders = []
    trainIdx = 0
    Xtrain, ytrain = [torch.cat(
            tuple([z[i].detach() for z in XYloaders[trainIdx]]), dim=0)
            for i in [0, 1]]
    if rf:
        train = Xtrain, ytrain
        forest = randomForest(train)

    for iloader in range(len(XYloaders)):
        loader = XYloaders[iloader]
        dataset = [torch.cat(
            tuple([z[i].detach() for z in loader]), dim=0)
            for i in [0, 1]]
        #knn
        if knn:
            f = knn((Xtrain, ytrain), dataset)
        if randomForest:
            Xtest, ytest = dataset
            f = forest.predict(Xtest)
            f = torch.tensor(f).unsqueeze(1).float()
        scores = A(f, dataset[1])
        print('ER', torch.mean(torch.pow(f-dataset[1], 2)))
        #print(torch.min(torch.pow(f-dataset[1], 2)))
        #print(torch.max(torch.pow(f-dataset[1], 2)))
        RX = torch.cat((scores, dataset[0]), dim=1)
        RY = torch.cat((f, dataset[1]), dim=1)
        RXdatasets.append(torch.utils.data.TensorDataset(RX, RY))
        RXloaders.append(torch.utils.data.DataLoader(RXdatasets[-1], 
            batch_size=loader.batch_size))
    print(len(RXloaders[0]), len(RXloaders[1]))
    return RXloaders 

###############################
#cp evaluation
def flowEvaluator(flow, loader, testSize):
    print("evaluating", flow.mode)
    
    flow.eval()
    B = torch.cat(tuple([flow(z[0]).detach() for z in loader]), dim=0)[:testSize, :1]
    AX = torch.cat(tuple([z[0].detach() for z in loader]), dim=0)[:testSize]
    A, X = AX[:, :1], AX[:, 1:]
    AFY = torch.cat(tuple([z[1] for z in loader]), dim=0)[:testSize]
    F, Y = AFY[:, :1], AFY[:, 1:]
    
    details = []
    idx = torch.randperm(len(B))
    calIdx, testIdx = idx[:int(len(B)/2)], idx[int(len(B)/2):]
    tryAlpha = [0.05, 0.1, 0.32]
    for alpha in tryAlpha:
        print(alpha)
        detail = []
        for i in testIdx:
            b = B[calIdx]
            a = A[calIdx]
            x = X[calIdx]
            nq = int((1 - alpha) * len(b))
            q = torch.sort(b, 0)[0][nq] * torch.ones([1, 1])
            gap = invA(flow.inverse(torch.cat((q, X[i:i+1]), dim=1))[:, :1])  
            good = 1. * (Y[i] < (F[i] + gap)) * (Y[i] > (F[i] - gap))
            xT = X[i, 1]
            yT = Y[i] 
            fT = F[i]
            uT = F[i] + gap
            dT = F[i] - gap
            rT = min([abs(uT-yT), abs(dT-yT)])
            detail.append([s.item() for s in [gap, rT, good, xT, yT, fT, uT, dT]])
        details.append(detail)
    return details

### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#models
### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

#utils 
def build_relu(in_dim, hidden_dim, num_layers):
    v = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for n in range(num_layers):
        v.append(nn.Linear(hidden_dim, hidden_dim))
        v.append(nn.ReLU())
    v.append(nn.Linear(hidden_dim, 1))
    v = OrderedDict([(str(i), v[i]) for i in range(len(v))])
    v = nn.Sequential(v)
    return v

def normalizeModel(net, x):
    gx = net(x)
    g0 = net(torch.zeros(x.shape))
    return gx# - g0

#flowFixie
class flowFixie(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowFixie, self).__init__()
        self.mode = mode
        print(self.mode)
        
    def forward(self, x):
        return x    

    def logJacobian(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        return torch.cat((torch.ones(x1.shape), x2), dim=1)
    
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
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet,x2)
        y1 = torch.exp(x1)/(self.eps + torch.pow(g, 2))
        return torch.cat((y1, x2), dim=1)
    
    def loss(self, x):
        g = normalizeModel(self.reluNet, x[:, 1:])
        s = torch.pow(torch.exp(x[:, :1]) -  torch.pow(g, 2), 2)
        return torch.sum(s)

    def inverse(self, b):
        b1, b2 = b[:, :1], b[:, 1:]
        g = normalizeModel(self.reluNet, b2)
        ap = torch.log(b1 * (self.eps + torch.pow(g, 2)))
        return ap
####################
#exp distribution
class flowLRExp(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowLRExp, self).__init__()
        self.add_module('reluNet', build_relu(dim-1, hidden_dim, num_layers))
        self.eps = EPS
        self.ell = nn.Parameter(torch.tensor(1).float())
        self.ell.requires_grad = True
        print(mode)

    def forward(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet,x2)
        y1 = torch.exp(x1)/(self.eps + torch.pow(g, 2))
        return torch.cat((y1, x2), dim=1)
    
    def loss(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet, x2)
        y1 = torch.exp(x1)/(self.eps + torch.pow(g, 2))
        scale = torch.pow(self.ell, 2)
        s = -torch.log(scale) +  scale * y1 - x1 + torch.log(torch.pow(g, 2))
        return torch.sum(s)

    def inverse(self, b):
        b1, b2 = b[:, :1], b[:, 1:]
        g = normalizeModel(self.reluNet, b2)
        ap = torch.log(b1 * (self.eps + torch.pow(g, 2)))
        return ap
########################

#uniform ditribution
class flowSigma(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowSigma, self).__init__()
        self.add_module('reluNet', build_relu(dim-1, hidden_dim, num_layers))
        self.eps = EPS
        self.ell = nn.Parameter(torch.tensor([2, 2]).float())
        self.ell.requires_grad = True
        print(mode)

    def forward(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet,x2)
        y1 = torch.sigmoid(x1 + g)
        return torch.cat((y1, x2), dim=1)
    
    def loss(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet, x2)
        b = torch.sigmoid(x1 + g)
        al, bet = torch.pow(self.ell, 2)
        lz = torch.lgamma(al) + torch.lgamma(bet)- torch.lgamma(al+bet)
        j = b * (1 - b)
        p = torch.pow(b, al-1) * torch.pow(1 - b, bet-1)
        s = - torch.log(j + self.eps) + lz -torch.log(p + self.eps)
        return torch.sum(s)

    def inverse(self, b):
        b1, b2 = b[:, :1], b[:, 1:]
        g = normalizeModel(self.reluNet, b2)
        ap = torch.logit(b1) - g
        return ap


########################
#gauss ditribution
class flowSum(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowSum, self).__init__()
        self.add_module('reluNet', build_relu(dim-1, hidden_dim, num_layers))
        self.eps = EPS
        self.ell = nn.Parameter(torch.tensor([0, 1]).float())
        self.ell.requires_grad = True
        print(mode)

    def forward(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet,x2)
        y1 = x1 * torch.exp(g)
        return torch.cat((y1, x2), dim=1)
    
    def loss(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet, x2)
        sigma = torch.pow(self.ell[1], 2) + self.eps
        b = x1 * torch.exp(g)
        j = torch.exp(g)
        s = - g + torch.pow((b - self.ell[0])/sigma, 2)/2 + torch.log(torch.sqrt(sigma))
        return torch.sum(s)

    def inverse(self, b):
        b1, b2 = b[:, :1], b[:, 1:]
        g = normalizeModel(self.reluNet, b2)
        ap = b1 * torch.exp(-g)
        return ap


## ### ### ### ### ### ### ### ### ### ### ### ### ### ###
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
        
        if self.type == 'LRExp':
            self.add_module('flow', 
                    flowLRExp(dim, hidden_dim, num_layers, mode))
        
        if self.type == 'sigma':
            self.add_module('flow', 
                    flowSigma(dim, hidden_dim, num_layers, mode))
        
        if self.type == 'sum':
            self.add_module('flow', 
                    flowSum(dim, hidden_dim, num_layers, mode))
    
    def forward(self, x):
        return self.flow(x)

    def inverse(self, x):
        return self.flow.inverse(x)


### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#visualize results
### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#print results on the terminal
def printResults(name, modelNames,visual):
    
    #select all-experiment file 
    fileName = "results/" +  name + ".scores.all"
    scores = torch.load(fileName)

    ialpha = 0
    means = []
    for details in scores:
        m = []
        for imodel in range(len(details)):
            gaps = [s[0] for s in details[imodel][ialpha]]
            vals = [s[2] for s in details[imodel][ialpha]]
            gte = [s[1] for s in details[imodel][ialpha]]
            m.append([np.mean(s) for s in [gaps, gte, vals]]) 
        means.append(m)
    means = np.array(means)
    
    names = modelNames

    #print averages and standard deviations
    print("---------------------------------\n",
            name,
            "\n---------------------------------")
    m = np.round(np.mean(means, axis=0), 3)
    std = np.round(np.std(means, axis=0), 3)
    for imodel in range(len(names)):
        modelName = names[imodel]
        val = str(m[imodel][2]) +'('+str(std[imodel][2])+')' 
        size = str(m[imodel][0]) +'('+str(std[imodel][0])+')'
        gte = str(m[imodel][1]) +'('+str(std[imodel][1])+')' 
        print(modelName)
        print("average size:", size)
        print("average gap-to-error:", gte)
        print("average validity:", val,"\n")

    #plot the intervals for the synthetic data sets
    #[gap, rT, good, xT, yT, fT, uT, dT for alpha=.05,.1,.32]
    if visual == 0: return 0
    expIndex = 0
    expName = 'iE_' + str(expIndex)
    fileName = 'results/' + name + '.details.' + expName +'.npy'
    intervals = np.load(fileName, allow_pickle=True)
    names = intervals[-1]
    intervals = [x for x in intervals[:-1]]
    legends = [s for s in names]
    legends.append('true')
    legends.append('predictions')
    plt.figure(figsize=(10, 10))
    colors = ['r.', 'b.', 'g.', 'y.', 'c.', 'm.']
    ialpha = 0
    for iModel in range(len(intervals[:-1])):
        details = np.array(intervals[iModel][ialpha])
        print(details.shape)
        gap, rt, good, x, y, f, u, d = [details[:, i] for i in range(len(details[0]))]
        color = colors[iModel]
        plt.plot(x, u, color, alpha=.8, label=legends[iModel])
        plt.plot(x, d, color, alpha=.8)
    plt.plot(x, y, 'ok', alpha=.3, label = legends[-2])
    plt.plot(x, f, '*k', markersize=3, alpha=.8, label=legends[-1])
    plt.legend(prop={'size':20})#, loc='lower left')
    fileName = 'results/' + name + '.plot'+expName+'.pdf'
    plt.savefig(fileName)
    plt.show()
    return 0


"""
#uniform ditribution
class flowSigma(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowSigma, self).__init__()
        self.add_module('reluNet', build_relu(dim-1, hidden_dim, num_layers))
        self.eps = EPS
        print(mode)

    def forward(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet,x2)
        y1 = torch.sigmoid(x1 + g)
        return torch.cat((y1, x2), dim=1)
    
    def loss(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet, x2)
        b = torch.sigmoid(x1 + g)
        j = b * (1 - b)
        s = - torch.log(j + self.eps)
        return torch.sum(s)

    def inverse(self, b):
        b1, b2 = b[:, :1], b[:, 1:]
        g = normalizeModel(self.reluNet, b2)
        ap = torch.logit(b1) - g
        return ap

class flowSum(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, mode):   
        super(flowSum, self).__init__()
        self.add_module('reluNet', build_relu(dim-1, hidden_dim, num_layers))
        self.eps = EPS
        self.ell = nn.Parameter(torch.tensor([0, 1]).float())
        self.ell.requires_grad = True
        print(mode)

    def forward(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet,x2)
        y1 = x1 + g
        return torch.cat((y1, x2), dim=1)
    
    def loss(self, x):
        x1, x2 = x[:, :1], x[:, 1:]
        g = normalizeModel(self.reluNet, x2)
        sigma = torch.pow(self.ell[1], 2) + self.eps
        s = torch.pow((x1 + g - self.ell[0])/sigma, 2)/2 + torch.log(torch.sqrt(sigma))
        return torch.sum(s)

    def inverse(self, b):
        b1, b2 = b[:, :1], b[:, 1:]
        g = normalizeModel(self.reluNet, b2)
        ap = b1 - g
        return ap
"""

