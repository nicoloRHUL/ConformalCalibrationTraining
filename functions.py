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

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init

from tqdm import tqdm, trange
from models import *
from wscClass import *
##############################################################
fileNames = {
    "housing": "housing.data",
    "concrete": "Concrete_Data.csv",
    "energy": "ENB2012_data.xlsx",
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

#data
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
        data = data.drop(columns=['state', 'county', 'community', 'communityname', 'fold'], axis=1)
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
    test_size, proper_size = 0.2, 0.5
    X_1, X_test, y_1, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=test_size, random_state=seed)
    return (X_1, y_1), (X_test, y_test)

def getAndSplit(name, seed):
    X, y = getDataset(name)
    allSets = sklearnSplitter(X, y, seed)
    return allSets

def sklearnScaler(allSets):
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
    return models

def createValidation(loader, ratio = .2, maxVal = 100, random = 1):
    X = torch.cat(tuple([z[0] for z in loader]), dim=0)
    Y = torch.cat(tuple([z[1] for z in loader]), dim=0)
    cut = min([100, int(len(X) * ratio)])
    idx = torch.randperm(len(X))
    X, y = [X[idx[:cut]], X[idx[cut:]]], [Y[idx[:cut]], Y[idx[cut:]]]
    sets = [torch.utils.data.TensorDataset(X[i], y[i]) 
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
    print('training flow:' + mode)
    loaderVal, loaderTrain = createValidation(loader)
    x, y = next(iter(loaderVal))
    input_dim = x.shape[1]
    num_epochs, h_dim, n_layers = netPars
    print("input_dim, batch_size, n_batches", input_dim, x.shape[0], len(loaderTrain))
    
    model = flowModel(mode, input_dim, h_dim, n_layers)

    if mode == 'fixed': 
        return model
    
    opt = optim.Adam(model.parameters(), lr)
    modelPars = mode, input_dim, h_dim, n_layers
    obj = []
    iEpoch = 0
    old = 100000
    r = 100
    t = 5
    while iEpoch < num_epochs:
        model = trainModel(model, loaderTrain, opt)
        inputs = torch.cat(tuple([z[0] for z in loaderVal]), dim=0)
        score = model.flow.loss(inputs)
        if torch.isnan(score) == False: 
            obj.append(score.item())
        else: print('problem')
        iEpoch = iEpoch + 1
        if iEpoch%(t+1)==0: 
            av =sum(obj[-t:])/t 
            print(iEpoch, 'av', av)    
        if iEpoch > t:
            av =sum(obj[-t:])/t 
            if av >= old: 
                print(iEpoch,'--> av=',av) 
                iEpoch = num_epochs
            else: old = av
    bestEpochs = np.argmin(obj)
    print("retraining with best pars ...", bestEpochs)
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
def conformity(f, y):
    a = abs(f - y)
    return a

def getRXDatasets(XYloaders):
    RXdatasets = []
    RXloaders = []
    trainIdx = 0
    Xtrain, ytrain = [torch.cat(
            tuple([z[i].detach() for z in XYloaders[trainIdx]]), dim=0)
            for i in [0, 1]]
    train = Xtrain, ytrain
    forest = randomForest(train)

    for iloader in range(len(XYloaders)):
        loader = XYloaders[iloader]
        dataset = [torch.cat(
            tuple([z[i].detach() for z in loader]), dim=0)
            for i in [0, 1]]
        Xtest, ytest = dataset
        f = forest.predict(Xtest)
        f = torch.tensor(f).unsqueeze(1).float()
        scores = conformity(f, dataset[1])
        MSE = torch.mean(torch.pow(f-dataset[1], 2))
        print('MSE regressor:', MSE )
        RX = torch.cat((scores, dataset[0]), dim=1)
        RY = torch.cat((f, dataset[1]), dim=1)
        RXdatasets.append(torch.utils.data.TensorDataset(RX, RY))
        RXloaders.append(torch.utils.data.DataLoader(RXdatasets[-1], 
            batch_size=loader.batch_size))
    print(len(RXloaders[0]), len(RXloaders[1]))
    return RXloaders, MSE 

###############################
#cp evaluation
def flowEvaluator(flow, loader, testSize):
    flow.eval()
    B = torch.cat(tuple([flow(z[0]).detach() for z in loader]), dim=0)[:testSize, :1]
    AX = torch.cat(tuple([z[0].detach() for z in loader]), dim=0)[:testSize]
    A, X = AX[:, :1], AX[:, 1:]
    AFY = torch.cat(tuple([z[1] for z in loader]), dim=0)[:testSize]
    F, Y = AFY[:, :1], AFY[:, 1:]
    
    details = []
    idx = torch.randperm(len(B))
    calIdx, testIdx = idx[:int(len(B)/2)], idx[int(len(B)/2):]
    bcal = B[calIdx, :1]
    y, f, x = Y[testIdx], F[testIdx], X[testIdx]
    tryAlpha = [0.05, 0.1, 0.32]
    for alpha in tryAlpha:
        print("alpha", alpha)
        n = np.ceil((1 - alpha) * (len(bcal) + 1))
        nq = int(n) - 1
        q = torch.sort(bcal.squeeze(), axis=0)[0][nq] * torch.ones([len(y), 1])
        gap = flow.inverse(torch.cat((q, x), dim=1))[:, :1]
        good = torch.sum(1*(abs(y-f)<=gap))/len(y)
        err = (abs(f-y)).squeeze()
        gap = gap.squeeze()
        corr = torch.dot(err, gap)/torch.sqrt(
                torch.dot(err, err) * torch.dot(gap, gap))
        print('wsc...')
        wscScore = wsc_unbiased(x.detach().numpy(), err.detach().numpy(), gap.detach().numpy())
        detail = [good.item(), torch.mean(gap).item(), corr.item(), wscScore]
        details.append(detail)
    return details


### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#visualize results
### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
#print results on the terminal
def printResults(name, modelNames):
    
    fileName = "results/" +  name + ".scores.all.npy"
    scores = np.load(fileName)

    ialpha = 0
    means = []
    for details in scores:
        m = []
        for imodel in range(len(details)):
            val, size, corr, wscScore = details[imodel][ialpha]
            #print(wscScore)
            m.append([np.mean(s) for s in [val, size, corr, wscScore]]) 
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
        val = str(m[imodel][0]) +'('+str(std[imodel][0])+')' 
        size = str(m[imodel][1]) +'('+str(std[imodel][1])+')'
        corr = str(m[imodel][2]) +'('+str(std[imodel][2])+')' 
        wsc = str(m[imodel][3]) +'('+str(std[imodel][3])+')' 
        print(modelName)
        print("average size:", size)
        print("average error-gap corr:", corr)
        print("average validity:", val)
        print("conditional validity:", wsc,"\n")

    return 0

