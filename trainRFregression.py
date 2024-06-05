import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor  #Random Forest algorithm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import urllib.request
from functions import *

##############################################################
#data
def generateInput(N = 2000, d = 3):
    x = -1 + 2 * np.random.rand(N, 1) * 2
    X = []
    for i in range(d):
        q = np.power(x, i)
        X.append(q)
    return np.concatenate(tuple(X), axis=1), x

def generateOutput(X, err):
    w = np.random.randn(len(X[0]), 1)
    y = X @ w + err * np.random.randn(len(X), 1)
    return y.squeeze()

def getDataset(name, base_path = './datasets/'):
    file_name = base_path + fileNames[name]
    
    if name == 'synth-cos':
        X, x = generateInput()
        err = .1 +  np.cos(3.14/2 * abs(x)) * (abs(x)<.5)
        y = generateOutput(X, err)

    
    if name == 'synth-inverse':
        X, x = generateInput()
        err = .1 + 1/ (.1 + abs(x)) * (abs(x) < .5) 
        y = generateOutput(X, err)
    
    if name == 'synth-linear':
        X, x = generateInput()
        err = (.1  + abs(x) * (abs(x) > .5)) 
        y = generateOutput(X, err)
    
    if name == 'synth-squared':
        X, x = generateInput()
        err = .1 + 1 * x * x * (abs(x) >.5) 
        y = generateOutput(X, err)
    
    if name == 'synth-log':
        X, x = generateInput()
        eps = 0.01
        err = .1 + np.log(abs(x) + eps) * (abs(x) <.5) 
        y = generateOutput(X, err)
    
    if name == "concrete":
        data = pd.read_csv(file_name, header=0).values
        data = np.array(data).astype(np.float32)
        X, y = data[:, :-1], data[:, -1]
    
    if name == "energy":
        data = pd.read_excel(file_name, header=0, engine="openpyxl").values
        data = np.array(data).astype(np.float32)
        X, y = data[:, :-1], data[:, -1]
    
    if name=="facebook_1":
        df = pd.read_csv(file_name)        
        y = df.iloc[:,53].values
        X = df.iloc[:,0:53].values
    
    if name=="CASP":
        df = pd.read_csv(file_name)        
        y = df.iloc[:,0].values
        X = df.iloc[:,1:].values        
  
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
        attrib = pd.read_csv(base_path + 'communities_attributes.csv', 
                delim_whitespace = True)
        data = pd.read_csv(base_path + 'communities.data', 
                names = attrib['attributes'])
        data = data.drop(columns=['state', 'county', 'community', 
            'communityname', 'fold'], axis=1)
        data = data.replace('?', np.nan)
        imputer = sklearn.impute.SimpleImputer(missing_values = np.nan, 
                strategy = 'mean')
        imputer = imputer.fit(data)
        data = imputer.transform(data)
        X = data[:, 0:100]
        y = data[:, 100]
    return X, y

def pcaReduction(X, ncomp = 10):
    n = min([X.shape[1], ncomp])
    pca = PCA(n_components=n)
    pca.fit(X)
    x = pca.fit_transform(X)
    return x


#########################
currentdir='./'
datadir = currentdir + 'datasets/'
yaxdir = currentdir + 'yax/'
outdir = currentdir + 'results/'

nExp = 1
nestimators = 20
eps = 1e-4

for datasets in [datasynth, datareal]:
    if datasets == datasynth: 
        names = datasynth
        datatype = 'synth'
        resultsdir = outdir + 'synth/'
    else: 
        names = datareal
        datatype = 'real'
        resultsdir = outdir + 'real/'

    allMaes, setsizes, t = [], [], 0
    for name in names:
        mae = []
        for iExp in range(nExp):
            print('----------------------\n'+str(iExp) +' '+ name+'\n----------------------')
            X, y = getDataset(name, datadir)
            
            # Normalize input and output and reshuffle
            X = X/abs(eps + np.max(X, axis = 0))
            y = y/abs(eps + np.max(y, axis = 0))
            choice = np.random.choice(X.shape[0], size=X.shape[0], 
                    replace=False)
            
            # Reduce size of the training set
            trainsize = min([10000, int(X.shape[0] * 0.5)])
            train_y, train_X = y[choice[:trainsize]], X[choice[:trainsize]]
            print('|train_X|, |train_y|, |choice|', train_X.shape, train_y.shape, choice.shape)
            
            
            # Fit RF
            model = RandomForestRegressor(random_state = len(name) * 1234, 
                    n_estimators = nestimators)
            model.fit(train_X, train_y)
            
            # Test RF and prepare NF-train/test sets
            testsize = min([1000, X.shape[0] - trainsize])
            test_X, test_y = X[choice[-testsize:]], y[choice[-testsize:]]
            validate = model.predict(test_X)
            MAE = np.mean(abs(validate - test_y))
            print('size, MAE', validate.shape[0], MAE)

            # Add regularization positive constant 
            test_a = eps + abs(validate - test_y)
            test_x = pcaReduction(test_X, 10)
            output = np.concatenate(
                    (test_y.reshape(-1, 1),test_a.reshape(-1, 1), test_x), 
                    axis = 1)
            allcols = ['label', 'a']  + [
                    'feature '+ str(i) for i in range(test_x.shape[1])] 
            outputFrame = pd.DataFrame(data = output, columns=allcols)
            
            # Save (A, X) data set
            filename = yaxdir + name + '_yax.csv'
            outputFrame.to_csv(filename, sep=',', header=True, index=False)
            mae.append(MAE)
        allMaes.append(mae) 
        setsizes.append(len(test_y))
    
    fileName = resultsdir + 'allMaes.npy'
    np.save(fileName, allMaes)
    fileName = resultsdir + 'sizes.npy'
    np.save(fileName, setsizes)

