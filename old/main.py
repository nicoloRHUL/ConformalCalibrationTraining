import numpy as np
import argparse
import torch
import torchvision
import torch.nn as nn

from models import *
from functions import *

savedir = 'results'
nExps = 5
seedStart = 12345
testSize = 500
lr = 1e-4
batch_size = 25
defaultNumlayers = 5
defaultNumhiddens = 100
defaultModes = 'fixed-LR-LRflow-LRexp-LRexpFlow'

def train(args, num_epochs = 100):

    modes = args.modes.split(sep='-')
    modelNames = getModelList(modes)
    print("--------------------------------------------------")
    print(modelNames)
    print("--------------------------------------------------")
    
    name = args.dataset
    scores = []
    MSEs = []
    for iExp in range(nExps):
        #initialize the random generator
        seed = seedStart * int(np.exp(iExp))
        torch.manual_seed(seed)
        np.random.seed(seed)
    
        #load or create the data sets
        allSets= getAndSplit(name, seed)
        XYsets = sklearnScaler(allSets)

        #get input-output loaders
        XYloaders = getLoaders(XYsets, batch_size)
        print('XYloader: batch_size, n_batches', 
                XYloaders[1].batch_size, len(XYloaders[0]))
        
        #run RF and get (A, X)-data to train the flows  
        RXloaders, MSE = getRXDatasets(XYloaders)
        print('RXloader: batch_size, n_batches', 
                RXloaders[1].batch_size, len(RXloaders[0]))
        
        #train flow on the training set
        netPars = num_epochs, args.n_hidden, args.n_layers
        print('n epochs, hidden size, n layers', netPars)
        details = []
        tryAlpha=[.05, .1, .32]
        trainIdx = 0         
        for modelName in modelNames:
            flow = trainer(modelName, RXloaders[trainIdx], netPars, lr)
            details.append(flowEvaluator(flow, RXloaders[1], testSize))
            vals, sizes, corrs, wscScore = details[-1][0]#alpha=0.05
            print(modelName, ', [val, size, corr, wsc]:', [vals, sizes, corrs, wscScore])
        scores.append(details)
        MSEs.append(MSE)
    fileName = savedir + '/' + name + '.scores.all.npy'
    np.save(fileName, scores)
    fileName = savedir + '/' + name + '.MSE.all.npy'
    np.save(fileName,MSEs)
    return modelNames, args.dataset




# ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
if __name__ == '__main__':
    
    ############################################################
    #arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, dest='dataset', 
            choices=(
                'housing',
                'homes', 
                'CASP', 
                'concrete', 
                'energy', 
                'facebook_1', 
                'bike', 
                'community', 
                'blog',
                'synth-cos',
                'synth-squared',
                'synth-inverse',
                'synth-linear'
                )
            )
    parser.add_argument("--numlayers", 
            dest='n_layers', 
            default=defaultNumlayers, 
            type=int,
            help="Number of layers in the nonlinearity. [5]"
            )
    parser.add_argument("--numhiddens", 
            dest='n_hidden', 
            default=defaultNumhiddens,
            type=int,
            help="Hidden size of inner layers of nonlinearity. [100]"
            )
    parser.add_argument("--modes", 
            dest='modes', 
            default=defaultModes,
            help="Write model names ['fixed', 'LR', 'LRflow', 'LRexp', 'LRexpFlow'] separated by '-', e.g. fixed-LR."
            )

    args = parser.parse_args()
    
    #################################### 
    modelNames, dataset = train(args)
    printResults(dataset, modelNames)


