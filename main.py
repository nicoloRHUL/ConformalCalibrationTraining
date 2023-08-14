import numpy as np
import argparse
import torch
import torchvision
import torch.nn as nn



from functions import *

savedir = 'results'
nExps = 5
seedStart = 12345
testSize = 500
lr = 1e-4
batch_size = 25
defaultNumlayers = 5
defaultNumhiddens = 100
defaultModes = 'fixed-LR-LRExp-sigma-sum'

def train(args, num_epochs = 100):

    modes = args.modes.split(sep='-')
    modelNames = getModelList(modes)
    print("--------------------------------------------------")
    print(modelNames)
    print("--------------------------------------------------")
    
    name = args.dataset
    scores = []
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
        RXloaders = getRXDatasets(XYloaders)
        print('RXloader: batch_size, n_batches', 
                RXloaders[1].batch_size, len(RXloaders[0]))
        
        netPars = num_epochs, args.n_hidden, args.n_layers
        print('n epochs, hidden size, n layers', netPars)
        
        details = []
        #[gap, rT, good, xT, yT, fT, uT, dT for alpha=.05,.1,.32]
        trainIdx = 1 #train flow on the second part ot the training set
        for modelName in modelNames:
            flow = trainer(modelName, RXloaders[trainIdx], netPars, lr)
            details.append(flowEvaluator(flow, RXloaders[2], testSize))
        #details.append(modelNames)
        
        fileName = savedir + '/'+ name + '.details.iE_' + str(iExp) + '.npy'
        np.save(fileName, details)
        tryAlpha=[.05, .1, .32]
        ialpha = 0
        print("alpha", tryAlpha[ialpha])
        for imodel in range(len(details)):
            gaps = [s[0] for s in details[imodel][ialpha]]
            vals = [s[2] for s in details[imodel][ialpha]]
            gte = [s[1] for s in details[imodel][ialpha]]
            print(modelNames[imodel], ', [gaps, gte, vals]:', [np.mean(s) for s in [gaps, gte, vals]] ) 
        scores.append(details)
    fileName = savedir + '/' + name + '.scores.all'
    torch.save(scores, fileName)

    #print(scores, 1)
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
            help="Write model names ['fixed', 'LR', 'invLR', 'sum', 'invSum'] separated by '-', e.g. fixed-LR."
            )

    args = parser.parse_args()
    
    #################################### 
    modelNames, dataset = train(args)
    #dataset = args.dataset
    #for synth
    #printResults(dataset, modelNames, 1)
    #for real
    printResults(dataset, modelNames, 0)


