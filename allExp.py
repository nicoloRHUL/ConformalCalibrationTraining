import numpy as np 
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from functions import *
baseSeed = 543210
######################################################################
localdir = './'
datadir = localdir +'yax/'
outdir = localdir + 'results/'

eta = .00001
T = 3000
testSize = 300 
setup = eta, T
print("\nexp setup:")
print("eta, T", setup)
nExps = 1
for datasets in [datareal, datasynth]:
    if datasets == datasynth: resultsdir = outdir + 'synth/'
    else: resultsdir = outdir + 'real/'
    for idata in range(len(datasets)):
        dataset = datasets[idata]
        datafile = datadir + dataset + '_yax.csv'
        data = pd.read_csv(datafile, header='infer')
        cols = data.columns
        for t in range(len(cols)): print(t, '-', cols[t])
        all_y = data.to_numpy()[:, 0]
        all_a = data.to_numpy()[:, 1]
        all_x = data.to_numpy()[:, 2:]
        all_z = data.to_numpy()[:, 1:]
        print('\nn features', all_x.shape[1])
        print('dataset size', all_x.shape[0])
           
        allscores = []
        allMAEs = []
        for iExp in range(nExps):
            print("------------------\nrun " 
                    + dataset+': '+ str(iExp)+" of "+str(nExps)
                    +"\n-----------------")

            # Split the data into training and test sets
            d = train_test_split(all_z, all_y, 
                    test_size = .5, random_state = baseSeed + iExp * 1234)
            train_X, test_X, train_y, test_y = d
            
            # Separate the scores from the features and convert to torch
            train_a, train_X = [torch.from_numpy(x).float() 
                    for x in [train_X[:, 0], train_X[:, 1:]]]
            test_a, test_X = [torch.from_numpy(x).float() 
                    for x in [test_X[:, 0], test_X[:, 1:]]] 
            train_y, test_y = [torch.from_numpy(x).float()   
                    for x in [train_y, test_y]]
            print('(|train|, |x[0]|, |test|)', 
                    (len(train_a), train_X.shape[1], len(test_y)))
            
            # Compute MAE
            MAE = torch.sum(test_a)/len(test_y)
            print('MAE', MAE.item())
            data = train_a, train_X
            
            # Train and evaluate the NF models
            scores = []
            for k in range(len(nameStrings)):        
                # Train NF on train-val split
                model, obj = optimize(k, data, T, 1)
                bestn = np.argmin(obj)
                print('best val at '+str(bestn)+' iterations: retraining...')
                model, obj = optimize(k, data, bestn, 0)
                a, x = test_a[:testSize], test_X[:testSize, :] 
                y = test_y[:testSize]
                scores.append(evaluation(model, a, x, y))
            
            # Print results
            print('iexp=', iExp)
            print(nameStrings)
            s1 = 'model, val, size, conditional'
            print(s1+'\n------------------')
            for k in range(len(scores)):
                r =  1
                score = np.array(scores[k][r])
                s2 = nameStrings[k]+''.join(
                        [', ' + str(np.round(x, 4))  for x in score])
                print(s2)
            allscores.append(scores)
            allMAEs.append(MAE.item())
        
        # Save results
        allscores = torch.tensor(allscores)
        torch.save(allscores, resultsdir + dataset + 'results')
        print(dataset, 'dataset: scores=\n', torch.mean(allscores, dim=0))
        allMAEs = torch.tensor(allMAEs)
        torch.save(allMAEs, resultsdir + dataset + 'MAEs')
        print(dataset, 'dataset: MAE=', torch.mean(allMAEs, dim=0))

print('done!')            


