import numpy as np
import torch
from functions import *
#########################
currentdir='./'
datadir = currentdir + 'datasets/'
yaxdir = currentdir + 'yax/'
nameScores = ['coverage', 'size', 'WSC']
tryAlpha = [0.05, 0.1, 0.35]
nameStrings = ['baseline', 'ER', 'Gauss', 'Uniform']
outdir = currentdir + 'results/'
###############################################
maes = 0
tables = 1
indexAlpha = [0, 1, 2]
allexps = 0

for datasets in [datasynth, datareal]:
    if datasets == datasynth: 
        datanames = datasynth
        datatype = 'synth'
    else: 
        datanames = datareal
        datatype = 'real'

    # Load results
    resultsdir = outdir + datatype + '/'
    sizes = [[] for iAlpha in indexAlpha]
    coverages = [[] for iAlpha in indexAlpha]
    wscs = [[] for iAlpha in indexAlpha]
    coverageGaps = [[] for iAlpha in indexAlpha]
    wscGaps = [[] for iAlpha in indexAlpha]
    
    # Compute averages and stds and print tables
    for dataset in datasets:
        resultsFile = resultsdir+dataset+'results'
        results = torch.load(resultsFile).numpy()
        means = np.mean(results, axis = 0)
        stds = np.std(results, axis = 0)
        
        maeFile = resultsdir+dataset+'MAEs'
        maes = torch.load(maeFile).numpy()
        maestd = np.std(maes, axis = 0)
        
        for iAlpha in indexAlpha:
            coverages[iAlpha].append([abs(results[:, imodel, iAlpha, 0]).tolist() 
                for imodel in range(len(means))])
            sizes[iAlpha].append([results[:, imodel, iAlpha, 1].tolist() 
                for imodel in range(len(means))])
            wscs[iAlpha].append([abs(results[:, imodel, iAlpha, 2]).tolist() 
                for imodel in range(len(means))])
    coverages, sizes, wscs = np.array(coverages), np.array(sizes), np.array(wscs)
    
    # All datasets average and std
    allscores = np.array([np.mean(np.mean(x, axis = 3), axis=1).squeeze() for x in [coverages, sizes, wscs]])
    allstds = np.array([np.mean(np.std(x, axis = 3), axis=1).squeeze() for x in [coverages, sizes, wscs]])

    # Print all-data results
    if allexps:
        print('-------------\n'+ datatype + '\n---------------')
        
        for iAlpha in indexAlpha:
            q = 'alpha='+str(tryAlpha[iAlpha])
            s = [q] + [', '+x for x in nameScores]
            print(''.join(s))
            for imodel in range(len(allscores[0][iAlpha])):
                name = nameStrings[imodel] + ': '
                values = [np.round(allscores[r][iAlpha][imodel], 3) for r in range(len(allscores))]
                std = [np.round(allstds[r][iAlpha][imodel], 3) for r in range(len(allscores))]
                S = [name]
                for r in range(len(values)):
                    S.append(str(values[r])+'('+str(std[r])+')')
                print(''.join(S))
    
    # Print single-data set results
    if tables:
        values = np.array([np.mean(x, axis=3) for x in [coverages, sizes, wscs]])
        stds = np.array([np.std(x, axis = 3) for x in [coverages, sizes, wscs]])
        for idata in range(len(datasets)):
            
            dataset = datasets[idata]
            print('\n%+++++++++++++++++++++++\n'+ dataset +'\n+++++++++++++++++++++++\n')
            q = datanames[idata] 
            s = [q] + [', '+x for x in nameScores]
            print(''.join(s))
            for iAlpha in indexAlpha:
                print("%--------------------------------------")
                print('alpha='+str(tryAlpha[iAlpha])) 
                iscore = 0
                for imodel in range(len(values[iscore, iAlpha, idata])):
                    name = nameStrings[imodel] + ': '
                    v = [np.round(values[iscore][iAlpha][idata][imodel], 3) 
                            for iscore in range(len(nameScores))]
                    s = [np.round(stds[iscore][iAlpha][idata][imodel], 3) 
                            for iscore in range(len(nameScores))]
                    S = [name]
                    for r in range(len(v)):
                        S.append(str(v[r])+'('+str(s[r])+')')
                    print(''.join(S))
    
    if maes:
        maeFile = resultsdir+'allMaes.npy'
        allMaes = np.load(maeFile)
        mean = (np.mean(allMaes, axis = 1).squeeze())
        std = (np.std(allMaes, axis = 1).squeeze())
        
        print("-------------\n" + datatype + "\n---------------")
        s = ['dataset'] + [', '+ x for x in datanames]
        sv = ['MAE: '] + [', ' + str(np.round(mean[idata], 3))+'('+str(np.round(std[idata], 3))+')' 
                for idata in range(len(mean))]
        print(''.join(s))
        print(''.join(sv))
