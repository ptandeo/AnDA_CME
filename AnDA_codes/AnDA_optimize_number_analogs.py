# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

# analog data assimilation

from AnDA_codes.AnDA_analog_forecasting_with_cov_output import AnDA_analog_forecasting
import numpy as np
from scipy.stats import multivariate_normal


def optimize_nb_analogs(AF,nb_analogs = [50,100,200,300,400,500,750],kfold=5,verbose=False):
    
    T = AF.catalog.analogs.shape[0]
    lf = np.int(np.floor(T/kfold))
    n  = AF.catalog.successors.shape[1]
    rmse_An = np.zeros([n,len(nb_analogs),kfold])
    ll_An_all = np.zeros([len(nb_analogs),kfold])
    ll_An_middle = np.zeros([len(nb_analogs),kfold])
    for v in range(kfold): 
        if verbose: print("Fold ", v+1,"/",kfold)
        i_test = np.arange(v*lf,(v+1)*lf) 
        i_train = np.setdiff1d(np.arange(0,T),i_test)
        class catalog_train:
            analogs = AF.catalog.analogs[i_train,:] 
            successors = AF.catalog.successors[i_train,:] 
        class catalog_test:
            analogs = AF.catalog.analogs[i_test,:] 
            successors = AF.catalog.successors[i_test,:] 
        class AFtmp:
            k = 50 # number of analogs
            neighborhood = AF.neighborhood
            catalog = catalog_train # catalog with analogs and successors
            regression = 'local_linear' # chosen regression ('locally_constant', 'increment', 'local_linear')
            sampling = 'gaussian' # chosen sampler ('gaussian', 'multinomial')
            kernel = AF.kernel # chosen kernel ('gaussian', 'tricube')
            kdt=None
            initialized=False
            global_linear=None

        for j, k in enumerate(nb_analogs):
            if verbose: print("   nb_analogs = ",k)
            # parameters of the analog forecasting method
            AFtmp.k = k
            
    
            x_pred_An  = np.zeros((lf,n))
            cov_An  = np.zeros((lf,n,n))
            for i in range(lf):
                x0 = catalog_test.analogs[i,:]
                xf_An, x_pred_An[i,:], cov_An[i,:,:]  = AnDA_analog_forecasting(x0.reshape((1,n)), AFtmp)
            # comparison to the turth (F=8 test sequence)
            rmse_An[:,j,v]  = np.sqrt(np.mean((catalog_test.successors-x_pred_An)**2,axis=0))
            ll_An_all[j,v] = np.mean([multivariate_normal.pdf(catalog_test.successors[t,:], mean=x_pred_An[t,:], cov=cov_An[t,:,:]) for t in range(1,lf)])
            ll_An_middle[j,v] =np.mean( [multivariate_normal.pdf(catalog_test.successors[t,2], mean=x_pred_An[t,2], cov=cov_An[t,2,2].reshape(1,1)) for t in range(1,lf)])

    rmse_all = np.mean(np.mean(rmse_An,axis=2),axis= 0)
    k_best = nb_analogs[np.where(rmse_all==np.min(rmse_all))[0][0]]                                                        
    k_best_ll = nb_analogs[np.where(ll_An_all==np.min(ll_An_all))[0][0]]  
    k_best_ll_middle = nb_analogs[np.where(ll_An_middle==np.min(ll_An_middle))[0][0]]  
    res = {"k_best": k_best, "k_best_ll": k_best_ll, "k_best_ll_middle": k_best_ll_middle, \
           "rmse": rmse_An, "nb_analogs": nb_analogs}
    return res

    
