#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:49:55 2019

@author: valerie
"""

# analog data assimilation

# analog data assimilation
import os
os.chdir('/home/valerie/Dropbox/AnDA_CME')
from AnDA_codes.AnDA_generate_data import AnDA_generate_data
from AnDA_codes.AnDA_analog_forecasting_with_cov_output import AnDA_analog_forecasting
from AnDA_codes.AnDA_optimize_number_analogs import optimize_nb_analogs
from AnDA_codes.AnDA_model_forecasting import AnDA_model_forecasting
from AnDA_codes.AnDA_data_assimilation_temp import AnDA_data_assimilation
from AnDA_codes.AnDA_stat_functions import AnDA_RMSE
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.integrate import odeint
from AnDA_codes.AnDA_dynamical_models import AnDA_Lorenz_63, AnDA_Lorenz_96
import copy
from scipy.stats import multivariate_normal



""" Configuration """
nb_loop_train = 1000 
nb_loop_test = 500 

optim_nb_analogs = False
nb_analogs = [50,100,200,300,400]
nb_analogs = [100,150,200,250,300,350,400,450,500]

dt_states = 1 # number of integration times between consecutive states (for xt and catalog)
dt_obs = 1 # number of integration times between consecutive observations (for yo)
dt_obs_g = 1 # number of integration times between consecutive observations (for yo)
dt_states = dt_obs
save_res, save_fig = True, False

# # parameters
class GD:  
    model = 'Lorenz_96'
    class parameters:
        F = 8
        J = 40
    dt_integration = 0.05 # integration time
    dt_states = dt_states # number of integration times between consecutive states (for xt and catalog)
    dt_obs = dt_obs # number of integration times between consecutive observations (for yo)
    var_obs = [17,18,19,20,21] # indices of the observed variables
    nb_loop_train = nb_loop_train*dt_obs # size of the catalog
    nb_loop_test = nb_loop_test*dt_obs # size of the true state and noisy observations
    sigma2_catalog = 0.001 # variance of the model error to generate the catalog   
    sigma2_obs = 1 # variance of the observation error to generate observations  

# # run the data generation
catalog_good, xt, yo = AnDA_generate_data(GD,seed=True)

# keep only a subset of variables
catalog_good.analogs = catalog_good.analogs[:,17:22]
catalog_good.successors = catalog_good.successors[:,17:22]
yo.values = yo.values[:,17:22]

n = catalog_good.analogs.shape[1]
global_analog_matrix=np.ones([n,n])
if np.all(global_analog_matrix == 1):
    i_var_neighboor = np.arange(n,dtype=np.int64)
    i_var = np.arange(n, dtype=np.int64)
    stop_condition = 1


dt_obs_values = [1,2,4]
F_values = np.array([5,6,7,7.5,8,8.5,9,10,11])
nb_rep = 10
rmse_all, rmse_An_all = np.zeros((nb_rep,len(F_values),len(dt_obs_values))), np.zeros((nb_rep,len(F_values),len(dt_obs_values)))
rmse_An_middle = np.zeros((nb_rep,len(F_values),len(dt_obs_values)))
ll_An_middle_mean = np.zeros((nb_rep,len(F_values),len(dt_obs_values)))
ll_An_all_mean = np.zeros((nb_rep,len(F_values),len(dt_obs_values)))

NB_ANALOGS, NB_ANALOGS_middle = np.zeros((nb_rep,len(F_values),len(dt_obs_values))), np.zeros((nb_rep,len(F_values),len(dt_obs_values)))
NB_ANALOGS_ll, NB_ANALOGS_ll_middle = np.zeros((nb_rep,len(F_values),len(dt_obs_values))), np.zeros((nb_rep,len(F_values),len(dt_obs_values)))

T = xt.values.shape[0]
for i_dt, dt_obs in enumerate(dt_obs_values):
    dt_states = dt_obs
    class GD:  
        model = 'Lorenz_96'
        class parameters:
            F = 8
            J = 40
        dt_integration = 0.05 # integration time
        dt_states = dt_states # number of integration times between consecutive states (for xt and catalog)
        dt_obs = dt_obs # number of integration times between consecutive observations (for yo)
        var_obs = [17,18,19,20,21] # indices of the observed variables
        nb_loop_train = nb_loop_train*dt_obs # size of the catalog
        nb_loop_test = nb_loop_test*dt_obs*(nb_rep+1) # size of the true state and noisy observations
        sigma2_catalog = 0.001 # variance of the model error to generate the catalog   
        sigma2_obs = 1 # variance of the observation error to generate observations  

    catalog_good, xt_long, yo_long = AnDA_generate_data(GD,seed=False) 
    
    for rep in tqdm(range(nb_rep)):
        T_rep = np.int(nb_loop_test*dt_obs/GD.dt_integration)
        class xt:
            values = xt_long.values[(rep*T_rep):((rep+1)*T_rep),:]
            times = np.arange(T_rep)
        class yo:
            values = yo_long.values[(rep*T_rep):((rep+1)*T_rep),:]
            times = np.arange(T_rep)
    
        for i_F,F_bad in enumerate(F_values):
            print("F = ",F_bad)
            class GD_bad:
                model = 'Lorenz_96'
                class parameters:
                    F = F_values[i_F]
                    J = 40
                dt_integration = 0.05 # integration time
                dt_states = dt_states # number of integration times between consecutive states (for xt and catalog)
                dt_obs = dt_obs # number of integration times between consecutive observations (for yo)
                var_obs = [17,18,19,20,21] # indices of the observed variables
                nb_loop_train = nb_loop_train*dt_obs # size of the catalog
                nb_loop_test = nb_loop_test*dt_obs # size of the true state and noisy observations
                sigma2_catalog = 0.001 # variance of the model error to generate the catalog   
                sigma2_obs = 1 # variance of the observation error to generate observations    
        
            catalog_bad, tej1, tej2 = AnDA_generate_data(GD_bad,seed=False)
            # keep only a subset of variables
            catalog_bad.analogs = catalog_bad.analogs[:,17:22]
            catalog_bad.successors = catalog_bad.successors[:,17:22]
            
            # parameters of the analog forecasting method
            # parameters of the analog forecasting method
            class AF:
                k = 200 # number of analogs
                neighborhood = global_analog_matrix
                catalog = catalog_bad # catalog with analogs and successors
                regression = 'local_linear' # chosen regression ('locally_constant', 'increment', 'local_linear')
                sampling = 'gaussian' # chosen sampler ('gaussian', 'multinomial')
                kernel = 'tricube'
                kdt=None
                initialized=False
            class AF_middle:
                k = 200 # number of analogs
                neighborhood = global_analog_matrix
                catalog = catalog_bad # catalog with analogs and successors
                regression = 'local_linear' # chosen regression ('locally_constant', 'increment', 'local_linear')
                sampling = 'gaussian' # chosen sampler ('gaussian', 'multinomial')
                kernel = 'tricube'
                kdt=None
                initialized=False
            if optim_nb_analogs:
                res_bad = optimize_nb_analogs(AF,verbose=True,nb_analogs=nb_analogs)
                print("Best nb analogs", res_bad["k_best"])
                AF.k = res_bad["k_best"]        # parameters of the filtering method
                m_tmp = np.mean(res_bad["rmse"][2,:,:],axis=1)
                print("Best nb analogs (component 19)",nb_analogs[np.where(m_tmp==np.min(m_tmp))[0][0]])
                AF_middle.k = nb_analogs[np.where(m_tmp==np.min(m_tmp))[0][0]]
                print("Best nb analogs (LL)",res_bad["k_best_ll"])
                print("Best nb analogs (LL, component 19)",res_bad["k_best_ll_middle"])

            
                NB_ANALOGS[rep,i_F,i_dt] =  res_bad["k_best"] 
                NB_ANALOGS_middle[rep,i_F,i_dt] =  nb_analogs[np.where(m_tmp==np.min(m_tmp))[0][0]] 
                NB_ANALOGS_ll[rep,i_F,i_dt] =  res_bad["k_best_ll"] 
                NB_ANALOGS_ll_middle[rep,i_F,i_dt] =  res_bad["k_best_ll_middle"] 
    
    
            x_pred_An, x_pred_An_middle  = 0.*xt.values[:,17:22], 0.*xt.values[:,17:22]
            cov_An, cov_An_middle  = np.zeros((xt.values.shape[0],n,n)), np.zeros((xt.values.shape[0],n,n))
#            for i in range(xt.values[0:-1:dt_obs,:].shape[0]):
            for i in range(xt.values[0:-1,:].shape[0]):
                x0 = yo.values[i,:]
                xf_An, x_pred_An[i+1,:], cov_An[i+1,:,:]  = AnDA_analog_forecasting(x0[17:22].reshape((1,n)), AF)
                xf_An, x_pred_An_middle[i+1,:], cov_An_middle[i+1,:,:]  = AnDA_analog_forecasting(x0[17:22].reshape((1,n)), AF_middle)
            ll_An_all = [multivariate_normal.pdf(xt.values[t,17:22], mean=x_pred_An[t,:], cov=cov_An[t,:,:]) for t in range(1,x_pred_An.shape[0])]
            ll_An_middle = [multivariate_normal.pdf(xt.values[t,19], mean=x_pred_An[t,2], cov=cov_An[t,2,2]) for t in range(1,x_pred_An.shape[0])]
    
            rmse_An_all[rep,i_F,i_dt]  = np.sqrt(np.mean((yo.values[1:,17:22]-x_pred_An[1:,:])**2))
            rmse_An_middle[rep,i_F,i_dt]  = np.sqrt(np.mean((yo.values[1:,19]-x_pred_An_middle[1:,2])**2))
            
            ll_An_all_mean[rep,i_F,i_dt] = np.mean(ll_An_all)
            ll_An_middle_mean[rep,i_F,i_dt] = np.mean(ll_An_middle)
            

            """ === Sauvegarde === """
            DAT = {
                   "rmse_An_all":rmse_An_all,
                   "rmse_An_middle":rmse_An_middle,
                   "NB_ANALOGS":NB_ANALOGS,
                   "NB_ANALOGS_middle":NB_ANALOGS_middle,
                   "NB_ANALOGS_ll":NB_ANALOGS_ll,
                   "NB_ANALOGS_ll_middle":NB_ANALOGS_ll_middle,
                   "ll_An_all_mean": ll_An_all_mean,
                   "ll_An_middle_mean": ll_An_middle_mean}
            if save_res:
                filename = 'ForecastRMSE_observations_no_nb_analog_optim.pkl'
                f = open(filename,"wb")
                pickle.dump(DAT,f)
                f.close()
    
load_res = False
#load_res = True
if load_res:
    filename = 'ForecastRMSE.pkl'
    f = open(filename,"rb")
    DAT = pickle.load(f)
    f.close()
    rmse_all, rmse_An_all, rmse_An_middle = DAT["rmse_all"], DAT["rmse_An_all"], DAT["rmse_An_middle"]
    dt_obs_values = [1,2,4]
    F_values = np.array([5,6,7,7.5,8,8.5,9,10,11])
    nb_rep = 1

    

""" === Plots === """
for i_dt, dt_obs in enumerate(dt_obs_values):
    print(i_dt)
    plt.figure(i_dt,figsize =(9,9))
    plt.clf()
   # IC_min = np.array([np.quantile(rmse_all[:,i_F,0],0.025) for i_F in range(len(F_values))])
   # IC_max = np.array([np.quantile(rmse_all[:,i_F,0],0.975) for i_F in range(len(F_values))])
   # plt.fill_between(F_values, IC_min,IC_max,
    #                        color="black", alpha=0.2)
    #plt.plot(F_values,np.array([np.mean(rmse_all[:,i_F,0]) for i_F in range(len(F_values))]),"ko-",label="ode (All components)")
    #plt.plot(F_values,np.array([np.mean(rmse_middle[:,i_F]) for i_F in range(len(F_values))]),"kd:",label="ode")
    
    IC_min = np.array([np.quantile(rmse_An_all[:3,i_F,i_dt],0.025) for i_F in range(len(F_values))])
    IC_max = np.array([np.quantile(rmse_An_all[:3,i_F,i_dt],0.975) for i_F in range(len(F_values))])
    plt.fill_between(F_values, IC_min,IC_max,
                            color="red", alpha=0.2)
    plt.plot(F_values,np.array([np.mean(rmse_An_all[:3,i_F,i_dt]) for i_F in range(len(F_values))]),"r*-",label="LLR (5 components)")
    IC_min = np.array([np.quantile(rmse_An_middle[:3,i_F,i_dt],0.025) for i_F in range(len(F_values))])
    IC_max = np.array([np.quantile(rmse_An_middle[:3,i_F,i_dt],0.975) for i_F in range(len(F_values))])
    plt.fill_between(F_values, IC_min,IC_max,
                            color="orange", alpha=0.2)
    plt.plot(F_values,np.array([np.mean(rmse_An_middle[:3,i_F,i_dt]) for i_F in range(len(F_values))]),"r*:",label="LLR (1 component)")
    plt.grid()
    plt.xlabel("F value",size = 20)
    plt.ylabel("1 step ahead forecast RMSE",size = 16)
    plt.title("Lorenz 96 (dt = "+str(0.05*dt_obs)+")",size = 20)
    plt.legend()
    if save_fig: plt.savefig("RMSE_forecast_F_5to11_dt_obs_"+str(dt_obs)+"_observations_no_nb_analog_optim.png", bbox_inches='tight', dpi=400)

# for i_dt, dt_obs in enumerate(dt_obs_values):
#     print(i_dt)
#     plt.figure(i_dt,figsize =(9,9))
#     plt.clf()
#     # IC_min = np.array([np.quantile(rmse[:,i_F],0.025) for i_F in range(len(F_values))])
#     # IC_max = np.array([np.quantile(rmse[:,i_F],0.975) for i_F in range(len(F_values))])
#     # plt.fill_between(F_values, IC_min,IC_max,
#     #                         color="black", alpha=0.2)
#     #plt.plot(F_values,np.array([np.mean(rmse_all[:,i_F,0]) for i_F in range(len(F_values))])/rmse_all[:,4,i_dt],"ko-",label="ode (All components)")
#     #plt.plot(F_values,np.array([np.mean(rmse_middle[:,i_F]) for i_F in range(len(F_values))]),"kd:",label="ode")
    
#     # IC_min = np.array([np.quantile(rmse_An[:,i_F],0.025) for i_F in range(len(F_values))])
#     # IC_max = np.array([np.quantile(rmse_An[:,i_F],0.975) for i_F in range(len(F_values))])
#     # plt.fill_between(F_values, IC_min,IC_max,
#     #                         color="red", alpha=0.2)
#     plt.plot(F_values,np.array([np.mean(rmse_An_all[:,i_F,i_dt]) for i_F in range(len(F_values))])/rmse_An_all[:,4,i_dt],"r*-",label="LLR (5 components)")
#     plt.plot(F_values,np.array([np.mean(rmse_An_middle[:,i_F,i_dt]) for i_F in range(len(F_values))])/rmse_An_middle[:,4,i_dt],"r*:",label="LLR (1 component)")
#     plt.grid()
#     plt.xlabel("F value",size = 20)
#     plt.ylabel("Relative forecast RMSE",size = 16)
#     plt.title("Lorenz 96 (dt = "+str(0.05*dt_obs)+")",size = 20)
#     plt.legend()
#     if save_fig: plt.savefig("Relative_RMSE_forecast_F_5to11_dt_obs_"+str(dt_obs)+".png", bbox_inches='tight', dpi=400)
