
# coding: utf-8

# In[5]:


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
pylab.rcParams['figure.figsize'] = (16, 9)

# analog data assimilation
import sys
sys.path.insert(0, '/home/ptandeo/Dropbox/Documents/Codes/Python/AnDA_CME')
from AnDA_codes.AnDA_generate_data import AnDA_generate_data
from AnDA_codes.AnDA_analog_forecasting import AnDA_analog_forecasting
from AnDA_codes.AnDA_model_forecasting import AnDA_model_forecasting
from AnDA_codes.AnDA_data_assimilation import AnDA_data_assimilation
from AnDA_codes.AnDA_stat_functions import AnDA_RMSE


# In[6]:


F_values = array([6,7,9,10]) # F values of the bad L-96 models
nb_analogs = 200 # number of analogs
nb_dt = 4 # number of dt for the forecast (nb_dt x 0.05 in L96 times)
nb_Ne = 500 # number of ensembles
K = 200 # maximum number of times to compute CME
N_iter = 10 # number of independant observation sets to get confidence intervals
variance_obs = 1 # variance of the observations


# In[7]:


def mooving_average(x, N):
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if N%2 == 0:
            a, b = i - (N-1)//2, i + (N-1)//2 + 2
        else:
            a, b = i - (N-1)//2, i + (N-1)//2 + 1

        #cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out

def model_evidence(ll_good, ll_bad, K, yo):
    out = np.sum((mooving_average(ll_good-ll_bad,K)>0)/len(yo.time)*100)
    return out


# In[9]:


# parameters
class GD:
    model = 'Lorenz_96'
    class parameters:
        F = 8
        J = 40
    dt_integration = 0.05 # integration time
    dt_states = nb_dt # number of integration times between consecutive states (for xt and catalog)
    dt_obs = nb_dt # number of integration times between consecutive observations (for yo)
    var_obs = [17,18,19,20,21] # indices of the observed variables
    nb_loop_train = 1000*nb_dt # size of the catalog
    nb_loop_test = 500*nb_dt # size of the true state and noisy observations
    sigma2_catalog = 0.001 # variance of the model error to generate the catalog   
    sigma2_obs = variance_obs # variance of the observation error to generate observations    
# run the data generation
catalog_good, xt, yo = AnDA_generate_data(GD)

# keep only a subset of variables
catalog_good.analogs = catalog_good.analogs[:,17:22]
catalog_good.successors = catalog_good.successors[:,17:22]
yo.values = yo.values[:,17:22]
n = catalog_good.analogs.shape[1]
global_analog_matrix=np.ones([n,n])

# parameters of the analog forecasting method
class AF:
    k = nb_analogs # number of analogs
    neighborhood = global_analog_matrix
    catalog = catalog_good # catalog with analogs and successors
    regression = 'local_linear' # chosen regression ('locally_constant', 'increment', 'local_linear')
    sampling = 'gaussian' # chosen sampler ('gaussian', 'multinomial')
    kernel = 'tricube'
    initialized=False
# parameters of the filtering method
class DA:
    method = 'AnEnKF' # chosen method ('AnEnKF', 'AnEnKS', 'AnPF')
    N = nb_Ne # number of members (AnEnKF/AnEnKS) or particles (AnPF)
    xb = xt.values[0,17:22]; B = 0.1*np.eye(n)
    H = np.eye(n)
    R = GD.sigma2_obs*np.eye(n)
    @staticmethod
    def m(x):
        return AnDA_analog_forecasting(x,AF)
# run the analog data assimilation
x_hat_analog_good = AnDA_data_assimilation(yo, DA)


# In[ ]:


class yo_iter:
    values = [];
    time = [];
tab_ME_AnDA = zeros([N_iter,len(F_values),K])
for i_iter in range(N_iter):
    p_iter = int(np.shape(yo.values)[0]/N_iter)
    print([i_iter*p_iter, (i_iter+1)*p_iter])
    yo_iter.mro = yo.mro
    yo_iter.time = yo.time[i_iter*p_iter:(i_iter+1)*p_iter]
    yo_iter.values = yo.values[i_iter*p_iter:(i_iter+1)*p_iter,:]
    for i_F in range(len(F_values)):
        print(F_values[i_F])
        # parameters
        class GD:
            model = 'Lorenz_96'
            class parameters:
                F = F_values[i_F]
                J = 40
            dt_integration = 0.05 # integration time
            dt_states = nb_dt # number of integration times between consecutive states (for xt and catalog)
            dt_obs = nb_dt # number of integration times between consecutive observations (for yo)
            var_obs = [17,18,19,20,21] # indices of the observed variables
            nb_loop_train = 1000*nb_dt # size of the catalog
            nb_loop_test = 1 # size of the true state and noisy observations
            sigma2_catalog = 0.001 # variance of the model error to generate the catalog   
            sigma2_obs = variance_obs # variance of the observation error to generate observations    
        # run the data generation
        catalog_bad, tej1, tej2 = AnDA_generate_data(GD)
        # keep only a subset of variables
        catalog_bad.analogs = catalog_bad.analogs[:,17:22]
        catalog_bad.successors = catalog_bad.successors[:,17:22]
        # parameters of the analog forecasting method
        class AF:
            k = nb_analogs # number of analogs
            neighborhood = global_analog_matrix
            catalog = catalog_bad # catalog with analogs and successors
            regression = 'local_linear' # chosen regression ('locally_constant', 'increment', 'local_linear')
            sampling = 'gaussian' # chosen sampler ('gaussian', 'multinomial')
            kernel = 'tricube'
            initialized=False
        # parameters of the filtering method
        class DA:
            method = 'AnEnKF' # chosen method ('AnEnKF', 'AnEnKS', 'AnPF')
            N = nb_Ne # number of members (AnEnKF/AnEnKS) or particles (AnPF)
            xb = xt.values[i_iter*p_iter,17:22]; B = 0.1*np.eye(n)
            H = np.eye(n)
            R = GD.sigma2_obs*np.eye(n)
            @staticmethod
            def m(x):
                return AnDA_analog_forecasting(x,AF)
        # run the analog data assimilation
        x_hat_analog_bad = AnDA_data_assimilation(yo_iter, DA)
        # compute model evidence
        ME_AnDA = zeros(K)
        for k in range(K):
            ME_AnDA[k] = model_evidence(x_hat_analog_good.loglik_center[i_iter*p_iter:(i_iter+1)*p_iter], x_hat_analog_bad.loglik_center, k, yo_iter)
        # stock results
        tab_ME_AnDA[i_iter,i_F,:] = ME_AnDA


# In[ ]:


save('tab_ME_AnDA_noscaling.npy', tab_ME_AnDA)
save('tab_yo_noscaling.npy', yo.time)
