import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg

from generate_data import *
from compute_risk import *
from tqdm import tqdm


# methods and hyperparameters
method_list = ['lasso']
method = 'lasso'
print(method)

n_simu = 100; sigma = 1.; n_test = 2000; func='linear' ; path_result = 'result/ex1/'
d=600; rho = 1; solver = 'cd'; coef = 'sparse-100'

##################################################################
# Setting different parameters for different scenarios:

# For overparametrized case:
n = 400
# For underparametrized case:
# n = 800

##################################################################

os.makedirs(path_result, exist_ok=True)
eps = int(coef.split('-')[1])/d

   
def run_one_simulation(method, lams, i):
    np.random.seed(i)
    Sigma, beta0, X, Y, Y_p, X_test, Y_test, rho2, sigma2 = generate_data(
        n, d, rho_ar1=0., rho=rho, sigma=sigma, func=func, coef=coef,
        n_test=n_test,  cov_shift=False)
    X, X_test = X / np.sqrt(d), X_test / np.sqrt(d)

    nu = sigma * np.random.normal(size=(n,1))
    nu_p = sigma * np.random.normal(size=(n,1))
    nu_test = sigma * np.random.normal(size=(n_test,1))
    
    res = []
    for _type, (_Y, _Y_p, _Y_test) in zip(['emergent', 'intrinsic'], [[Y, Y_p, Y_test], [nu, nu_p, nu_test]]):
        err_T, err_F, err_R, _, _ = comp_err(X, _Y, _Y_p, X_test, _Y_test, 
                                            method, lams, return_nnz=True, beta0=beta0, cov_shift=False, denormalized=True)

        dof_F = (err_F - err_T) / (2*sigma2)
        dof_R = comp_dof(err_R, err_T, sigma2)

        _res = np.c_[np.tile([i, n, d, sigma, _type], [len(err_T),1]), lams,
                    err_T, err_F, err_R, dof_F, dof_R]
        res.append(_res)
    res = np.concatenate(res, axis=0)
    return res


with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:

    df_res = pd.DataFrame()
    lams = params_lam

    res = parallel(
        delayed(run_one_simulation)(method, lams, i) for i in tqdm(np.arange(n_simu), desc='i')
    )    

    res = pd.DataFrame(np.concatenate(res,axis=0), columns=
        ['seed', 'n', 'd', 'sigma', 'type', 'lam',
         'err_T', 'err_F', 'err_R', 'dof_F', 'dof_R']
    )
    df_res = pd.concat([df_res, res],axis=0)

    df_res.to_csv('{}res_{}_{}_{}_{:.02f}.csv'.format(
        path_result, method+'-iso', n, coef, sigma), index=False)



def run_one_simulation_thm(lam, phi, sparse, sigma):
    risk_list = []
    nu = rho / np.sqrt(sparse)
    for _type in ['emergent', 'intrinsic']:
        if _type=='intrinsic':
            sparse = 0
        try:
            a, tau, err_R, dof_F, dof_R = compute_risk_lasso(phi, lam, sparse, nu, sigma)
        except:
            a, tau, err_R, dof_F, dof_R = np.nan, np.nan, np.nan, np.nan, np.nan
            print(phi,lam)
        risk_list.append([phi, lam, sparse, nu, sigma, _type, a, tau, err_R, dof_F, dof_R])
    risk_list = np.array(risk_list)
    return risk_list
    

with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:

    df_res = pd.DataFrame()
    lams = params_lam 

    res = parallel(
        delayed(run_one_simulation_thm)(lam, d/n, eps, sigma) for lam in tqdm(lams, desc='lam')
    )    

    res = pd.DataFrame(np.concatenate(res,axis=0), columns=
        ['phi', 'lam', 'sparse', 'nu', 'sigma', 'type', 'a', 'tau', 'err_R', 'dof_F', 'dof_R']
    )
    df_res = pd.concat([df_res, res],axis=0)

    df_res.to_csv('{}res_risk_{}_{}_{}_{:.02f}.csv'.format(
        path_result, method+'-iso', n, coef, sigma), index=False)

