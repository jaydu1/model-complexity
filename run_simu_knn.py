import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg

from generate_data import *
from compute_risk import *
from tqdm import tqdm

n_simu = 100
n_test = 1000

# methods and hyperparameters
method = 'kNN'; cov_shift = True; rho_ar1=0.25

##################################################################
# Setting different parameters for different experiments

# For ex2:
sigma = 0.4; coef = 'sorted'; func='quad'; path_result = 'result/ex2/'
#   For overparametrized case:
n = 200; d = 300
#   For underparametrized case:
# n = 500; d = 300

# For ex3:
# sigma = .5; coef = 'random'; path_result = 'result/ex3/'
# for n in np.arange(100,1100,100), compute with the following
# d = 100; func='linear'
# d = 100; func='clf'
##################################################################

params = np.arange(1,d+1)

os.makedirs(path_result, exist_ok=True)
print(method)

def run_one_simulation(method, params, i, cov_shift=False):
    np.random.seed(i)

    Sigma, beta0, X, Y, Y_p, X_test, Y_test, rho2, sigma2 = generate_data(
        n, d, rho_ar1=rho_ar1, sigma=sigma, func=func, coef=coef,
        n_test=n_test, cov_shift=cov_shift, rho_ar1_shift=rho_ar1)

    nu = sigma * np.random.normal(size=(n,1))
    nu_p = sigma * np.random.normal(size=(n,1))
    nu_test = sigma * np.random.normal(size=(n_test,1))
    if cov_shift:
        nu_test = np.r_[nu_test, nu_test]

    res = []
    for _type, (_Y, _Y_p, _Y_test) in zip(['emergent', 'intrinsic'], [[Y, Y_p, Y_test], [nu, nu_p, nu_test]]):
        err_T, err_F, err_R = comp_err(X, _Y, _Y_p, X_test, _Y_test, method, params, cov_shift=cov_shift)
        if cov_shift:
            err_R, err_R_shift = err_R
        else:
            err_R_shift = err_R
        dof_F = (err_F - err_T) / (2*sigma2)
        dof_R = comp_dof(err_R, err_T, sigma2)
        dof_R_shift = comp_dof(err_R_shift, err_T, sigma2)

        _res = np.c_[np.tile([i, n, d, rho2, sigma2, _type, method], [params.shape[0],1]), params, 
                    err_T, err_F, err_R, err_R_shift, dof_F, dof_R, dof_R_shift]
        res.append(_res)
    res = np.concatenate(res, axis=0)
    return res



with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:

    df_res = pd.DataFrame()

    res = parallel(
        delayed(run_one_simulation)(method, params, i, cov_shift) for i in tqdm(np.arange(n_simu), desc='i')
    )

    res = pd.DataFrame(np.concatenate(res,axis=0), columns=
        ['seed', 'n', 'd', 'rho2', 'sigma2', 'type', 'method', 'k',
         'err_T', 'err_F', 'err_R', 'err_R_shift', 'dof_F', 'dof_R', 'dof_R_shift']
    )
    df_res = pd.concat([df_res, res],axis=0)

    df_res.to_csv('{}res_{}_{}_{}_{}_{}_{:.02f}.csv'.format(
        path_result, method, n, d, func, coef, sigma), index=False)    
