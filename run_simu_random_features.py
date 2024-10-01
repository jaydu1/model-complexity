import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from scipy.special import expit

from generate_data import *
from compute_risk import *
from tqdm import tqdm


n_simu = 100
n_test = 1000
p = 300
n = 100; d=300; sigma = 0.4; func='quad'; coef = 'random'; params_p = np.arange(d+1); path_result = 'result/ex2/'; cov_shift=False
rf_func_name = 'tanh'; rf_func = np.tanh

os.makedirs(path_result, exist_ok=True)

method = 'partial_ridge'
params = np.c_[np.full_like(params_p, 0.), params_p]

def run_one_simulation(method, params, i):
    np.random.seed(i)
    Sigma, beta0, X, Y, Y_p, X_test, Y_test, rho2, sigma2 = generate_data(
        n, d, rho_ar1=0., sigma=sigma, func='quad', coef=coef, n_test=n_test)
    F = np.random.normal(0, 1/np.sqrt(d), size=(p, d))
    X = rf_func(X @ F)
    X_test = rf_func(X_test @ F)

    nu = sigma * np.random.normal(size=(n,1))
    nu_p = sigma * np.random.normal(size=(n,1))
    nu_test = sigma * np.random.normal(size=(n_test,1))

    res = []
    for _type, (_Y, _Y_p, _Y_test) in zip(['emergent', 'intrinsic'], [[Y, Y_p, Y_test], [nu, nu_p, nu_test]]):
        err_T, err_F, err_R = comp_err(X, _Y, _Y_p, X_test, _Y_test, method, params)
        dof_F = (err_F - err_T) / (2*sigma2) #comp_dof(err_F, err_T, sigma2)
        dof_R = comp_dof(err_R, err_T, sigma2)
        
        _res = np.c_[np.tile([i, n, d, rho2, sigma2, _type, method], [params.shape[0],1]), params, 
                    err_T, err_F, err_R, dof_F, dof_R]
        res.append(_res)
    res = np.concatenate(res, axis=0)
    return res


with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:
    df_res = pd.DataFrame()
    res = parallel(
        delayed(run_one_simulation)(method, params, i) for i in tqdm(np.arange(n_simu), desc='i')
    )

    res = pd.DataFrame(np.concatenate(res,axis=0), columns=
        ['seed', 'n', 'd', 'rho2', 'sigma2', 'type', 'method', 'lam', 'p',
         'err_T', 'err_F', 'err_R', 'dof_F', 'dof_R']
    )
    df_res = pd.concat([df_res, res],axis=0)

    df_res.to_csv('{}res_rf_{}_{}_{}_{}_{}_{:.02f}.csv'.format(
        path_result, rf_func_name, n, d, func, coef, sigma), index=False)
