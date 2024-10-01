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
method = 'ridgeless'
lam = 0.0

n = 400; sigma = .4;func='quad'; coef = 'sorted'; path_result = 'result/ex1/'; cov_shift=True
params_gamma = np.linspace(0.1, 2., 50); rho_ar1 = 0.25
os.makedirs(path_result, exist_ok=True)
print(method)

def run_one_simulation(method, params, i, cov_shift=False):
    np.random.seed(i)
    
    params = np.atleast_2d(params)
    p = int(n * params[0,1])
    
    Sigma, beta0, X, Y, Y_p, X_test, Y_test, rho2, sigma2 = generate_data(
        n, p, rho_ar1=rho_ar1, sigma=sigma, func=func, coef=coef,
        n_test=n_test, cov_shift=cov_shift, rho_ar1_shift=rho_ar1)
    
    nu = sigma * np.random.normal(size=(n,1))
    nu_p = sigma * np.random.normal(size=(n,1))
    nu_test = sigma * np.random.normal(size=(n_test,1))
    if cov_shift:
        nu_test = np.r_[nu_test, nu_test]
    res = []
    for _type, (_Y, _Y_p, _Y_test) in zip(
        ['emergent', 'intrinsic'], [[Y, Y_p, Y_test], [nu, nu_p, nu_test]]):
        
        err_T, err_F, err_R = comp_err(X, _Y, _Y_p, X_test, _Y_test, method, 
                                       params[:,:1], cov_shift=cov_shift)
        if cov_shift:
            err_R, err_R_shift = err_R
        else:
            err_R_shift = err_R
        dof_F = (err_F - err_T) / (2*sigma2)
        dof_R = comp_dof(err_R, err_T, sigma2)
        dof_R_shift = comp_dof(err_R_shift, err_T, sigma2)
        
        _res = np.c_[np.tile([i, n, p, sigma, _type, method], [params.shape[0],1]), params, 
                    err_T, err_F, err_R, err_R_shift, dof_F, dof_R, dof_R_shift]
        res.append(_res)
    res = np.concatenate(res, axis=0)
    return res


def run_one_simulation_theory(params, i):
    np.random.seed(i)
    

    def comp(n, lam, gamma):
        d = int(n * gamma)
        Sigma, beta0, _, _, _, _, _, _, _, = generate_data(
            n, d, rho_ar1=rho_ar1, sigma=sigma, func=func, coef=coef,
            n_test=n_test, cov_shift=cov_shift, rho_ar1_shift=rho_ar1)
        err_R = comp_ridge_theoretic_risk(d/n, lam, Sigma, beta0, sigma)
        dof_theory = comp_ridge_dof(d/n, lam, Sigma, beta0, sigma)
        res = np.append([i, n, d, lam, err_R], dof_theory)
        return res
    
    with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:
        res = parallel(
            delayed(comp)(n, lam, gamma) for gamma in tqdm(params, desc='lam')
        )
    
    return res


with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:
        
    df_res = pd.DataFrame()
    params = np.c_[np.full_like(params_gamma, lam), params_gamma]
    res = parallel(
        delayed(run_one_simulation)(method, param, i, cov_shift) 
        for i in tqdm(np.arange(n_simu), desc='i') for param in params
    )

    res = pd.DataFrame(np.concatenate(res,axis=0), columns=
        ['seed', 'n', 'd', 'sigma', 'type', 'method', 'lam', 'gamma',
         'err_T', 'err_F', 'err_R', 'err_R_shift', 'dof_F', 'dof_R', 'dof_R_shift']
    )
    df_res = pd.concat([df_res, res],axis=0)

    df_res.to_csv('{}res_emp_{}_{}_{}_{}_{:.02f}.csv'.format(
        path_result, method, n, func, coef, sigma), index=False)
    

    df_res_theory = pd.DataFrame()
    res = parallel(
        delayed(run_one_simulation_theory)(params_gamma, i) for i in tqdm(np.arange(25), desc='i')
    )

    res = pd.DataFrame(np.concatenate(res,axis=0), columns=
        ['seed', 'n', 'd', 'lam', 'err_R', 'dof_F', 'dof_R_i', 'dof_R_e']
    )
    df_res_theory = pd.concat([df_res_theory, res],axis=0)

    df_res_theory.to_csv('{}res_theory_{}_{}_{}_{}_{:.02f}.csv'.format(
        path_result, method, n, func, coef, sigma), index=False)