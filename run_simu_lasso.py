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
method = 'lasso'

n_simu = 500; n_test = 1000; func='linear' ; path_result = 'result/ex2/'; 
cov_shift = False; sigma = 1.; n = 200; solver = 'lars'; 

##################################################################
# Setting different parameters for different scenarios:

# For overparametrized case:
d=300; params_lam = np.logspace(-4, 0, 250); coef = 'sparse-100'
# For underparametrized case:
# d=30; params_lam = np.logspace(-4, 0, 1000); coef = 'sparse-10'

##################################################################

path_solver = 'lar' if n>d else 'lasso'

os.makedirs(path_result, exist_ok=True)


def run_one_simulation(method, lams, i):
    try:
        np.random.seed(i)
        Sigma, beta0, X, Y, Y_p, X_test, Y_test, rho2, sigma2 = generate_data(
            n, d, rho_ar1=0., sigma=sigma, func=func, coef=coef,
            n_test=n_test,  cov_shift=cov_shift)

        nu = sigma * np.random.normal(size=(n,1))
        nu_p = sigma * np.random.normal(size=(n,1))
        nu_test = sigma * np.random.normal(size=(n_test,1))
        if cov_shift:
            nu_test = np.r_[nu_test, nu_test]
        res = []
        for _type, (_Y, _Y_p, _Y_test) in zip(['emergent', 'intrinsic'], [[Y, Y_p, Y_test], [nu, nu_p, nu_test]]):
            if solver=='lars':
                err_T, err_F, err_R, lams, nnz, nnz_tp = comp_err_nnz(X, _Y, _Y_p, X_test, _Y_test, method, path_solver, beta0, lams)
                err_R_shift = err_R * np.nan
            else:
                err_T, err_F, err_R, nnz, nnz_tp = comp_err(X, _Y, _Y_p, X_test, _Y_test, 
                                                    method, lams, return_nnz=True, beta0=beta0, cov_shift=cov_shift)
                # nnz_tp = np.full_like(nnz, np.nan)
                if cov_shift:
                    err_R, err_R_shift = err_R
                else:
                    err_R_shift = err_R
            dof_F = (err_F - err_T) / (2*sigma2)
            dof_R = comp_dof(err_R, err_T, sigma2)
            dof_R_shift = comp_dof(err_R_shift, err_T, sigma2)

            _res = np.c_[np.tile([i, n, d, sigma, _type, method], [len(nnz),1]), lams, nnz, nnz_tp,
                        err_T, err_F, err_R, err_R_shift, dof_F, dof_R, dof_R_shift]
            res.append(_res)
        res = np.concatenate(res, axis=0)
        return res
    except:
        print(i)
        return np.zeros((0,16))

with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:

    df_res = pd.DataFrame()
    lams = params_lam

    res = parallel(
        delayed(run_one_simulation)(method, lams, i) for i in tqdm(np.arange(n_simu), desc='i')
    )    

    res = pd.DataFrame(np.concatenate(res,axis=0), columns=
        ['seed', 'n', 'd', 'sigma', 'type', 'method', 'lam', 'nnz', 'nnz_tp',
         'err_T', 'err_F', 'err_R', 'err_R_shift', 'dof_F', 'dof_R', 'dof_R_shift']
    )
    df_res = pd.concat([df_res, res],axis=0)

    df_res.to_csv('{}res_{}_{}_{}_{}_{}_{}_{:.02f}.csv'.format(
        path_result, method, solver, n, d, func, coef, sigma), index=False)    
