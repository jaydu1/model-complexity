import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from scipy.sparse.linalg import eigsh
from scipy.linalg import sqrtm
from sklearn.datasets import make_classification


def create_toeplitz_cov_mat(sigma_sq, first_column_except_1):
    '''
    Parameters
    ----------
    sigma_sq: float
        scalar_like,
    first_column_except_1: array
        1d-array, except diagonal 1.
    
    Return:
    ----------
        2d-array with dimension (len(first_column)+1, len(first_column)+1)
    '''
    first_column = np.append(1, first_column_except_1)
    cov_mat = sigma_sq * sp.linalg.toeplitz(first_column)
    return cov_mat


def ar1_cov(rho, n, sigma_sq=1):
    """
    Parameters
    ----------
    sigma_sq : float
        scalar
    rho : float
        scalar, should be within -1 and 1.
    n : int
    
    Return
    ----------
        2d-matrix of (n,n)
    """
    if rho!=0.:
        rho_tile = rho * np.ones([n - 1])
        first_column_except_1 = np.cumprod(rho_tile)
        cov_mat = create_toeplitz_cov_mat(sigma_sq, first_column_except_1)
    else:
        cov_mat = np.identity(n)
    return cov_mat



def generate_data(
    n, p, coef='sorted', func='linear',
    rho=1.,
    rho_ar1=0., sigma=1, df=np.inf, n_test=1000, sigma_quad=1., 
    cov_shift=False, rho_ar1_shift=0.25,
    X=None, X_test=None,
):
    if func=='clf':
        X, Y = make_classification(n+n_test, p, scale=sigma)
        
        
        X_test = X[-n_test:]
        Y_test = Y[-n_test:,None]
        X = X[:-n_test]
        Y = Y[:-n_test,None]
        
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100).fit(X, Y)

        sigma2 = np.mean((clf.predict(X_test) - Y_test[:,0])**2)
        
        if cov_shift:        
            X_test_shift = 1.25 * X_test + 0.25
            X_test = np.r_[X_test, X_test_shift]
            Y_test = np.r_[Y_test, Y_test]
        
        return None, None, X, Y, Y, X_test, Y_test, np.nan, sigma2
        
        
    Sigma = ar1_cov(rho_ar1, p)

    if df==np.inf:
        Z = np.random.normal(size=(n,p))
        Z_test = np.random.normal(size=(n_test,p))
        Z_test_shift = np.random.normal(size=(n_test,p))
    else:
        Z = np.random.standard_t(df=df, size=(n,p)) / np.sqrt(df / (df - 2))
        Z_test = np.random.standard_t(df=df, size=(n_test,p)) / np.sqrt(df / (df - 2))
        Z_test_shift = np.random.standard_t(df=df, size=(n_test,p)) / np.sqrt(df / (df - 2))
        
    Sigma_sqrt = sqrtm(Sigma)
    if X is None or X_test is None:
        X = Z @ Sigma_sqrt
        X_test = Z_test @ Sigma_sqrt
    
    if cov_shift:
        Sigma_shift = ar1_cov(rho_ar1_shift, p)
        
        X_test_shift = 1.25 * Z_test_shift @ sqrtm(Sigma_shift) + 0.25
        X_test = np.r_[X_test, X_test_shift]
    
    if sigma<np.inf:
        if coef.startswith('eig'):
            top_k = int(coef.split('-')[1])
            _, beta0 = eigsh(Sigma, k=top_k)
            rho2 = (1-rho_ar1**2)/(1-rho_ar1)**2/top_k            
            beta0 = np.mean(beta0, axis=-1)
            beta0 = beta0[np.argsort(-np.abs(beta0))]
        elif coef=='sorted':
            beta0 = np.random.normal(size=(p,))
            beta0 = beta0[np.argsort(-np.abs(beta0))] #/ np.sqrt(np.arange(1,p+1))
            beta0 /= np.linalg.norm(beta0)
            rho2 = 1.
        elif coef=='sorted-uniform':
            beta0 = np.random.uniform(-1, 1, size=(p,))
            beta0 = beta0[np.argsort(-np.abs(beta0))] #/ np.sqrt(np.arange(1,p+1))
            beta0 /= np.linalg.norm(beta0)
            rho2 = 1.            
        elif coef=='uniform':
            beta0 = np.ones(p,) / np.sqrt(p)
            rho2 = 1.
        elif coef=='random':
            beta0 = np.random.normal(size=(p,))
            beta0 /= np.linalg.norm(beta0)
            rho2 = 1.
        elif coef.startswith('sparse'):
            s = int(coef.split('-')[1])
            beta0 = np.zeros(p,)
            beta0[:s] = 1/np.sqrt(s)
            rho2 = 1.
    else:
        rho2 = 0.
        beta0 = np.zeros(p)
    beta0 *= rho
    rho2 *= rho**2

    Y = X@beta0[:,None]   
    Y_test = X_test@beta0[:,None]
    
    if func=='linear':
        pass
    elif func=='tanh':
        Y = np.tanh(Y)
        Y_test = np.tanh(Y_test)        
    elif func.startswith('power'):
        coef = float(eval(func.split('-')[1]))
        power = float(eval(func.split('-')[2]))
        Y += coef * Y**power
        Y_test += coef * Y_test**power
    elif func=='quad':
        Y += sigma_quad * (np.mean(X**2, axis=-1) - np.trace(Sigma)/p)[:, None]
        Y_test += sigma_quad * (np.mean(X_test**2, axis=-1) - np.trace(Sigma)/p)[:, None]
    elif func=='nonlinear':
        s = int(p * 0.9)
        S = np.random.choice(p, s, replace=False)
        Sp = np.setdiff1d(np.arange(p), S)
        alpha = 1.
        Y = X[:, S] @ beta0[S,None] + alpha * np.mean(X[:, Sp] > 0., axis=1, keepdims=True)
        Y_test = X_test[:, S] @ beta0[S,None] + alpha * np.mean(X_test[:, Sp] > 0., axis=1, keepdims=True)
    else:
        raise ValueError('Not implemented.')

    if sigma>0.:
        if df==np.inf:
            Yp = Y.copy() + sigma * np.random.normal(size=(n,1))
            Y = Y + sigma * np.random.normal(size=(n,1))            
            Y_test += sigma * np.random.normal(size=(n_test*(1+cov_shift),1))
        else:
            Yp = Y + np.random.standard_t(df=df, size=(n,1))
            Y = Y + np.random.standard_t(df=df, size=(n,1))
            Y_test += np.random.standard_t(df=df, size=(n_test*(1+cov_shift),1))
            sigma = np.sqrt(df / (df - 2))
    else:
        Yp = Y
        sigma = 0.

    return Sigma, beta0, X, Y, Yp, X_test, Y_test, rho2, sigma**2


