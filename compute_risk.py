import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, lars_path
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

import scipy as sp
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
from scipy.optimize import bisect, fixed_point
from scipy.stats import norm

from joblib import Parallel, delayed
from functools import reduce
import itertools

import warnings
warnings.filterwarnings('ignore') 


############################################################################
#
# Theoretical evaluation of ridge
#
############################################################################
# isotopic features

def v_phi_lam(phi, lam, a=1):
    '''
    The unique solution v for fixed-point equation
        1 / v(-lam;phi) = lam + phi * int r / 1 + r * v(-lam;phi) dH(r)
    where H is the distribution of eigenvalues of Sigma.
    For isotopic features Sigma = a*I, the solution has a closed form, which reads that
        lam>0:
            v(-lam;phi) = (-(phi+lam/a-1)+np.sqrt((phi+lam/a-1)**2+4*lam/a))/(2*lam)
        lam=0, phi>1
            v(-lam;phi) = 1/(a*(phi-1))
    and undefined otherwise.
    '''
    assert a>0
    
    min_lam = -(1 - np.sqrt(phi))**2 * a
    if phi<=0. or lam<min_lam:
        raise ValueError("The input parameters should satisfy phi>0 and lam>=min_lam.")
    
    if phi==np.inf:
        return 0
    elif lam!=0:
        return (-(phi+lam/a-1)+np.sqrt((phi+lam/a-1)**2+4*lam/a))/(2*lam)
    elif phi<=1.:
        return np.inf
    else:
        return 1/(a*(phi-1))
    

def vb_phi_lam(phi, lam, a=1, v=None):    
    if lam==0:
        if phi>1:
            return 1/(phi-1)
        else:
            return phi/(phi-1)
    else:
        if v is None:
            v = v_phi_lam(phi,lam,a)
        return phi/(1/a+v)**2/(
            1/v**2 - phi/(1/a+v)**2)
    
    
def vv_phi_lam(phi, lam, a=1, v=None):
    if lam==0:
        if phi>1:
            return phi/(a**2*(phi-1)**3)
        else:
            return np.inf
    else:
        if v is None:
            v = v_phi_lam(phi,lam,a)
        return 1./(
            1/v**2 - phi/(1/a+v)**2)
    
    
def tv_phi_lam(phi, phi_s, lam, v=None):
    if v is None:
        v = v_phi_lam(phi_s,lam)
        
    if v==np.inf:
        return phi/(1-phi)
    elif lam==0. and phi>1:
        return phi/(phi_s**2 - phi)
    else:
        tmp = phi/(1+v)**2
        tv = tmp/(1/v**2 - tmp)
        return tv
    

def tc_phi_lam(phi_s, lam, v=None):
    if v is None:
        v = v_phi_lam(phi_s,lam)
    if v==np.inf:
        return 0.
    elif lam==0 and phi_s>1:
        return (phi_s - 1)**2/phi_s**2
    else:
        return 1/(1+v)**2
    

def vb_lam_phis_phi(lam, phi_s, phi, v=None):    
    if lam==0 and phi_s<=1:
        return 1+phi_s/(phi_s-1)
    else:
        if v is None:
            v = v_phi_lam(phi_s,lam)
        vsq_inv = 1/v**2
        return vsq_inv/(vsq_inv - phi/(1+v)**2)
    
    
def vv_lam_phis_phi(lam, phi_s, phi, v=None):
    if lam==0 and phi_s<=1:
        return np.inf
    else:
        if v is None:
            v = v_phi_lam(phi_s,lam)
        return phi/(
            1/v**2 - phi/(1+v)**2)


    
def v_general(phi, lam, Sigma=None, v0=None):
    if Sigma is None:
        return v_phi_lam(phi, lam)
    else:
        p = Sigma.shape[0]
        
        if phi==np.inf:
            return 0
        elif lam==0 and phi<=1:
            return np.inf        
        
        if v0 is None:
            v0 = v_phi_lam(phi, lam)
            
        v = v0
        eps = 1.
        n_iter = 0
        while eps>1e-3:
            if n_iter>1e4:
                if eps>1e-2: 
                    warnings.warn("Not converge within 1e4 steps.")
                break
            v = 1/(lam + phi * np.trace(np.linalg.solve(np.identity(p) + v * Sigma, Sigma)) / p)
            eps = np.abs(v-v0)/(np.abs(v0)+1e-3)
            v0 = v
            n_iter += 1
        return v


def tv_general(phi, phi_s, lam, Sigma=None, v=None):
    if lam==0 and phi_s<1:
        return phi/(1 - phi)
    if Sigma is None:
        return tv_phi_lam(phi, phi_s, lam, v)
    else:
        if v is None:
            v = v_general(phi_s, lam, Sigma)
        if v==np.inf:
            return phi/(1-phi)
        
        p = Sigma.shape[0]
        tmp = phi * np.trace(
                np.linalg.matrix_power(
                np.linalg.solve(np.identity(p) + v * Sigma, Sigma), 2)
        ) / p
        tv = tmp/(1/v**2 - tmp)
        return tv


def tc_general(phi_s, lam, Sigma=None, beta=None, v=None):
    if lam==0 and phi_s<1:
        return 0
    if Sigma is None:
        return tc_phi_lam(phi_s, lam, v)
    else:
        if v is None:
            v = v_general(phi_s, lam, Sigma)
        if v==np.inf:
            return 0.
        p = Sigma.shape[0]
        tmp = np.linalg.solve(np.identity(p) + v * Sigma, beta[:,None])
        tc = np.trace(tmp.T @ Sigma @ tmp)
        return tc


def omega(psi):
    psi = np.clip(psi, 1e-16, np.inf)
    return 1 + psi / 2 - np.sqrt(1 + psi**2 / 4)


def comp_dof(err_test, err_train,  sigma2):
    psi = (err_test - err_train) / sigma2    
    df = omega(psi)
    return df


def comp_ridge_theoretic_risk(gamma, lam, Sigma, beta, sigma):
    sigma2 = sigma**2
    if gamma == np.inf:
        rho2 = beta.T @ Sigma @ beta
        return rho2 + sigma2
    else:
        v = v_general(gamma, lam, Sigma)        
        tc = tc_general(gamma, lam, Sigma, beta, v)
        tv_s = tv_general(gamma, gamma, lam, Sigma, v)
        B = (1 + tv_s) * tc
        V = (1 + tv_s) * sigma2
        
        return B+V

def comp_ridge_dof(gamma, lam, Sigma, beta, sigma):

    p = Sigma.shape[0]

    v = v_general(gamma, lam, Sigma)
    mu = 1/v
    lamv = 1 - gamma if np.isinf(v) else lam * v
    dof_F = 1 - lamv
    tmp = gamma * np.trace(
                np.linalg.matrix_power(
                np.linalg.solve(mu * np.identity(p) + Sigma, Sigma), 2)
        ) / p
    tmp2 = np.linalg.solve(mu * np.identity(p) + Sigma, beta[:,None])
    V = tmp
    B = mu**2 * np.trace(tmp2.T @ Sigma @ tmp2) / sigma**2
    D = 1 - tmp

    
    dof_R_i = omega((1 - lamv**2) * (V/D + 1))
    dof_R_e = omega((1 - lamv**2) * ((V+B)/D + 1))

    # err_R = comp_ridge_theoretic_risk(gamma, lam, Sigma, beta, sigma)
    # dof_R_e = omega((1 - lamv**2) * err_R / sigma**2)
    return dof_F, dof_R_i, dof_R_e


############################################################################
#
# Theoretical evaluation of Lasso
#
############################################################################
def soft_threshold(x, tau):
    return np.maximum(np.abs(x) - tau, 0.) * np.sign(x)


def F1(a, zeta, epsilon, delta, lam, nu_p):
    '''
    Function F1 in Equation (32a)

    For X ~ N(0,1) and Theta ~ sparse * P_{nu} + (1-sparse) * P_0,
        zeta = sqrt(c * delta) * nu / tau
    '''
    prob = epsilon * (norm.cdf(-a+zeta) + norm.cdf(-zeta-a)) + 2 * (1-epsilon) * norm.cdf(-a) 
    if lam <= 1e-6:
        return prob - delta
    else:
        return lam / np.sqrt(delta) + a * (nu_p / zeta) / delta * (prob - delta)


def F2(zeta, epsilon, delta, lam, nu_p, sigma):
    '''
    Function F2 in Equation (32b)

    For X ~ N(0,1) and Theta ~ epsilon * P_{nu} + (1-epsilon) * P_0,
        zeta = sqrt(c * delta) * nu / tau
        SNR = sparse * nu**2 / sigma**2.

    Parameters
    ----------
    zeta : float
        The ratio nu_p / tau.
    epsilon : float
        The probability of nonzero signals.
    delta : float
        The inverse data aspect ratio n/p.
    lam : float
        The regularization parameter.
    nu_p : float
        The signal strength.
    sigma : float
        The noise level.
    '''
    a_max = np.maximum(1000,1000*lam)
    try:
        a = bisect(F1, 0, a_max, (zeta, epsilon, delta, lam, nu_p))
    except:
        null_risk = epsilon * nu_p**2 + sigma**2
        zetap = nu_p / np.sqrt(null_risk)
        a = bisect(F1, 0, a_max, (zetap, epsilon, delta, lam, nu_p))
    
    f = (sigma**2 * zeta**2 / nu_p**2 - 1) * delta 
    f += epsilon * ((zeta-a) * norm.pdf(zeta+a) + (a**2+1) * norm.cdf(-zeta-a))
    f += epsilon * zeta**2 * (norm.cdf(a-zeta) - norm.cdf(-a-zeta))
    f += epsilon * ((-zeta-a) * norm.pdf(-zeta+a) + (a**2+1) * norm.cdf(zeta-a))
    f += 2 * (1-epsilon) * (-a * norm.pdf(a) + (a**2+1) * norm.cdf(-a))
    f = np.abs(f) + zeta
    
    return f


def compute_risk_lasso(phi, lam, epsilon, nu, sigma, tau=None):
    '''
    Compute the risk of the Lasso ensemble.

    Parameters
    ----------
    phi : float
        The ratio p/n.
    lam : float
        The regularization parameter.
    epsilon : float
        The probability of Gaussian features.
    nu : float
        The signal strength. For X ~ N(0,1) and Theta ~ sparse * P_{nu} + (1-sparse) * P_0,
        SNR = sparse * nu**2 / sigma**2.
    sigma : float
        The noise level.
    '''
    if np.isinf(phi) or np.isinf(lam):
        return epsilon*nu**2+sigma**2, epsilon*nu**2+sigma**2
        
    delta = 1 / phi # The inverse data aspect ratio k/p
    psi = 1 / delta

    nu_p = nu * np.sqrt(delta)
    lam = np.maximum(lam, 1e-7)
    if lam<=1e-6 and psi<=1:
        if psi<1:
            tau2 = 1 / (1 - psi) * sigma**2
            tau = np.sqrt(tau2)
            a = 0
        elif psi==1:
            tau = tau2 = np.inf
            a = 0
    else:
        if tau is None or np.isinf(tau) or np.isnan(tau):
            tau = sigma
        zeta = nu_p/tau
        zeta = fixed_point(F2, zeta, args=(epsilon, delta, lam, nu_p, sigma))
        a = bisect(F1, 0, np.maximum(1000,1000*lam), (zeta, epsilon, delta, lam, nu_p))
        tau = nu_p / zeta
        tau2 = tau**2

    R1 = tau2

    beta = 0. if lam==0 else lam / a
    nu = np.sqrt(delta) * beta / tau

    dof_F = 1 - nu / delta
    tmp = (tau2 - beta**2 / delta)/ sigma**2
    tmp = np.clip(tmp, 1e-16, np.inf)
    dof_R = 1 + tmp / 2 - np.sqrt(1 + tmp**2 / 4)
    return a, tau, R1, dof_F, dof_R



    
############################################################################
#
# Empirical evaluation
#
############################################################################


class Ridgeless(object):
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        self.beta = sp.linalg.lstsq(X, Y, check_finite=False, lapack_driver='gelsy')[0]
        
    def predict(self, X_test):
        return X_test @ self.beta
    
    
class NegativeRidge(object):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def fit(self, X, Y):
        n, p = X.shape
        if n<=p:
            L = np.linalg.cholesky(X.dot(X.T) + self.alpha * np.eye(n))
            self.beta = X.T @ np.linalg.solve(L.T, np.linalg.solve(L, Y))
        else:
            L = np.linalg.cholesky(X.T.dot(X) + self.alpha * np.eye(p))
            self.beta = np.linalg.solve(L.T, np.linalg.solve(L, X.T.dot(Y)))
        
    def predict(self, X_test):
        return X_test @ self.beta
    

class PartialRidge(object):
    def __init__(self, p, alpha, **kwargs):
        self.p = p
        if alpha==0:
            self.cls = Ridgeless(**kwargs)
        elif alpha<0:
            self.cls = NegativeRidge(alpha=alpha, **kwargs)
        else:
            self.cls = Ridge(alpha=alpha, fit_intercept=False, solver='lsqr', **kwargs)
            
    def fit(self, X, Y):
        sqrt_n = np.sqrt(X.shape[0])
        return self.cls.fit(X[:,:self.p]/sqrt_n, Y/sqrt_n)
    
    def predict(self, X):
        return self.cls.predict(X[:,:self.p])

    
# def wrap_class(clf, p, **kwargs):
#     class ClassWrapper(clf):
#         def __init__(self, p=p, **kwargs):
#             super(clf, self).__init__(**kwargs)
#             self.p = p
#             self.clf = clf(**kwargs)

#         def fit(self, X, Y):
#             sqrt_n = np.sqrt(X.shape[0])
#             if self.p>0:
#                 return super(clf, self).fit(X[:,:self.p]/sqrt_n, Y/sqrt_n)
#             else:
#                 self.mean = np.mean(Y)
#                 return self

#         def predict(self, X_test):
#             if self.p>0:
#                 return super(clf, self).predict(X_test[:,:self.p])
#             else:
#                 return np.zeros(X_test.shape[0])
#     return ClassWrapper(p, **kwargs)


# PartialRidge = lambda lam, p : wrap_class(Ridge_Ridgeless, p, alpha=lam)
# PartialLasso = lambda lam, p, **kwargs : wrap_class(Lasso, p, alpha=np.maximum(lam,1e-15), fit_intercept=False, tol=1e-8, max_iter=10000, **kwargs)

    
def fit_predict(X, Y, X_test, method, param, **kwargs):
    if method in ['tree', 'random_forest', 'NN', 'kNN']:
        if method=='tree':
            regressor = DecisionTreeRegressor(max_features=1./3, min_samples_split=5)#, splitter='random')
        elif method=='random_forest':
            regressor = RandomForestRegressor(n_estimators=param[0], max_leaf_nodes=param[1], max_depth=param[2],
                                              max_features='sqrt', bootstrap=False)
        elif method=='NN':
            regressor = MLPRegressor(activation='identity', solver='sgd', 
                                     hidden_layer_sizes=[param[0]], random_state=param[1],
#                                      learning_rate_init=0.1,
                                     early_stopping=True, max_iter=5000)
        elif method=='kNN':
            regressor = KNeighborsRegressor(n_neighbors=param).fit(X, Y)
        regressor = regressor.fit(X, Y)
                
#         if method=='random_forest':
#             estimators = regressor.estimators_
#             def predict(w, i):
#                 regressor.estimators_ = estimators[0:i]
#                 return regressor.predict(X_test)
#             Y_hat = [predict(regressor, i) for i in range(param[0])]
#         else:
        Y_hat = regressor.predict(X_test)
    elif method=='logistic':
        clf = LogisticRegression(
            random_state=0, fit_intercept=False, C=1/np.maximum(param,1e-6)
        ).fit(X, Y.astype(int))
        Y_hat = clf.predict_proba(X_test)[:,1].astype(float)    
    else:
        if method.startswith('partial'):
            lam, p = param
        else:
            lam = param
            p = X.shape[1]
        method = method.replace('partial_','')
        if method.startswith('ridge'):
            regressor = PartialRidge(p, lam)
            # regressor = Ridge(alpha=lam)
            
        elif method.startswith('lasso'):
            regressor = Lasso(alpha=np.maximum(lam,1e-6), 
                    tol=1e-8, max_iter=10000, fit_intercept=False)
            sqrt_k = np.sqrt(X.shape[0])
            # X, Y = X*sqrt_k, Y*sqrt_k
            # regressor = PartialLasso(lam, p)
        else:
            raise ValueError('No implementation for {}.'.format(method))
        
        regressor.fit(X, Y)
        Y_hat = regressor.predict(X_test)
    if len(Y_hat.shape)<2:
        Y_hat = Y_hat[:,None]
    return Y_hat


def fit_predict_support(X, Y, X_test, method, param, beta0, denormalized=False):
    if 'lasso' not in method:
        raise ValueError('Method must be Lasso!')
    if method.startswith('partial'):
        lam, p = param
    else:
        lam = param
        p = X.shape[1]
        
    regressor = Lasso(alpha=np.maximum(lam,1e-6), fit_intercept=False, tol=1e-8, max_iter=10000)
    # regressor = PartialLasso(lam, p)
    # print(denormalized, kwargs)
    if denormalized:
        regressor.fit(X*np.sqrt(X.shape[0]), Y*np.sqrt(X.shape[0]))
    else:
        regressor.fit(X, Y)
    Y_hat = regressor.predict(X_test)
    nnz = np.sum(np.abs(regressor.coef_) > 1e-16)
    if len(Y_hat.shape)<2:
        Y_hat = Y_hat[:,None]
    if beta0 is not None:
        nnz_tp = np.sum((np.abs(regressor.coef_) > 1e-16) & (np.abs(beta0) > 1e-16))
    else:
        nnz_tp = np.nan
    return Y_hat, nnz, nnz_tp



def comp_err(X, Y, Yp, X_test, Y_test, method, params, return_nnz=False, beta0=False, cov_shift=False, **kwargs):
    n_test = Y_test.shape[0]
    X_eval = np.r_[X, X_test]
    
    if return_nnz:
        func = fit_predict_support
    else:
        func = fit_predict
        
    with Parallel(n_jobs=8, verbose=0) as parallel:
        res = parallel(
            delayed(func)(X, Y, X_eval, method, param, beta0=beta0, **kwargs)
            for param in params
        )
        
    if return_nnz:
        res, nnz, nnz_tp = zip(*res)
        nnz = np.array(nnz)
        nnz_tp = np.array(nnz_tp)
    Y_hat = np.stack(res, axis=-1)
#     Y_hat = np.concatenate(res, axis=-1)

    err_T = np.mean((Y_hat[:-n_test,:]-Y[:,:,None])**2, axis=(0,1))
    err_F = np.mean((Y_hat[:-n_test,:]-Yp[:,:,None])**2, axis=(0,1))
    
    if cov_shift:
        n_test = int(n_test/2)
        err_R = np.mean((Y_hat[-2*n_test:-n_test,:,:]-Y_test[:-n_test,:,None])**2, axis=(0,1))
        err_R_shift = np.mean((Y_hat[-n_test:,:,:]-Y_test[-n_test:,:,None])**2, axis=(0,1))
        res = [err_T, err_F, [err_R, err_R_shift]]
    else:    
        err_R = np.mean((Y_hat[-n_test:,:,:]-Y_test[:,:,None])**2, axis=(0,1))
        res = [err_T, err_F, err_R]
        
    if return_nnz:
        res.append(nnz)
        res.append(nnz_tp)
        
    return res



def comp_err_nnz(X, Y, Yp, X_test, Y_test, method, path_solver, beta0, lams=None):
    if method!='lasso':
        raise ValueError('Method must be Lasso!')
    
    alphas, active, coefs = lars_path(X, Y[:,0], alpha_min=0,
                                      method=path_solver, max_iter=1000000, return_path=True)
    
    if lams is not None:
        betas = np.zeros((coefs.shape[0], len(lams)))

        for j, lam in enumerate(lams):
            if lam >= alphas[0]:
                pass
            else:
                i = np.where(lam >= alphas)[0][0]
                betas[:,j] = coefs[:,i-1] + (coefs[:,i] - coefs[:,i-1])/(alphas[i]-alphas[i-1])*(lam-alphas[i-1])
    else:
        betas = coefs
        lams = alphas
    nnz = np.sum(betas!=0, axis=0)
    nnz_tp = np.sum((betas!=0)&(beta0[:,None]!=0), axis=0)
    
    Y_hat = X @ betas
    Y_test_hat = X_test @ betas
    
    err_T = np.mean((Y_hat-Y)**2, axis=0)
    err_F = np.mean((Y_hat-Yp)**2, axis=0)
    err_R = np.mean((Y_test_hat-Y_test)**2, axis=0)
    

#     err_F = 2*np.cov(Y_hat, Y, rowvar=False)[-1,:-1]
#     opt_F = np.clip(opt_F, 1e-16, np.inf)
#     dof_F = 1 + opt_F / 2 - np.sqrt(1 + opt_F**2 / 4)
#     dof_F = np.cov(Y_hat, Y, rowvar=False)[-1,:-1]
    return err_T, err_F, err_R, lams, nnz, nnz_tp#, dof_F
        
    