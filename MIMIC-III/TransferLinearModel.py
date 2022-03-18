import numpy as np
import scipy as sp 
from sklearn import linear_model

from sklearn import linear_model

def lasso_simple(X, Y, max_iter=1000):
    n, p = X.shape
    a = np.sqrt(np.log(p)/n)
    num_alphas = 100

    # Need to include intercept in X since fit_intercept = True
    lasso = linear_model.LassoCV(cv=10, n_alphas=num_alphas, 
                                 alphas=np.linspace(0.1*a, 2*a, num_alphas), 
                                 fit_intercept=False, max_iter=max_iter).fit(X,Y)

    return(lasso.coef_, lasso.alpha_)

def lasso_path_selected(X, Y):
        n, p = X.shape
        a = np.sqrt(np.log(p)/n)
        alphas_lasso, coefs_lasso, dual_gaps = \
                 linear_model.lasso_path(X, Y, alphas=np.linspace(0.1*a, 2*a, 50))
        i = np.argmin(dual_gaps)
        return(np.squeeze(coefs_lasso)[:,i], alphas_lasso[i])

    
class TransferLinearModel:
    def __init__(self, n_tasks=2):
        self.n_tasks = n_tasks

        
    def set_n_tasks(self, n_tasks):
        self.n_tasks = n_tasks


    def _refine(self, X, Y, coef):
        return(lasso_simple(X, Y - X @ coef, self.max_iter))
        
        
    def least_squares(self, tar, aux):
        X0, Y0 = tar
        X1, Y1 = aux
        
        coef01, _ = lasso_simple(np.vstack([X0, X1]), np.concatenate([Y0, Y1]), self.max_iter)
        delta0, _ = self._refine(X0, Y0, coef01)
        
        return(coef01 + delta0, coef01, delta0)
    