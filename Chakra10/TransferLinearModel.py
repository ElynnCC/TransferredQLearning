import numpy as np
import scipy as sp 
from sklearn import linear_model


def LassoSingle(X, Y, max_iter=1000, CV=False):
    n, p = X.shape
    a = np.sqrt(np.log(p)/n)
    
    # Need to include intercept in X since fit_intercept = True
    if CV:
        num_alphas = 100
        lasso = linear_model.LassoCV(cv=10, n_alphas=num_alphas, 
                                 alphas=np.linspace(0.1*a, 2*a, num_alphas), 
                                 fit_intercept=False, max_iter=max_iter).fit(X,Y)
        return(lasso.coef_, lasso.alpha_)
    else:
        lasso = linear_model.Lasso(alpha=a, fit_intercept=False, max_iter=max_iter).fit(X,Y)
        return(lasso.coef_, [a])


def LinearRegression(tar, aux, max_iter=1000):
    X0, Y0 = tar
    X1, Y1 = aux

    coef01 = linear_model.LinearRegression(fit_intercept=False).fit(
        np.vstack([X0, X1]), np.concatenate([Y0, Y1])).coef_
    delta0, _ = LassoSingle(X0, Y0 - X0 @ coef01, max_iter)
   
    return(coef01 + delta0, coef01, delta0)


def Lasso(tar, aux, max_iter=1000):
    X0, Y0 = tar
    X1, Y1 = aux

    coef01, _ = LassoSingle(np.vstack([X0, X1]), 
                             np.concatenate([Y0, Y1]), max_iter)
    delta0, _ = LassoSingle(X0, Y0 - X0 @ coef01, max_iter)

    return(coef01 + delta0, coef01, delta0)
 
    
   

'''
def lasso_path_selected(X, Y):
        n, p = X.shape
        a = np.sqrt(np.log(p)/n)
        alphas_lasso, coefs_lasso, dual_gaps = \
                 linear_model.lasso_path(X, Y, alphas=np.linspace(0.1*a, 2*a, 50))
        i = np.argmin(dual_gaps)
        return(np.squeeze(coefs_lasso)[:,i], alphas_lasso[i])

'''
    
    
        
    