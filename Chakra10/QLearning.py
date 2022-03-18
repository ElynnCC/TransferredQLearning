import numpy as np
import TransferLinearModel as TLM
from sklearn import linear_model


def covariates(sa):
    s = sa[:2]
    a = sa[2:]
    return(np.array([1, s[0], a[0], s[0]*a[0], a[1], s[1]*a[1], a[0]*a[1], s[1]]))


def max_reward(obs, betas, stage):
    num_obs, _ = obs.shape

    sta = obs[:, [1,7]] # horizon 0 and 1 obs are the same

    act0 = obs[:, [2,4]]  # A1, A2 
    act1 = obs[:, [2,4]]  # A1, A2 

    act0[:,stage] = np.ones((num_obs)) * -1
    act1[:,stage] = np.ones((num_obs))

    X0 = np.apply_along_axis(covariates, 1, np.hstack([sta, act0]))
    X1 = np.apply_along_axis(covariates, 1, np.hstack([sta, act1]))

    rew0 = np.hstack([X0, obs[:,8:]]) @ betas
    rew1 = np.hstack([X1, obs[:,8:]]) @ betas

    act_opt = 2 * (rew1 - rew0 > 0).astype(int) - 1
    rew_opt = np.amax( np.hstack([rew0[:,None], rew1[:,None]]), axis=1 )

    return((act_opt, rew_opt))


def LinearRegression(batch):
    
    num_obs, dim_obs, horizon = batch['obs'].shape
    
    hat_ret = np.copy(batch['rew'])

    betas = np.zeros((dim_obs, horizon)) 
    
    # Last stage
    X = batch['obs'][:,:,-1]
    Y = hat_ret[:,-1]
    betas[:,-1] = linear_model.LinearRegression(fit_intercept=False).fit(X, Y).coef_

    # From the second last stage backward
    for t in np.arange(horizon-2, -1, -1):
        _, rew_opt = max_reward( batch['obs'][:,:,t+1], betas[:,t+1], t+1 ) 
        hat_ret[:,t] += rew_opt

        X = batch['obs'][:,:,t]
        Y = hat_ret[:,t]

        betas[:,t] = linear_model.LinearRegression(fit_intercept=False).fit(X, Y).coef_

    return(betas)


def Lasso(batch, max_iter=1000, CV=False):
    
    num_obs, dim_obs, horizon = batch['obs'].shape
    
    hat_ret = np.copy(batch['rew'])

    betas = np.zeros((dim_obs, horizon)) 
    
    X = batch['obs'][:,:,-1]
    Y = hat_ret[:,-1]

    betas[:,-1], _ = TLM.LassoSingle(X, Y, max_iter, CV)

    for t in np.arange(horizon-2, -1, -1):
        _, rew_opt = max_reward( batch['obs'][:,:,t+1], betas[:,t+1], t+1 ) 
        hat_ret[:,t] += rew_opt

        X = batch['obs'][:,:,t]
        Y = hat_ret[:,t]

        betas[:,t], _ = TLM.LassoSingle(X, Y, max_iter, CV)

    return(betas)


def q_value(batch, coef):
    '''
       batch: num_obs x dim_obs x horizon
       coef: dim_obs x horizon
    '''
    
    num_obs, dim_obs, horizon = batch['obs'].shape
    
    qval = np.zeros((num_obs, horizon))
    
    for h in np.arange(horizon):
        qval[:, h] = batch['obs'][:,:,h] @ coef[:,h]
    
    return qval


def optimal_actions(batch, coef):
    
    num_obs, dim_obs, horizon = batch['obs'].shape
    
    opt_acts = np.zeros((num_obs, horizon))
    
    for h in np.arange(horizon):
        opt_acts[:, h], _ = max_reward(batch['obs'][:,:,h], coef[:,h], h) 
        
    return opt_acts
    