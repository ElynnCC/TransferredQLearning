import numpy as np
import scipy as sp


def backward_cumsum(x):
    return np.cumsum(x[::-1])[::-1]


'''
    MDP (Chakraborty et al. 2010; Song et al. 2015)
'''
class MDP_CHAK10:
    def __init__(self, dim_obs, noise_var, r2_coef=None, distn='n', gamma=1):
        '''
            State: {-1, 1}
            Action: {-1, 1}
            Transition: expit(.)
            Rewards: stage_0: 0; stage_1: linear in s, a, sa
            Optimal Q: linear for both stage
        '''
        # Dimension of non-zero coefficients = 8 ...
        self.dim_tru = 8
        assert dim_obs >= self.dim_tru
        self.dim_obs = dim_obs
        
        # Distiribution of irrelevant covariates
        if distn not in ('n', 'u'):
            raise ValueError("MDP observation distribution " 
                             + distn + " NOT accpeted!")
        else:
            self.distn = distn
        
        # Horizon
        self.horizon = 2
        
        # Stage 1 reward is always 0 ...
        # Stage 2 reward function is linear ...
        if r2_coef is None:
            self.r2_coef = np.ones(self.dim_tru)# 1, S1, A1, S1A1, A2, S2A2, A1A2, S2
            self.r2_coef[-1] = 0 # coef of S2 = 0
        elif len(r2_coef) == self.dim_tru:
            self.r2_coef = r2_coef # 1, S1, A1, S1A1, A2, S2A2, A1A2, S2
            self.r2_coef[-1] = 0 # coef of S2 = 0
        else:
            raise ValueError("The dimensions of r2_coef is not self.dim_tru!")
        
        # Transition probability coefficients (expit func) are fixed at 1 ...
        self.pr_coef = np.ones(2)
        
        # Reward of 2nd stage has noise ~ Normal(0, noise_var) ...
        self.noise_var = noise_var
        
        # True Q* func is linear and has analytical form ...
        Q_coef_tru = self.get_Q_coef() # dim_obs x horizon
        
        # For online learning (added on May 6th, 2021)
        assert 0 <= gamma <= 1
        self.gamma = gamma
        self.batch_size = 1 # defualt at 1 task 
        
        self.done = 0
        self.stage = 0
        self.reward = None
        self.state = None
        self.observe = None
        self.action = None
        self.q_value = None
        
        
    def set_r2_coef(self, r2_coef):
        assert len(r2_coef) == self.dim_tru
        self.r2_coef = r2_coef # 1, S1, A1, S1A1, A2, S2A2, A1A2, S2
        self.r2_coef[-1] = 0 # coef of S2 = 0
        
        
    def get_r2_coef(self):
        # 1, S1, A1, S1A1, A2, S2A2, A1A2, S2
        return self.r2_coef

    
    def get_Q_coef(self):
        Q_coef = np.zeros((self.dim_obs, self.horizon))
        
        Q_coef[:self.dim_tru,1] = self.r2_coef
        
        bs, ba = self.pr_coef
        
        q1 = .25 * (sp.special.expit(bs + ba) + sp.special.expit(- bs + ba))
        q2 = .25 * (sp.special.expit(bs - ba) + sp.special.expit(- bs - ba))
        q1p = .25 * (sp.special.expit(bs + ba) - sp.special.expit(- bs + ba))
        q2p = .25 * (sp.special.expit(bs - ba) - sp.special.expit(- bs - ba))
        
        f1 = np.abs(self.r2_coef[4] + self.r2_coef[5] + self.r2_coef[6])
        f2 = np.abs(self.r2_coef[4] + self.r2_coef[5] - self.r2_coef[6])
        f3 = np.abs(self.r2_coef[4] - self.r2_coef[5] + self.r2_coef[6])
        f4 = np.abs(self.r2_coef[4] - self.r2_coef[5] - self.r2_coef[6])
        
        Q_coef[0,0] = self.r2_coef[0] + q1 * f1 + q2 * f2 + (.5 - q1) * f3 + (.5 - q2) * f4
        
        Q_coef[1,0] = self.r2_coef[1] + q1p * f1 + q2p * f3 - q1p * f3 - q2p * f4
        Q_coef[2,0] = self.r2_coef[2] + q1 * f1 - q2 * f2 + (.5 - q1) * f3 - (.5 - q2) * f4
        Q_coef[3,0] = self.r2_coef[3] + q1p * f1 - q2p * f3 - q1p * f3 + q2p * f4
        
        return Q_coef
        
        
    def prob(self, s1, a1, s2=1):
        bs, ba = self.pr_coef
        pr1 = sp.special.expit(bs*s1 + ba*a1)
        if s2 == 1: 
            return pr1
        elif s2 == -1:
            return 1-pr1
        else:
            raise ValueError("Two states 1 and -1 only!")
            
            
    def rew2(self, sasa):
        s1, a1, s2, a2 = sasa
        r = self.r2_coef
        rew = r[1] + r[2]*s1 + r[3]*a1 + r[4]*s1*a1 \
              + r[5]*a2 + r[6]*s2*a2 + r[7]*a1*a2 \
              + np.random.normal(0, self.noise_var, 1)
        return rew

    
    def sample_next_state(self, sa1):
        s1 = sa1[0]
        a1 = sa1[1]
        pr1 = self.prob(s1, a1, 1)
        s1 = 2 * np.random.binomial(1,pr1,1) - 1
        return s1
    
    
    def sample_action(self, s1):
        a1 = 2 * np.random.binomial(1,.5,1) - 1
        return a1

    
    def covariates(self, sa):
        s = sa[:2]
        a = sa[2:]
        return np.array([1, s[0], a[0], s[0]*a[0], a[1], s[1]*a[1], a[0]*a[1], s[1]])

    
    def covariates_stage_0(self, sa):
        s, a = sa
        return np.array([1, s, a, s*a, 0, 0, 0, 0])
 

    def max_reward(self, obs, betas, stage):
        num_obs, dim_obs = obs.shape

        sta = obs[:, [1,7]] # horizon 0 and 1 obs are the same

        act0 = obs[:, [2,4]]  # A1, A2 
        act1 = obs[:, [2,4]]  # A1, A2 

        act0[:,stage] = np.ones((num_obs)) * -1
        act1[:,stage] = np.ones((num_obs))

        X0 = np.apply_along_axis(self.covariates, 1, np.hstack([sta, act0]))
        X1 = np.apply_along_axis(self.covariates, 1, np.hstack([sta, act1]))

        rew0 = np.hstack([X0, obs[:,8:]]) @ betas[:, stage]
        rew1 = np.hstack([X1, obs[:,8:]]) @ betas[:, stage]

        act_opt = 2 * (rew1 - rew0 > 0).astype(int) - 1
        rew_opt = np.amax( np.hstack([rew0[:,None], rew1[:,None]]), axis=1 )

        # num_obs x 1
        return (act_opt, rew_opt)


    def sample_random_sta(self, num_obs):
        sta = np.zeros((num_obs, self.horizon))
        rew = np.zeros((num_obs, self.horizon))
        
        act = 2 * np.random.binomial(1, .5, (num_obs, self.horizon)) - 1
        sta[:,0] = 2 * np.random.binomial(1, .5, num_obs) - 1
        
        sta[:,1] = np.squeeze( np.apply_along_axis(self.sample_next_state, 1,
                                      np.hstack([sta[:,0,None], act[:,0,None]])) )
        
        X = np.apply_along_axis(self.covariates, 1, np.hstack([sta, act])) 
        
        assert X.shape[1] == self.dim_tru
        rew[:,-1] = X @ self.r2_coef + np.random.normal(0, self.noise_var, num_obs)
        
        return (X, act, rew)
    
    
    def sample_random_obs(self, num_obs, distn='n'):
        '''
            distn: distribution of other irrelevant covariates
        '''
        X, act, rew = self.sample_random_sta(num_obs)
        obs = np.zeros((num_obs, self.dim_obs, self.horizon))
        
        obs[:,:self.dim_tru,:] = X[:,:,None]
        
        if distn == 'n':
            obs[:,self.dim_tru:,:] = np.random.normal(1, 1, (num_obs, self.dim_obs-self.dim_tru, self.horizon))
        elif distn == 'u':
            obs[:,self.dim_tru:,:] = np.random.uniform(-1.5, 1.5, (num_obs, self.dim_obs-self.dim_tru, self.horizon))
        else:
            raise ValueError("MDP observation distribution " 
                             + distn + " NOT accpeted!")
        
        return {'obs':obs, 'act':act, 'rew':rew} 
        
        
    def reset(self, batch_size):
        
        self.done = 0
        self.stage = 0
        self.batch_size = batch_size
        
        self.reward = np.zeros((self.batch_size, self.horizon))
        self.observe = np.zeros((self.batch_size, self.dim_obs, self.horizon))
        self.action = np.zeros((self.batch_size, self.horizon))
        self.state = np.zeros((self.batch_size, self.horizon))
        
        # Random stage 0 with dummy actions
        self.state[:, 0] = 2 * np.random.binomial(1, .5, self.batch_size) - 1
        
        self.observe[:, :self.dim_tru, 0] = np.apply_along_axis(
            self.covariates_stage_0, 
            1, 
            np.hstack([self.state[:,0,None], self.action[:,0,None]])
        )
       
        if self.distn == 'n':
            self.observe[:,self.dim_tru:,0] = np.random.normal(1, 1, 
                                                    (self.batch_size, 
                                                     self.dim_obs-self.dim_tru))
        elif self.distn == 'u':
            self.observe[:,self.dim_tru:,0] = np.random.uniform(-1.5, 1.5, 
                                                     (self.batch_size, 
                                                      self.dim_obs-self.dim_tru))
        
        # Array of size (self.batch_size, self.dim_obs, self.horizon)
        return self.observe
    
    
    def reset_with_obs(self, batch_size, obs):
        
        self.done = 0
        self.stage = 0
        self.batch_size = batch_size
        
        self.reward = np.zeros((self.batch_size, self.horizon))
        self.observe = np.zeros((self.batch_size, self.dim_obs, self.horizon))
        self.action = np.zeros((self.batch_size, self.horizon))
        self.state = np.zeros((self.batch_size, self.horizon))
        
        self.observe = obs
        self.state[:, 0] = self.observe[:, 1, 0]
        
        # Array of size (self.batch_size, self.dim_obs)
        return self.observe
        
        
    def greedy(self, q_coef):
        assert q_coef.shape[1] == self.horizon
        
        action, _ = self.max_reward(self.observe[:,:,self.stage], q_coef, stage=self.stage)
        
        return action
    
    def step(self, action):
        
        h = self.stage
        
        self.action[:,h] = action
        
        X = np.apply_along_axis(self.covariates, 
                                1, 
                                np.hstack([self.state, self.action]))
        
        self.observe[:, :self.dim_tru, h] = X[:,:]
        
        if h + 1 == self.horizon:
            self.done = 1
            self.reward[:, h] = np.squeeze(
                np.apply_along_axis(self.rew2,
                                    1,
                                    np.hstack([self.state, self.action]))
            )
            return (None, self.reward[:, h], self.done)
        else:
            self.state[:, h+1] = np.squeeze(
                np.apply_along_axis(self.sample_next_state, 
                                    1, 
                                    np.hstack([self.state[:,h,None], 
                                               self.action[:,h,None]])) 
            )
            
            self.observe[:, :self.dim_tru, h+1] = np.apply_along_axis(
                self.covariates_stage_0, 
                1, 
                np.hstack([self.state[:,h+1,None], self.action[:,h+1,None]])
            )
            
            if self.distn == 'n':
                self.observe[:,self.dim_tru:,h+1] = np.random.normal(1, 1, 
                                                    (self.batch_size, 
                                                     self.dim_obs-self.dim_tru))
            elif self.distn == 'u':
                self.observe[:,self.dim_tru:,h+1] = np.random.uniform(-1.5, 1.5, 
                                                     (self.batch_size, 
                                                      self.dim_obs-self.dim_tru))
            self.stage += 1
            # 
            return (self.observe[:,:,self.stage], self.reward[:, h], self.done)
        
    def add_new_traj(self, old_traj=None):
        assert self.done == 1
        
        if old_traj is None:
            return {'obs':self.observe, 'act':self.action, 'rew':self.reward} 
        else:
            new_traj = {'obs':np.concatenate([old_traj['obs'], self.observe], 
                                             axis=0),
                        'act':np.concatenate([old_traj['act'], self.action], 
                                             axis=0),
                        'rew':np.concatenate([old_traj['rew'], self.reward], 
                                             axis=0)}
            
            return new_traj
        
      

    