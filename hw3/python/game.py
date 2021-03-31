import numpy as np


class Game:
    nbActions = 0
    totalRounds = 0
    N = 0
    tabR = np.array([])

    def __init__(self):
        return

    def play(self, policy):
        policy.init(self.nbActions)
        reward = np.zeros(self.totalRounds)
        action = np.zeros(self.totalRounds, dtype=np.int)
        regret = np.zeros(self.totalRounds)

        for t in range(self.totalRounds):
            action[t] = policy.decision()
            reward[t] = self.reward(action[t])
            regret[t] = self.cumulativeRewardBestActionHindsight() - sum(reward)
            policy.getReward(reward[t])
            self.N += 1
            
        return reward, action, regret

    def reward(self, a):
        return self.tabR[a, self.N]

    def resetGame(self):
        self.N = 0

    def cumulativeRewardBestActionHindsight(self):
        """ q2.4 """
        cumulative_reward = np.sum(self.tabR[:, :self.N], axis=1)
        i = np.argmax(cumulative_reward)
        return cumulative_reward[i]


class gameConstant(Game):
    """
    DO NOT MODIFY
    """

    def __init__(self, totalRound=1000):
        super().__init__()
        self.nbActions = 2
        self.totalRounds = totalRound
        self.tabR = np.ones((self.nbActions, self.totalRounds))
        self.tabR[0] *= 0.8
        self.tabR[1] *= 0.2
        self.N = 0


class gameGaussian(Game):
    def __init__(self, nbActions, totalRound):
        super().__init__()
        """ q3.4.1 """
        self.nbActions = nbActions
        self.totalRounds = totalRound
        self.N = 0
        
        self.mu = np.random.uniform(low=0.0, high=1.0, size=self.nbActions)
        self.sigma = np.random.uniform(low=0.0, high=1.0, size=self.nbActions)
        
        self.tabR = np.ones((self.nbActions, self.totalRounds))    
        for i in range(self.totalRounds):
            g = np.random.normal(self.mu, self.sigma, size=self.nbActions)
            g = np.clip(g, 0.0, 1.0)
            self.tabR[:, i] = g


class gameAdverserial(Game):
    def __init__(self):
        super().__init__()
        self.nbActions = 2
        self.totalRounds = 1000
        self.N = 0
        
        self.tabR = np.ones((self.nbActions, self.totalRounds))
        self.tabR[0] *= 0.8
        self.tabR[1] *= 0.2
        
        # UCB parameters
        self.S = np.zeros(shape=self.nbActions, dtype=np.float)
        self.C = np.ones(shape=self.nbActions, dtype=np.float)
        
        # EXP3 parameters
        self.w = np.ones(shape=self.nbActions, dtype=np.float)
    
    def play(self, policy):
        policy.init(self.nbActions)
        reward = np.zeros(self.totalRounds)
        action = np.zeros(self.totalRounds, dtype=np.int)
        regret = np.zeros(self.totalRounds)

        for t in range(self.totalRounds):
            action[t] = policy.decision()
            
            r = self.reward(action[t])
            if "UCB" in policy.name():
                self.S[action] += r
                self.C[action] += 1
                v = self.S / self.C + np.sqrt(np.log(t+1)/(2*self.C))
                r = (1.0 - np.clip(v[np.argmax(v)], 0, 1)) * r
            else:
                l = 1 - r
                if l < r:
                    r = l 
            
            reward[t] = r
            regret[t] = self.cumulativeRewardBestActionHindsight() - sum(reward)
            policy.getReward(reward[t])
            self.N += 1
            
        return reward, action, regret
    
    def reward(self, action):
        reward = self.tabR[action, self.N]
        return reward
        
class gameLookupTable(Game):
    def __init__(self, tabInput, isLoss, states=None, nranges=None):
        super().__init__()
        self.N = 0
        
        if isLoss:
            self.tabR = 1.0 - tabInput
        else:
            self.tabR = tabInput
        
        self.states = states
        self.nranges = nranges
        self.nbActions, self.totalRounds = tabInput.shape
    
    def play(self, policy):
        if "Context" in policy.name():
            policy.init(self.nbActions, self.nranges)
        else:
            policy.init(self.nbActions)
            
        reward = np.zeros(self.totalRounds)
        action = np.zeros(self.totalRounds, dtype=np.int)
        regret = np.zeros(self.totalRounds)
            
        for t in range(self.totalRounds):
            if "Context" in policy.name():
                action[t] = policy.decision(self.states[t])
            else:
                action[t] = policy.decision()
            reward[t] = self.reward(action[t])
            regret[t] = self.cumulativeRewardBestActionHindsight() - sum(reward)
            policy.getReward(reward[t])
            self.N += 1
            
