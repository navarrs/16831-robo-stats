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
            regret[t] = self.cumulativeRewardBestActionHindsight(t) - sum(reward)
            policy.getReward(reward[t])
            self.N += 1
            
        return reward, action, regret

    def reward(self, a):
        return self.tabR[a, self.N]

    def resetGame(self):
        self.N = 0

    def cumulativeRewardBestActionHindsight(self, t):
        # cumulative_reward = np.sum(self.tabR[:, :t], axis=1)
        cumulative_reward = np.sum(self.tabR, axis=1)
        i = np.argmax(cumulative_reward)
        return cumulative_reward[i]


class gameConstant(Game):
    """
    DO NOT MODIFY
    """

    def __init__(self):
        super().__init__()
        self.nbActions = 2
        self.totalRounds = 1000
        self.tabR = np.ones((2, 1000))
        self.tabR[0] *= 0.8
        self.tabR[1] *= 0.2
        self.N = 0


class gameGaussian(Game):
    def __init__(self, nbActions, totalRound):
        super().__init__()
        self.nbActions = nbActions
        self.totalRounds = totalRound
        self.N = 0


class gameAdverserial(Game):
    def __init__(self):
        super().__init__()
        self.nbActions = 2
        self.totalRounds = 1000
        self.N = 0


class gameLookupTable(Game):
    def __init__(self, tabInput, isLoss):
        super().__init__()
        self.N = 0
