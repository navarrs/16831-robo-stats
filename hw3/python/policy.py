
import numpy as np


class Policy:
    """
    DO NOT MODIFY
    """

    def init(self, nbActions):
        self.nbActions = nbActions

    def decision(self):
        pass

    def getReward(self, reward):
        pass


class policyRandom(Policy):
    """
    DO NOT MODIFY
    """

    def decision(self):
        return np.random.randint(0, self.nbActions, dtype=np.int)

    def getReward(self, reward):
        pass


class policyConstant(Policy):
    """
    DO NOT MODIFY
    """

    def init(self, nbActions):
        self.chosenAction = np.random.randint(0, nbActions, dtype=np.int)

    def decision(self):
        return self.chosenAction

    def getReward(self, reward):
        pass


class policyGWM(Policy):
    def init(self, nbActions):
        self.nbActions = nbActions
        self.w = np.ones(shape=self.nbActions, dtype=np.float)
        self.t = 0

    def decision(self):
        pn = self.w / np.sum(self.w)
        self.chosenAction = np.where(np.random.multinomial(1, pn) == 1)[0]
        return self.chosenAction

    def getReward(self, reward):
        loss = 1 - reward
        loss_v = np.zeros(shape=self.nbActions, dtype=np.float)
        loss_v[self.chosenAction] = loss
        
        self.t += 1
        eta = np.sqrt(np.log(self.nbActions) / self.t)
        self.w = self.w * np.exp(-eta * loss_v)

class policyEXP3(Policy):
    def init(self, nbActions):
        self.nbActions = nbActions

    def decision(self):
        return 0

    def getReward(self, reward):
        pass


class policyUCB(Policy):
    def init(self, nbActions):
        self.nbActions = nbActions

    def decision(self):
        return 0

    def getReward(self, reward):
        pass
