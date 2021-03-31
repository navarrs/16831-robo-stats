
import numpy as np


class Policy:
    """
    DO NOT MODIFY
    """

    def init(self, nbActions):
        self.nbActions = nbActions

    def name(self):
        pass
    
    def decision(self):
        pass

    def getReward(self, reward):
        pass


class policyRandom(Policy):
    """
    DO NOT MODIFY
    """
    def name(self):
        return "policyRandom"
        
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

    def name(self):
        return "policyConstant"
    
    def decision(self):
        return self.chosenAction

    def getReward(self, reward):
        pass


class policyGWM(Policy):
    def init(self, nbActions):
        self.nbActions = nbActions
        self.w = np.ones(shape=self.nbActions, dtype=np.float)
        self.t = 0
        
    def name(self):
        return "policyGWM"

    def decision(self):
        """ q3.1.1 """
        pn = self.w / np.sum(self.w)
        self.chosenAction = np.where(np.random.multinomial(1, pn) == 1)[0]
        return self.chosenAction

    def getReward(self, reward):
        """ q3.1.1 """
        loss = 1 - reward
        loss_v = np.zeros(shape=self.nbActions, dtype=np.float)
        loss_v[self.chosenAction] = loss
        
        self.t += 1
        eta = np.sqrt(np.log(self.nbActions) / self.t)
        self.w = self.w * np.exp(-eta * loss_v)

class policyEXP3(Policy):
    def init(self, nbActions):
        self.nbActions = nbActions
        self.w = np.ones(shape=self.nbActions, dtype=np.float)
        self.t = 0
        
    def name(self):
        return "policyEXP3"
        
    def decision(self):
        """ q3.3 """
        self.pn = self.w / np.sum(self.w)
        self.chosenAction = np.where(np.random.multinomial(1, self.pn) == 1)[0]
        return self.chosenAction

    def getReward(self, reward):
        """ q3.3 """
        loss = 1 - reward
        loss_v = np.zeros(shape=self.nbActions, dtype=np.float)
        loss_v[self.chosenAction] = loss / self.pn[self.chosenAction]
        
        self.t += 1
        eta = np.sqrt(np.log(self.nbActions) / (self.t * self.nbActions))
        self.w = self.w * np.exp(-eta * loss_v)


class policyUCB(Policy):
    def init(self, nbActions):
        self.nbActions = nbActions
        self.alpha = 1.0
        
        self.t = 1
        self.S = np.zeros(shape=self.nbActions, dtype=np.float)
        self.C = np.ones(shape=self.nbActions, dtype=np.float)
        self.confidence = []
        
    def name(self):
        return "policyUCB"
        
    def firstPull(self, g0):
        self.S = g0
        
    def decision(self):
        v = self.S / self.C + np.sqrt(self.alpha*np.log(self.t)/(2*self.C))
        self.confidence.append(v)
        
        self.chosenAction = np.argmax(v)
        return self.chosenAction

    def getReward(self, reward):
        self.S[self.chosenAction] += reward
        self.C[self.chosenAction] += 1
        self.t += 1
        
    def getConfidence(self):
        return np.asarray(self.confidence)

class policyUCBContext(Policy):
    def init(self, nbActions, nActionRange):
        self.nbActions = nbActions
        self.nActionRange = nActionRange
        self.alpha = 1.0
        self.t = 1
        self.S = np.zeros(shape=self.nbActions, dtype=np.float)
        self.C = np.ones(shape=self.nbActions, dtype=np.float)
        self.confidence = []
        
    def name(self):
        return "policyUCBContext"
        
    def firstPull(self, g0):
        self.S = g0
        
    def decision(self, state):
        n = self.nActionRange[int(state)]
        
        v = self.S / self.C + np.sqrt(self.alpha*np.log(self.t)/(2*self.C))
        self.confidence.append(v)
        
        self.chosenAction = np.argmax(v[n[0]:n[1]])
        return self.chosenAction

    def getReward(self, reward):
        self.S[self.chosenAction] += reward
        self.C[self.chosenAction] += 1
        self.t += 1
        
    def getConfidence(self):
        return np.asarray(self.confidence)
