# ------------------------------------------------------------------------------
# @file: rwma.py
# @date: Feb 17, 2021
# @author: ingridn
#
# @brief: Implementation of the Randomized Weighted Majority Algorithm
# ------------------------------------------------------------------------------
import numpy as np

from environment import (
    Prediction,
    StochasticWorld,
    OddLoseExpert,
    OptimisticExpert,
    NegativeExpert
)
from wma import WeightedMajorityAlgorithm

np.set_printoptions(4)


class RandomizedWeightedMajorityAlgorithm(WeightedMajorityAlgorithm):
    """ Since the class is similar to WMA it inherits from the class to avoid 
    repeating code.
    """
    def pred_function(self) -> int:
        """ Prediction rule. In the case of the Randomized Weighted Majority 
        Algorithm it randomly predicts based on a multinomial distribution over
        the weights at the current time step. 
        """
        w = self._weights / np.sum(self._weights)
        expert = np.random.multinomial(1, w)
        return self._x[np.where(expert == 1)]
        
    def run(self):
        """ Runs the algorithm for T time steps. """
        print(f"\tinitial weights: {self._weights}")
        print(f"\tworld: {self._world.get_name()}")
        for t in range(0, self._T):        
            obs = self.receive_observations()
            self.receive_advice(t, obs)
            self._y_pred = self.pred_function()
            self._y_true = self.receive_label()
            incorrect_advice = (self._y_true != self._x).astype(int)
            self._weights = self._weights * (1 - self._eta * incorrect_advice)
            self.compute_regret(t)

            if t % self._ckpt_step == 0:
                print("\t[{}/{}]\n\t\tadvice: {}, y_pred: {}, y_true: {}".format(
                    t, self._T, self._x, self._y_pred, self._y_true))
                print(f"\t\tweights: {self._weights}")
                print(f"\t\tlearner_loss: {self._learner_loss[t]}")
                print(f"\t\texperts_loss: {self._experts_loss[t]}")
                print(f"\t\tregret: {self._regret[t]}")
