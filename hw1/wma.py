# ------------------------------------------------------------------------------
# @file: wma.py
# @date: Feb 17, 2021
# @author: Ingrid Navarro
#
# @brief: Implementation of the Weighted Majority Algorithm
# ------------------------------------------------------------------------------
import numpy as np

from environment import (
    Prediction,
    StochasticWorld,
    OddLoseExpert,
    OptimisticExpert,
    NegativeExpert
)
np.set_printoptions(4)


class WeightedMajorityAlgorithm(object):

    def __init__(self, H, world, T: int = 10, eta: float = 0.5):
        """ Initializes the WMA.
        Args
            H: hypothesis class
            world: environment that provides the true labels
            T: time steps to run the algorithm
            eta: penalty parameter
        """
        self._H = H
        self._num_experts = len(H)
        self._T = T
        self._eta = eta
        self._world = world
        self.build()

    def build(self) -> None:
        """ Initializes variables """
        self._weights = np.ones(shape=(self._num_experts), dtype=np.float32)
        self._x = np.zeros(shape=(self._num_experts), dtype=np.int)
        self._regret = np.zeros(shape=(self._T, 1), dtype=np.float32)
        self._learner_loss = np.zeros(shape=(self._T, 1), dtype=np.float32)
        self._experts_loss = np.zeros(
            shape=(self._T, self._num_experts), dtype=np.float32)

    def receive_advice(self, num_step: int) -> None:
        """ Receives the advice from all the experts
        Args
            num_step: current time step.
        """
        for n in range(self._num_experts):
            self._x[n] = self._H[n].give_advice(num_step)

    def receive_label(self) -> int:
        """ Receives true label from the world. """
        # note: if using stochastic, neither the expert advice nor the weights
        # are used by the world.
        return self._world.give_label(self._x, self._weights)

    def pred_function(self) -> int:
        """ Prediction rule. In the case of the Weighted Majority algorithm it 
        is based on the sign(x dot weights). 
        """
        value = np.dot(self._x, self._weights)
        if value >= 0.0:
            return Prediction.WIN.value
        return Prediction.LOSE.value

    def compute_regret(self, t: int) -> None:
        """
        Computes the cummulative learner loss, the experts loss and the regret. 
        t: current time step
        """
        self._learner_loss[t:] += float(not self._y_pred == self._y_true)
        for n in range(self._num_experts):
            self._experts_loss[t:, n] += float(not self._x[n] == self._y_true)
        # self._experts_loss[t:] += np.where(not self._x == self._y_true, self._x, self._y_true)

        self._regret[t] = (self._learner_loss[t] -
                           np.amin(self._experts_loss[t]))/(t+1)

    def run(self):
        """ Runs the algorithm for T time steps. """
        print(f"\tInitial weights: {self._weights}")
        print(f"\tWorld: {self._world.get_name()}")
        for t in range(0, self._T):
            self.receive_advice(t)
            self._y_pred = self.pred_function()
            self._y_true = self.receive_label()
            incorrect_advice = (self._y_true != self._x).astype(int)
            self._weights = self._weights * (1 - self._eta * incorrect_advice)
            self.compute_regret(t)
                        
            if t % 5 == 0:
                print("\t[{}/{}]\n\t\tadvice: {}, y_pred: {}, y_true: {}".format(
                    t, self._T, self._x, self._y_pred, self._y_true))
                print(f"\t\tincorrect advice: {incorrect_advice}")
                print(f"\t\tweights: {self._weights}")
                print(f"\t\tlearner_loss: {self._learner_loss[t]}")
                print(f"\t\texperts_loss: {self._experts_loss[t]}")
                print(f"\t\tregret: {self._regret[t]}")

    # Member getters
    def get_learners_loss(self) -> np.array:
        return self._learner_loss

    def get_experts_loss(self) -> np.array:
        return self._experts_loss

    def get_regret(self) -> np.array:
        return self._regret
