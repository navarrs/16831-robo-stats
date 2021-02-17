from world import (
    Prediction,
    StochasticWorld,
    OddLoseExpert,
    OptimisticExpert,
    NegativeExpert
)
import numpy as np
np.set_printoptions(4)


class WeightedMajorityAlgorithm(object):

    def __init__(self, H, world, T: int = 10, eta: float = 0.5):
        self._H = H
        self._num_experts = len(H)
        self._T = T
        self._eta = eta
        self._world = world
        self._regret = 0.0
        self._learner_loss = 0.0
        self._experts_loss = np.zeros(
            shape=(self._num_experts), dtype=np.float32)
        self.build()
        print("Initialized WMA")

    def build(self) -> None:
        self._weights = np.ones(shape=(self._num_experts), dtype=np.float32)
        self._x = np.zeros(shape=(self._num_experts), dtype=np.int)

    def receive_advice(self, num_step: int) -> None:
        for n in range(self._num_experts):
            self._x[n] = self._H[n].give_advice(num_step)

    def receive_label(self) -> int:
        return self._world.give_label()

    def pred_function(self, value: float) -> int:
        if value >= 0.0:
            return Prediction.WIN.value
        else:
            return Prediction.LOSE.value

    def compute_regret(self, t):

        self._learner_loss += float(self._y_pred == self._y_true)
        for n in range(self._num_experts):
            self._experts_loss[n] += (self._x[n] == self._y_true).astype(float)
        self._regret = self._learner_loss - np.amin(self._experts_loss)
        if t > 0:
            self._regret /= t

    def run(self):
        print(f"\tInitial weights: {self._weights}")
        print(f"\tWorld: {self._world.get_name()}")
        for t in range(0, self._T):
            self.receive_advice(t)
            value = np.dot(self._x, self._weights)
            self._y_pred = self.pred_function(value)
            self._y_true = self.receive_label()
            incorrect_advice = (self._y_true != self._x).astype(int)
            self._weights = self._weights * (1 - self._eta * incorrect_advice)
            self.compute_regret(t+1)

            print("\t[{}/{}]\n\t\tadvice: {}, y_pred: {}, y_true: {}\n\t\tweights: {}".format(
                t, self._T, self._x, self._y_pred, self._y_true, self._weights))
            print("\t\tregret: {:.4f}, learner_loss {}, experts_loss: {}".format(
                self._regret, self._learner_loss, self._experts_loss))
