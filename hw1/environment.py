# ------------------------------------------------------------------------------
# @file:    environment.py
# @date:    Feb 17, 2021
# @author:  Ingrid Navarro
#
# @brief:   Implementation of the world and experts used for Prediction with
#           Expert Advice (PWEA)
# ------------------------------------------------------------------------------
import numpy as np
import random

from enum import IntEnum
from typing import List


class Prediction(IntEnum):
    LOSE = -1
    WIN = 1

# ------------------------------------------------------------------------------
# Expert implementation


class Expert():
    """ Base class to implement experts rules. """

    def __init__(self, name: str, predictions: IntEnum = Prediction):
        """ Initializes the expert.
        Args:
            name: name of the expert.
        """
        self._name = name
        self._predictions = predictions
        self._prediction_list = list(map(int, predictions))

    def give_advice(self, num_game: int) -> int:
        """ Base method to implement the advice rule.
        Args:
            num_game: number of current game played.
        Return:
            advice: output generated by the expert.
        """
        return NotImplementedError

    # Member getters
    def get_name(self) -> str:
        return self._name

    def get_predictions(self) -> List[int]:
        return self._predictions


class OptimisticExpert(Expert):
    """ Optimistic expert: it always predicts WIN. """

    def give_advice(self, num_game: int) -> int:
        return self._predictions.WIN.value


class NegativeExpert(Expert):
    """ Negative expert: it always predicts LOSE """

    def give_advice(self, num_game: int) -> int:
        return self._predictions.LOSE.value


class OddLoseExpert(Expert):
    """ Odd Lose expert: predicts LOSE every odd game."""

    def give_advice(self, num_game: int) -> int:
        if num_game % 2 == 0:
            return Prediction.WIN.value
        return Prediction.LOSE.value

# ------------------------------------------------------------------------------
# World implementation


class World():
    """ Base class to implement world. """

    def __init__(self, name: str, labels: List[int] = [-1, 1]):
        """ Initializes the world.
        Args:
            name: name of the world.
            labels: labels that the world can produce.
        """
        self._name = name
        self._labels = labels

    def give_label(self, 
                   expert_advice: np.array = None, 
                   expert_weights: np.array = None
        ) -> int:
        """ Implements the world system to provide a label. 
        Args
        ----
            expert_advice: advice of the expert
        """
        return NotImplementedError

    # Member getters
    def get_name(self):
        return self._name


class StochasticWorld(World):
    """ Stochastic world: always chooses a random label. It does not use expert
        advice. 
    """

    def give_label(self, 
                   expert_advice: np.array = None, 
                   expert_weights: np.array = None) -> int:
        return random.choice(self._labels)


class DeterministicWorld(World):
    """ Deterministic world: always produces a consistent label. It has access
        to the expert advice and which can be used to implement a determinist
        rule to provide labels. 
    """

    def give_label(self, 
                   expert_advice: np.array = None, 
                   expert_weights: np.array = None
        ) -> int:
        """ Uses a deterministic rule to produce a label. """
        return self.rule_counter(expert_advice)
    
    def rule_counter(self, expert_advice):
        label = -1 if self._counter % 4 == 0 else 1
        self._counter += 1
        return label
    
    def rule_opposite(self, expert_advice):
        """ Produces opposite label to that produced by the OddLose expert. """
        return -1 if expert_advice[-1] == 1 else 1
    
    def set_counter(self):
        self._counter = 0


class AdversarialWorld(World):
    """ Adversarial world: based on expert advice and learner strategy it is 
        able to adjust its label. 
    """

    def give_label(self, 
                   expert_advice: np.array = None, 
                   expert_weights: np.array = None
        ) -> int:
        """ 
        Adversarial predicts a label based on the expert advice, expert weights
        and prediction rule. For WMA it produces the opposite label to that of 
        the majority of the experts. For RWMA it produces a label based on the 
        current weights of the experts. 
        """
        values, counts = np.unique(expert_advice, return_counts=True)
        if self._strategy == "wma":
            index = np.argmax(counts)
            return -1 if values[index] == 1 else 1
        elif self._strategy == "rwma":
            # w = counts / np.sum(counts)
            # choice = np.random.multinomial(1, w)
            w = expert_weights / np.sum(expert_weights)
            index = np.argmin(w)
            return expert_advice[index]

    # Member setters
    def set_strategy(self, strategy: str):
        self._strategy = strategy
