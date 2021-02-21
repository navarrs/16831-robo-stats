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
from typing import List, Tuple


class Prediction(IntEnum):
    LOSE = -1
    WIN = 1


class Weather(IntEnum):
    SUNNY = 0
    RAINY = 1


class Game(IntEnum):
    HOME = 0
    AWAY = 1

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

    def give_advice(self, num_game: int, observations: dict = None) -> int:
        """ Base method to implement the advice rule.
        Args:
            num_game: number of current game played.
            observations: world observations; not all experts can use it.
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

    def give_advice(self, num_game: int, observations: dict = None) -> int:
        return self._predictions.WIN.value


class NegativeExpert(Expert):
    """ Negative expert: it always predicts LOSE """

    def give_advice(self, num_game: int, observations: dict = None) -> int:
        return self._predictions.LOSE.value


class OddLoseExpert(Expert):
    """ Odd Lose expert: predicts LOSE every odd game."""

    def give_advice(self, num_game: int, observations: dict = None) -> int:
        if num_game % 2 == 0:
            return Prediction.WIN.value
        return Prediction.LOSE.value


class WeatherExpert(Expert):
    """ Weather expert: predicts WIN when the weather is sunny. """

    def give_advice(self, num_game: int, observations: dict = None) -> int:
        if "weather" in observations:
            weather = observations["weather"]
            if weather == Weather.SUNNY:
                return Prediction.WIN.value
        return Prediction.LOSE.value


class GameExpert(Expert):
    """ Game expert: predicts WIN if home game. """

    def give_advice(self, num_game: int, observations: dict = None) -> int:
        if "game" in observations:
            game = observations["game"]
            if game == Game.HOME:
                return Prediction.WIN.value
        return Prediction.LOSE.value


class WinStreakExpert(Expert):
    """ Win Streak expert: predicts WIN based on consecutive wins. """

    def give_advice(self, num_game: int, observations: dict = None) -> int:
        if "win_streak" in observations:
            win_streaks = observations["win_streak"]
            if win_streaks > 2:
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
        self._weather_list = list(map(int, Weather))
        self._game_list = list(map(int, Game))
        self._win_streaks = 0
        self._label = -1
        self._observations = {
            "weather": 0, "game": 0, "win_streak": 0
        }

    def generate_observations(self) -> None:
        """ Generates observations for the current state. """
        self._observations["weather"] = np.random.choice(self._weather_list)
        self._observations["game"] = np.random.choice(self._game_list)

    def step(self,
             expert_advice: np.array = None,
             expert_weights: np.array = None
             ) -> Tuple[int, dict]:
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

    def step(self,
             expert_advice: np.array = None,
             expert_weights: np.array = None
             ) -> Tuple[int, dict]:
        self.generate_observations()
        self._label = random.choice(self._labels)
        if self._label == Prediction.LOSE.value:
            self._win_streaks = 0
        else:
            self._win_streaks += 1
        self._observations["win_streak"] = self._win_streaks
        return self._label, self._observations


class DeterministicWorld(World):
    """ Deterministic world: always produces a consistent label. It has access
        to the expert advice and which can be used to implement a determinist
        rule to provide labels. 
    """

    def step(self,
             expert_advice: np.array = None,
             expert_weights: np.array = None
             ) -> Tuple[int, dict]:
        """ Uses a deterministic rule to produce a label. """
        self.generate_observations()
        self._label = self.rule()
        # self._label = self.rule_counter()

        if self._label == Prediction.LOSE.value:
            self._win_streaks = 0
        else:
            self._win_streaks += 1

        self._observations["win_streak"] = self._win_streaks
        return self._label, self._observations

    def rule(self) -> int:
        """ Produces label based on observations """
        weather = self._observations["weather"]
        game = self._observations["game"]
        win_streak = self._observations["win_streak"]
        # Since this world is deterministic, rules are simple if-elses
        if weather == Weather.RAINY.value:
            if win_streak > 2 and game == Game.HOME.value:
                return Prediction.WIN.value
            return Prediction.LOSE.value
        else:
            if win_streak < 2 and game == Game.AWAY.value:
                return Prediction.LOSE
        return Prediction.WIN.value

    def rule_counter(self) -> int:
        """ Used in 3.2, 3.3 and 3.4 """
        label = -1 if self._counter % 4 == 0 else 1
        self._counter += 1
        return label

    def set_counter(self):
        self._counter = 0


class AdversarialWorld(World):
    """ Adversarial world: based on expert advice and learner strategy it is 
        able to adjust its label. 
    """

    def step(self,
             expert_advice: np.array = None,
             expert_weights: np.array = None
        ) -> Tuple[int, dict]:
        """ 
        Adversarial predicts a label based on the expert advice, expert weights
        and prediction rule. For WMA it produces the opposite label to that of 
        the majority of the experts. For RWMA it produces a label based on the 
        current weights of the experts. 
        """
        if self._strategy == "wma":
            # values, counts = np.unique(expert_advice, return_counts=True)
            # index = np.argmax(counts)
            # return -1 if values[index] == 1 else 1
            self._label = -1 if np.dot(expert_advice,
                                       expert_weights) > 0 else 1
        elif self._strategy == "rwma":
            # w = counts / np.sum(counts)
            # choice = np.random.multinomial(1, w)
            # w = expert_weights / np.sum(expert_weights)
            # index = np.argmin(w)
            # return expert_advice[index]
            self._label = -1 if np.dot(expert_advice,
                                       expert_weights) > 0 else 1
        observations = self.generate_observations()
        return self._label, observations

    # Member setters
    def set_strategy(self, strategy: str):
        self._strategy = strategy
