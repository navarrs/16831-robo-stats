
import numpy as np 
import random

from enum import Enum
from typing import List

class Prediction(Enum):
    LOSE = -1
    WIN = 1
    ODD_LOSE = 2

class Expert():
    def __init__(self, name: str):
        self._name = name
    
    def get_name(self) -> str:
        return self._name
    
    def give_advice(self, num_game: int) -> int:
        return NotImplementedError
    
class OptimisticExpert(Expert):
    def give_advice(self, num_game: int) -> int:
        return Prediction.WIN.value
    
class NegativeExpert(Expert):
    def give_advice(self, num_game: int) -> int:
        return Prediction.LOSE.value
    
class OddLoseExpert(Expert):     
    def give_advice(self, num_game: int) -> int:
        if num_game % 2 == 0:
            return Prediction.WIN.value
        else:
            return Prediction.LOSE.value


class World():
    def __init__(self, name: str, labels: List[int] = [-1, 1], strategy: str = None):
        self._name = name
        self._labels = labels
        self._strategy = strategy
    
    def get_name(self):
        return self._name
    
    def give_label(self, expert_advice: np.array = None) -> int:
        return NotImplementedError

class StochasticWorld(World):        
    def give_label(self, expert_advice: np.array = None) -> int:
        return random.choice(self._labels)
    
class DeterministicWorld(World):
    def give_label(self, expert_advice: np.array = None) -> int:
        return self.rule_opposite(expert_advice)
        
    def rule_opposite(self, expert_advice):
        return -1 if expert_advice[-1] == 1 else -1
    
    def rule_max(self):
        values, counts = np.unique(expert_advice, return_counts=True)
        index = np.argmax(counts)
        return values[index]
    
    def rule_min(self):
        values, counts = np.unique(expert_advice, return_counts=True)
        index = np.argmin(counts)
        return values[index]

class AdversarialWorld(World):
    def give_label(self, expert_advice: np.array = None) -> int:
        values, counts = np.unique(expert_advice, return_counts=True)
        if self._strategy == "wma":
            index = np.argmax(counts)
            return -1 if values[index] == 1 else 1
        elif self._strategy == "rwma":
            w = counts / np.sum(counts)
            choice = np.random.multinomial(1, w)
            index = np.where(choice == 1)[0][0]
            return self._labels[index]
            
        
    