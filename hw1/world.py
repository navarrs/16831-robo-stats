
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
    def __init__(self, name: str, labels: List[int]):
        self._name = name
        self._labels = labels
    
    def get_name(self):
        return self._name
    
    def give_label(self) -> int:
        return NotImplementedError

class StochasticWorld(World):        
    def give_label(self) -> int:
        return random.choice(self._labels)
        
    