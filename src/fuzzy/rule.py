"""
Fuzzy Rule
"""
from collections import Counter
from functools import lru_cache
import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

class Rule:
    def __init__(self, index_x: tuple, counter=None) -> None:
        self.index_x = index_x
        if counter is None:
            self._counter = {}
        else:
            self._counter = counter
    
    def __add__(self, other: 'Rule') -> 'Rule':
        if self.index_x != other.index_x:
            raise ValueError("IF parts of Rule has to match to use `+`")
        return Rule(
            self.index_x,
            dict(Counter(self._counter) + Counter(other._counter)),
        )
    
    def __repr__(self) -> str:
        return f"IF {self.index_x} THEN {self.index_y}"
    
    def increase_counter(self, output_set_index: int, amount: int=1):
        """ Increases counter by amount """
        if output_set_index in self._counter:
            self._counter[output_set_index] += amount
        else:
            self._counter[output_set_index] = amount
    
    @property
    @lru_cache()
    def index_y(self):
        return max(self._counter, key=lambda key: self._counter[key])

    @property
    def counter(self) -> dict:
        return self._counter
