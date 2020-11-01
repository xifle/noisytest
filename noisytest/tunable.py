from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class HyperParameterRange:
    """Holds a parameter range. Iterable"""
    start: Any
    step: Any
    stop: Any

    def __iter__(self):
        return HyperParameterRangeIterator(self)


class HyperParameterRangeIterator:
    """Iterator for hyper parameter ranges"""

    def __init__(self, hyper_parameter_range):
        self._hyper_parameter_range = hyper_parameter_range
        self._value = hyper_parameter_range.start

    def __next__(self):
        result = self._value
        self._value += self._hyper_parameter_range.step
        if result <= self._hyper_parameter_range.stop:
            return result
        raise StopIteration


class HyperParameterMixin(ABC):
    """Mixin for the hyper-parameter interface of the noisytest optimizer

       A class with this mixin can present hyper parameters and their ranges to an
       optimizer.
    """

    @property
    @abstractmethod
    def hyper_parameters(self):
        """Returns a dictionary of attribute_name, HyperParameterRange pairs or an empty dictionary"""
        pass
