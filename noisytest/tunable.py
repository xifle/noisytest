from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class HyperParameterRange:
    start: Any
    step: Any
    stop: Any

    def __iter__(self):
        return HyperParameterRangeIterator(self)


class HyperParameterRangeIterator:

    def __init__(self, hyper_parameter_range):
        self._hyper_parameter_range = hyper_parameter_range
        self._value = hyper_parameter_range.start

    def __next__(self):
        result = self._value
        self._value += self._hyper_parameter_range.step
        if result <= self._hyper_parameter_range.stop:
            return result
        raise StopIteration


class HyperParameterMixin:

    @property
    @abstractmethod
    def hyper_parameters(self):
        pass
