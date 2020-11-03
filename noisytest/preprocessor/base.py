from abc import abstractmethod

import tensorflow as tf

from noisytest.tunable import HyperParameterMixin
from noisytest.preprocessor.data import InputTargetData


class Preprocessor(HyperParameterMixin):
    """Preprocessor for noise / target data"""

    def __init__(self, parent=None, **kwargs):
        self._parent = parent

        # Pass arguments down the chain
        if parent is None:
            self._arguments = kwargs
        else:
            self._arguments = parent.arguments

    def prepare_data(self, data):
        if self._parent is None:
            return data

        return self._parent.prepare_data(data)

    def prepare_input_target_data(self, data):
        if self._parent is None:
            return data
        return self._parent.prepare_input_target_data(data)

    @staticmethod
    def concat_input_target_data(a, b):
        return InputTargetData(
            tf.concat([a.input, b.input], 0),
            tf.concat([a.target, b.target], 0),
            a.no_of_time_samples + b.no_of_time_samples,
            a.no_of_annotated_blocks + b.no_of_annotated_blocks
        )

    @property
    def keywords_to_target_data(self):
        """Maps training data keywords to target data"""
        if self._parent is None:
            return None
        return self._parent.keywords_to_target_data

    @property
    def target_data_to_keywords(self):
        """Maps target data to training data keywords"""
        return dict((v, k) for k, v in self.keywords_to_target_data.items())

    def append_parent(self, new_parent):
        """Add a parent to the preprocessor chain"""
        obj = self

        while obj.parent:
            obj = obj.parent

        obj.parent = new_parent
        return self

    @property
    def arguments(self):
        """Returns arguments supplied to this instance or its parents as dict"""
        return self._arguments

    @property
    def parent(self):
        """Returns this instance's parent preprocessor"""
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent


class ImportPreprocessor(Preprocessor):
    """A preprocessor, which may be used for data import"""

    @abstractmethod
    def create_empty_input_target_data(self):
        pass


class NullPreprocessor(Preprocessor):
    """A preprocessor doing nothing (passing input data)"""

    def __init__(self):
        super().__init__(None)

    @property
    def hyper_parameters(self):
        return {}
