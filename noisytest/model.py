from dataclasses import dataclass
from typing import Any

import tensorflow as tf
import numpy as np
from sklearn import svm
from noisytest.tunable import HyperParameterMixin
from noisytest.tunable import HyperParameterRange


@dataclass
class ModelError:
    """Holds data describing classification error of a model"""
    subset_accuracy: Any
    class_false_negatives: Any
    class_false_positives: Any

    @property
    def fitness(self):
        """Fitness value used for hyper parameter search"""
        return self.subset_accuracy


class Model(HyperParameterMixin):
    """Describes a noisytest model"""

    def __init__(self, regularization=1.1, kernel='rbf', kernel_gamma=0.00057):
        self._svm = svm.SVC(C=regularization, kernel=kernel, gamma=kernel_gamma)

    def train(self, input_target_data):
        """Train the model using specified input/output data set"""

        weights = float(tf.size(input_target_data.target)) / (np.max(input_target_data.target.numpy()) + 1) / \
                  np.bincount((input_target_data.target.numpy() + 0.1).astype(int))
        self._svm.class_weight = {0: weights[0], 1: weights[1], 2: weights[2], 3: weights[3]}

        self._svm.fit(input_target_data.input, input_target_data.target)
        return self._svm.score(input_target_data.input, input_target_data.target)

    def validate(self, input_target_data):
        """Validate the model using specified input/output data set. Returns a ModelError instance"""

        subset_accuracy = self._svm.score(input_target_data.input, input_target_data.target)
        false_neg, false_pos = self._calc_error_per_class(self._svm.predict(input_target_data.input),
                                                          input_target_data.target)

        return ModelError(subset_accuracy, false_neg, false_pos)

    def predict(self, data):
        """Predict output data based on given input data"""

        return self._svm.predict(data)

    @staticmethod
    def _calc_error_per_class(predicted, truth):
        """Calculates the error for every class using predicted and ground-truth data"""

        false_positives = {}
        false_negatives = {}

        for (prediction, expectation) in zip(predicted, truth.numpy()):
            false_positives[round(prediction)] = false_positives.get(round(prediction), 0.0)
            false_positives[round(expectation)] = false_positives.get(round(expectation), 0.0)
            false_negatives[round(expectation)] = false_negatives.get(round(expectation), 0.0)

            if prediction != expectation:
                false_positives[round(prediction)] += 1.0 / len(predicted)
                false_negatives[round(expectation)] += 1.0 / len(predicted)

        return false_negatives, false_positives

    @property
    def svm(self):
        """Returns the instance of the underlying scikit learn SVM"""
        return self._svm

    @property
    def kernel(self):
        return self._svm.kernel

    @property
    def regularization(self):
        return self._svm.C

    @regularization.setter
    def regularization(self, value):
        self._svm.C = value

    @property
    def kernel_parameter(self):
        return self._svm.gamma

    @kernel_parameter.setter
    def kernel_parameter(self, value):
        self._svm.gamma = value

    @property
    def hyper_parameters(self):
        base = {'regularization': HyperParameterRange(0.5, 0.1, 2.5)}

        if self.kernel == 'rbf':
            base['kernel_parameter'] = HyperParameterRange(4.5e-04, 1.0e-05, 6.0e-04)

        return base
