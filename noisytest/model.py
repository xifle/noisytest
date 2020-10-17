from dataclasses import dataclass
from typing import Any
import noisytest.tunable

import tensorflow as tf
import numpy as np
from sklearn import svm


@dataclass
class ModelError:
    subset_accuracy: Any
    class_false_negatives: Any
    class_false_positives: Any

    @property
    def fitness(self):
        """Fitness value for hyper parameter search"""
        return self.subset_accuracy


class Model(noisytest.tunable.HyperParameterMixin):

    def __init__(self, input_preprocessor, preprocessor, regularization=1.1, kernel='rbf', kernel_gamma=0.00057):
        self._input_preprocessor = input_preprocessor  # additional preprocessor for reading in prediction data
        self._preprocessor = preprocessor
        self._svm = svm.SVC(C=regularization, kernel=kernel, gamma=kernel_gamma)

    def train(self, data):

        weights = float(tf.size(data.target)) / (np.max(data.target.numpy()) + 1) / \
                  np.bincount((data.target.numpy() + 0.1).astype(int))
        self._svm.class_weight = {0: weights[0], 1: weights[1], 2: weights[2], 3: weights[3]}

        processed = self._preprocessor.prepare_input_target_data(data)
        self._svm.fit(processed.input, processed.target)
        return self._svm.score(processed.input, processed.target)

    def validate(self, data):

        processed = self._preprocessor.prepare_input_target_data(data)

        subset_accuracy = self._svm.score(processed.input, processed.target)
        false_neg, false_pos = self.__calc_error_per_class(self._svm.predict(processed.input), processed.target)

        return ModelError(subset_accuracy, false_neg, false_pos)

    def predict(self, noise_data):

        time_frames = self._input_preprocessor.prepare_data(noise_data.time)
        noise_frames = self._preprocessor.append_parent(self._input_preprocessor).prepare_data(noise_data.noise_estimate)

        return time_frames.numpy(), self._svm.predict(noise_frames)

    @staticmethod
    def __calc_error_per_class(predicted, truth):
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
        return self._svm

    @property
    def preprocessor(self):
        return self._preprocessor

    @property
    def input_preprocessor(self):
        return self._input_preprocessor

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
        base = {'regularization': noisytest.tunable.HyperParameterRange(0.5, 0.1, 2.5)}

        if self.kernel == 'rbf':
            base['kernel_parameter'] = noisytest.tunable.HyperParameterRange(4.5e-04, 1.0e-05, 6.0e-04)

        return base
