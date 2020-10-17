from dataclasses import dataclass
from typing import Any

import tensorflow as tf
import numpy as np
from sklearn import svm


@dataclass
class ModelError:
    subset_accuracy: Any
    class_false_negatives: Any
    class_false_positives: Any


class NoisyModel:

    def __init__(self, preprocessor, regularization=1.1, kernel='rbf', kernel_gamma=0.00057):
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
