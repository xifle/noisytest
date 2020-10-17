from dataclasses import dataclass
from typing import Any
import tensorflow as tf
import numpy as np


@dataclass
class PreprocessorParameters:
    # Maps training data keywords to target data
    keywords_to_target_data: dict

    # The number of time samples per frame
    samples_per_frame: int = 4000

    # The stride for generating time-sample frames from original data
    time_frame_stride: int = 1000

    # The sample rate of the input data in Hz
    input_data_sample_rate = 10000


class Preprocessor(PreprocessorParameters):
    """Preprocessor for noise data"""

    def create_empty_input_sample(self):
        return tf.fill([0, self.samples_per_frame], 0.0)

    def create_empty_target_sample(self):
        return tf.fill([0], 0.0)

    def is_matching_sample_time(self, sample_time):
        assert (sample_time > 0)
        return abs(1.0 / sample_time - self.input_data_sample_rate) < self.input_data_sample_rate / 10.0

    def pad_if_necessary(self, data):
        if data.size < self.samples_per_frame:
            data = np.pad(data, (0, self.samples_per_frame - data.size), mode='reflect',
                          reflect_type='odd')
        return data

    def frame_data(self, data):
        return tf.signal.frame(data, self.samples_per_frame, self.time_frame_stride)

    def frame_training_data(self, data, label):
        frames = self.frame_data(data)

        # supports only one label for the moment
        target_value = float(self.keywords_to_target_data[label])
        return frames, tf.fill([tf.shape(frames)[0]], target_value)

    @property
    def frame_duration(self):
        """Frame duration in seconds"""
        return self.samples_per_frame / self.input_data_sample_rate
