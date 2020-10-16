from dataclasses import dataclass
from typing import Any
import tensorflow as tf


@dataclass
class PreprocessorParameters:
    # The number of time samples per frame
    samples_per_frame: int = 4000

    # The stride for generating time-sample frames from original data
    time_frame_stride: int = 1000

    # The sample rate of the input data in Hz
    input_data_sample_rate = 10000


class Preprocessor(PreprocessorParameters):

    def create_empty_input_sample(self):
        return tf.fill([0, self.samples_per_frame], 0.0)

    def create_empty_target_sample(self):
        return tf.fill([0], 0.0)

    def is_matching_sample_time(self, sample_time):
        assert(sample_time > 0)
        return abs(1.0 / sample_time - self.input_data_sample_rate) < self.input_data_sample_rate / 10.0

    @property
    def frame_duration(self):
        """Frame duration in seconds"""
        return self.samples_per_frame / self.input_data_sample_rate
