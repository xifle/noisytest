from dataclasses import dataclass
from typing import Any

import ac_config
import noisytest.noise_reader
import noisytest.meta_data_reader
import os
import tensorflow as tf
import numpy as np
import logging


@dataclass
class Experiment:
    input: Any
    target: Any
    no_of_samples: int
    no_of_annotated_blocks: int


label_map = {
    'ok': 0,
    'impact': 1,
    'highaccelerations': 2,
    'oscillations': 3
}
num_categories = max(label_map.values()) + 1


def convert_labels_to_categorical_result_tensor_sparse(labels, expected_size):
    result = float(label_map[labels[0]])
    return tf.fill([expected_size], result)


class ExperimentReader:

    def __init__(self, preprocessor):
        self._preprocessor = preprocessor
        self._do_pad_data = True

    @property
    def do_pad_data(self):
        return self._do_pad_data

    @do_pad_data.setter
    def do_pad_data(self, value: bool):
        self._do_pad_data = value

    def read(self, data_path: str, experiment_name: str) -> Experiment:
        full_filename = os.path.join(data_path, experiment_name)

        with noisytest.NoiseReader(full_filename) as noise_reader:
            noise_data = noise_reader.data()
            assert self._preprocessor.is_matching_sample_time(noise_reader.sample_time), \
                "%r: Sample time does not match preprocessor specs" % full_filename

        with noisytest.MetaDataReader(full_filename) as meta_reader:
            meta_data = meta_reader.data()

        input_samples = self._preprocessor.create_empty_input_sample()
        target_samples = self._preprocessor.create_empty_target_sample()

        total_number_of_samples = 0

        for block_range, labels in zip(meta_data.block_ranges, meta_data.block_labels):
            (start_time, end_time) = self.__start_end_time(full_filename, noise_data.time, block_range)

            logging.info("Processing chunk from t=", start_time, "to t=", end_time)

            block_data = noise_data.noise_estimate[(noise_data.time >= start_time) & (noise_data.time <= end_time)]
            if block_data.size < ac_config.samplesPerFrame:
                block_data = np.pad(block_data, (0, ac_config.samplesPerFrame - block_data.size), mode='reflect',
                                    reflect_type='odd')

            total_number_of_samples += block_data.size

            # determine stride
            stride = ac_config.timeFrameStride
            if meta_data.stride == -1:
                stride = ac_config.samplesPerFrame

            input_data = tf.signal.frame(block_data, ac_config.samplesPerFrame, stride)

            label_data = convert_labels_to_categorical_result_tensor_sparse(labels, tf.shape(input_data)[0])

            input_samples = tf.concat([input_samples, input_data], 0)
            target_samples = tf.concat([target_samples, label_data], 0)

        return Experiment(input_samples, target_samples, total_number_of_samples, len(meta_data.block_ranges))

    def __start_end_time(self, filename, time_data, block_range):
        """Determine start and end time of a data block. Returns a tuple (start, end) in seconds"""
        start_time = block_range[0]
        end_time = block_range[1]

        if end_time == -1:
            end_time = time_data[-1] + 0.1

        block_duration = end_time - start_time
        if block_duration < self._preprocessor.frame_duration and not self.do_pad_data:
            # Don't pad - use a broader chunk than specified instead
            start_time -= (self._preprocessor.frame_duration - block_duration) / 2.0
            end_time = start_time + self._preprocessor.frame_duration + 1.0 / self._preprocessor.input_data_sample_rate

        assert start_time < end_time, "%r: start time for a block must be less than end time" % filename
        return start_time, end_time
