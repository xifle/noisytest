from dataclasses import dataclass
from typing import Any

import ac_config
import noisytest.noise_reader
import noisytest.meta_data_reader
import os
import tensorflow as tf
import numpy as np


@dataclass
class Experiment:
    input: Any
    target: Any
    no_of_annotated_blocks: int
    no_of_samples: int


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

    def __init__(self, preprocessor_params):
        self._do_pad_data = True
        self._preprocessor_params = preprocessor_params

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

        with noisytest.MetaDataReader(full_filename) as meta_reader:
            meta_data = meta_reader.data()

        input = tf.fill([0, ac_config.samplesPerFrame], 0.0)
        target = tf.fill([0], 0.0)

        total_number_of_time_samples = 0

        for range, labels in zip(meta_data.block_ranges, meta_data.block_labels):
            start_time = range[0]
            end_time = range[1]

            if end_time == -1:
                end_time = noise_data.time[-1] + 0.1

            if not self._do_pad_data:
                # Don't pad - use a broader chunk than specified instead
                frame_duration = ac_config.samplesPerFrame / ac_config.inputDataSampleRate
                chunk_duration = end_time - start_time
                if chunk_duration < frame_duration:
                    start_time -= (frame_duration - chunk_duration) / 2.0
                    end_time = start_time + frame_duration + 1.0 / ac_config.inputDataSampleRate

            assert (start_time < end_time)
            # print("Processing chunk from t=", start_time, "to t=", end_time)

            chunk_data = noise_data.noise_estimate[(noise_data.time >= start_time) & (noise_data.time <= end_time)]
            if chunk_data.size < ac_config.samplesPerFrame:
                chunk_data = np.pad(chunk_data, (0, ac_config.samplesPerFrame - chunk_data.size),
                                    mode='reflect', reflect_type='odd')

            total_number_of_time_samples += chunk_data.size

            # determine stride
            stride = ac_config.timeFrameStride
            if meta_data.stride == -1:
                stride = ac_config.samplesPerFrame

            input_data = tf.signal.frame(chunk_data, ac_config.samplesPerFrame, stride)

            label_data = convert_labels_to_categorical_result_tensor_sparse(labels, tf.shape(input_data)[0])

            input = tf.concat([input, input_data], 0)
            target = tf.concat([target, label_data], 0)

        return Experiment(input, target, total_number_of_time_samples, len(meta_data.block_ranges))
