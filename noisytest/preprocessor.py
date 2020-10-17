from dataclasses import dataclass
from typing import Any
from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np


@dataclass
class InputTargetData:
    input: Any
    target: Any
    no_of_time_samples: int
    no_of_annotated_blocks: int


class Preprocessor:
    """Preprocessor for noise / target data"""

    def __init__(self, parent):
        self._parent = parent

    def prepare_data(self, data):
        if self._parent is not None:
            data = self._parent.prepare_data(data)
        return data

    def prepare_input_target_data(self, data):
        if self._parent is not None:
            data = self._parent.prepare_input_target_data(data)
        return data

    @staticmethod
    def concat_input_target_data(a, b):
        return InputTargetData(
            tf.concat([a.input, b.input], 0),
            tf.concat([a.target, b.target], 0),
            a.no_of_time_samples + b.no_of_time_samples,
            a.no_of_annotated_blocks + b.no_of_annotated_blocks
        )

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent


class InputOnlyPreprocessor(ABC, Preprocessor):
    """Preprocessor for noise data"""

    def prepare_data(self, data):
        data = super().prepare_data(data)
        data = self.process(data)
        return data

    def prepare_input_target_data(self, data):
        data = super().prepare_input_target_data(data)
        data.input = self.process(data.input)
        return data

    @abstractmethod
    def process(self, data):
        pass


class TimeDataFramer(Preprocessor):

    def __init__(self, parent, keywords_to_target_data):
        super().__init__(parent)
        self._keywords_to_target_data = keywords_to_target_data
        self._samples_per_frame = 4000
        self._time_frame_stride = 1000
        self._input_data_sample_rate = 10000

    def create_empty_input_target_data(self):
        return InputTargetData(
            tf.fill([0, self.samples_per_frame], 0.0),
            tf.fill([0], 0.0),
            0, 0)

    def prepare_data(self, data):
        data = super().prepare_data(data)
        data = self.__frame_data(data)
        return data

    def prepare_input_target_data(self, data):
        data = super().prepare_input_target_data(data)
        data = self.__pad_if_necessary(data)

        data.input = self.__frame_data(data.input)
        target_value = float(self.keywords_to_target_data[data.target])
        data.target = tf.fill([tf.shape(data.input)[0]], target_value)

        return data

    def __pad_if_necessary(self, data):
        if data.input.size < self.samples_per_frame:
            data.input = np.pad(data.input, (0, self.samples_per_frame - data.input.size), mode='reflect',
                                reflect_type='odd')
            data.no_of_time_samples = data.input.size
        return data

    def __frame_data(self, data):
        return tf.signal.frame(data, self.samples_per_frame, self.time_frame_stride)

    @property
    def samples_per_frame(self):
        """The number of time samples per frame"""
        return self._samples_per_frame

    @property
    def time_frame_stride(self):
        """The stride for generating time-sample frames from original data"""
        return self._time_frame_stride

    @property
    def input_data_sample_rate(self):
        """The sample rate of the input data in Hz"""
        return self._input_data_sample_rate

    @property
    def frame_duration(self):
        """Frame duration in seconds"""
        return self.samples_per_frame / self.input_data_sample_rate

    @property
    def keywords_to_target_data(self):
        """Maps training data keywords to target data"""
        return self._keywords_to_target_data

    def is_matching_sample_time(self, sample_time):
        assert (sample_time > 0)
        return abs(1.0 / sample_time - self.input_data_sample_rate) < self.input_data_sample_rate / 10.0


class Spectrogram(InputOnlyPreprocessor):

    def __init__(self, parent):
        super().__init__(parent)
        self._fft_sample_length = 1024
        self._fft_stride_length = 834

    def process(self, data):
        return tf.abs(tf.signal.stft(data, self.fft_sample_length, self.fft_stride_length))

    @property
    def fft_sample_length(self):
        """FFT length in samples"""
        return self._fft_sample_length

    @property
    def fft_stride_length(self):
        """FFT stride length (hyperparameter)"""
        return self._fft_stride_length


class SpectrogramCompressor(InputOnlyPreprocessor):

    def __init__(self, parent):
        super().__init__(parent)
        self._compression_factor = 3

    def process(self, data):
        return tf.matmul(data, self.__averaging_tensor(tf.shape(data)[-1], self.compression_factor))

    @staticmethod
    def __averaging_tensor(original_size, compression_rate):
        """A tensor to compress the last dimension of tensor data by compression_rate"""
        if original_size % compression_rate > 0:
            print('originalSize:', original_size, 'compressionRate:', compression_rate)
            raise ValueError('original size must be a multiple of the compression rate!')

        averaging_block_line = tf.fill([1, int(compression_rate)], 1.0)
        padded = tf.pad(averaging_block_line, tf.constant([[0, 0], [0, int(original_size)]]))
        repeated = tf.tile(padded, multiples=[1, int(original_size / compression_rate)])
        result = tf.reshape(tf.slice(repeated, [0, 0], [1, int(original_size * original_size / compression_rate)]),
                            [original_size, int(original_size / compression_rate)])
        return result

    @property
    def compression_factor(self):
        """Compression factor"""
        return self._compression_factor


class Mag2Log(InputOnlyPreprocessor):

    def process(self, data):
        return tf.math.log(data + 1e-06)


class DiscreteCosineTransform(InputOnlyPreprocessor):

    def process(self, data):
        if tf.rank(data) > 2:
            time_frequency_flipped = tf.transpose(data, perm=[0, 2, 1])
        else:
            time_frequency_flipped = tf.transpose(data, perm=[1, 0])

        return tf.signal.dct(time_frequency_flipped, type=2, n=tf.shape(time_frequency_flipped)[-1], norm='ortho')


class Flatten(InputOnlyPreprocessor):

    def process(self, data):
        return tf.reshape(data, [-1, tf.shape(data)[-1] * tf.shape(data)[-2]])
