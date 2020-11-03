from dataclasses import dataclass
from typing import Any
from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np
from noisytest.tunable import HyperParameterMixin
from noisytest.tunable import HyperParameterRange


@dataclass
class InputTargetData:
    """Holds input/target pairs for training/validation"""
    input: Any
    target: Any
    no_of_time_samples: int
    no_of_annotated_blocks: int


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


class InputOnlyPreprocessor(Preprocessor):
    """Preprocessor operating only on the noise data (leaving samples dimensions untouched)"""

    def prepare_data(self, data):
        result = super().prepare_data(data)
        result = self.process(result)
        return result

    def prepare_input_target_data(self, data):
        result = super().prepare_input_target_data(data)
        result.input = self.process(result.input)
        return result

    @abstractmethod
    def process(self, data):
        pass


class TimeDataFramer(ImportPreprocessor):
    """Import preprocessor for time-data. Implements framing and padding to get equal-length chunks"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._keywords_to_target_data = self.arguments.get('keywords_to_target_data', None)
        self._samples_per_frame = self.arguments.get('samples_per_frame', 4000)
        self._time_frame_stride = self.arguments.get('time_frame_stride', 1000)
        self._input_data_sample_rate = self.arguments.get('input_data_sample_rate', 10000)

    def create_empty_input_target_data(self):
        return InputTargetData(
            tf.fill([0, self.samples_per_frame], 0.0),
            tf.fill([0], 0.0),
            0, 0)

    def prepare_data(self, data):
        result = super().prepare_data(data)
        result = self._frame_data(result)
        return result

    def prepare_input_target_data(self, data):
        result = super().prepare_input_target_data(data)
        result = self._pad_if_necessary(result)

        if not self.keywords_to_target_data:
            raise ValueError("preprocessor requires valid label keywords")

        result.input = self._frame_data(result.input)
        target_value = float(self.keywords_to_target_data[result.target])
        result.target = tf.fill([tf.shape(result.input)[0]], target_value)

        return result

    def _pad_if_necessary(self, data):
        if data.input.size < self.samples_per_frame:
            data.input = np.pad(data.input, (0, self.samples_per_frame - data.input.size), mode='reflect',
                                reflect_type='odd')
            data.no_of_time_samples = data.input.size
        return data

    def _frame_data(self, data):
        return tf.signal.frame(data, self.samples_per_frame, self.time_frame_stride)

    @property
    def samples_per_frame(self):
        """The number of time samples per frame"""
        return self._samples_per_frame

    @samples_per_frame.setter
    def samples_per_frame(self, value):
        """The number of time samples per frame"""
        self._samples_per_frame = value

    @property
    def time_frame_stride(self):
        """The stride for generating time-sample frames from original data"""
        return self._time_frame_stride

    @time_frame_stride.setter
    def time_frame_stride(self, value):
        """The stride for generating time-sample frames from original data"""
        self._time_frame_stride = value

    @property
    def input_data_sample_rate(self):
        """The sample rate of the input data in Hz"""
        return self._input_data_sample_rate

    @input_data_sample_rate.setter
    def input_data_sample_rate(self, value):
        """The sample rate of the input data in Hz"""
        self._input_data_sample_rate = value

    @property
    def frame_duration(self):
        """Frame duration in seconds"""
        return self.samples_per_frame / self.input_data_sample_rate

    def is_matching_sample_time(self, sample_time):
        assert (sample_time > 0)
        return abs(1.0 / sample_time - self.input_data_sample_rate) < self.input_data_sample_rate / 10.0

    @property
    def keywords_to_target_data(self):
        """Maps training data keywords to target data"""
        return self._keywords_to_target_data

    @property
    def hyper_parameters(self):
        return {}


class Spectrogram(InputOnlyPreprocessor):
    """Preprocessor to generate a spectrogram from input data using a STFT"""

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._fft_sample_length = self.arguments.get('fft_sample_length', 1024)
        self._fft_stride_length = self.arguments.get('fft_stride_length', 834)

    def process(self, data):
        return tf.abs(tf.signal.stft(data, self.fft_sample_length, self.fft_stride_length))

    @property
    def fft_sample_length(self):
        """FFT length in samples"""
        return self._fft_sample_length

    @fft_sample_length.setter
    def fft_sample_length(self, value):
        self._fft_sample_length = value

    @property
    def fft_stride_length(self):
        """FFT stride length (hyperparameter)"""
        return self._fft_stride_length

    @fft_stride_length.setter
    def fft_stride_length(self, value):
        self._fft_stride_length = value

    @property
    def hyper_parameters(self):
        return {'fft_stride_length': HyperParameterRange(224, 10, self.fft_sample_length + 1)}


class SpectrogramCompressor(InputOnlyPreprocessor):
    """Preprocessor to compress a spectrogram along its frequency axis"""

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._compression_factor = self.arguments.get('compression_factor', 3)

    def process(self, data):
        return tf.matmul(data, self._averaging_tensor(tf.shape(data)[-1], self.compression_factor))

    @staticmethod
    def _averaging_tensor(original_size, compression_rate):
        """A tensor to compress the last dimension of tensor data by compression_rate"""
        if original_size % compression_rate > 0:
            raise ValueError(f"original size ({original_size}) "
                             f"must be a multiple of the compression rate ({compression_rate})!")

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

    @compression_factor.setter
    def compression_factor(self, value):
        """Compression factor"""
        self._compression_factor = value

    @property
    def hyper_parameters(self):
        return {}


class Mag2Log(InputOnlyPreprocessor):
    """Preprocessor for conversion to logarithmic representation"""

    def process(self, data):
        return tf.math.log(data + 1e-06)

    @property
    def hyper_parameters(self):
        return {}


class DiscreteCosineTransform(InputOnlyPreprocessor):
    """Preprocessor performing a discrete cosine transform type II"""

    def process(self, data):
        if tf.rank(data) > 2:
            time_frequency_flipped = tf.transpose(data, perm=[0, 2, 1])
        else:
            time_frequency_flipped = tf.transpose(data, perm=[1, 0])

        return tf.signal.dct(time_frequency_flipped, type=2, n=tf.shape(time_frequency_flipped)[-1], norm='ortho')

    @property
    def hyper_parameters(self):
        return {}


class Flatten(InputOnlyPreprocessor):
    """Preprocessor to flatten the last two dimension of an input tensor"""

    def process(self, data):
        return tf.reshape(data, [-1, tf.shape(data)[-1] * tf.shape(data)[-2]])

    @property
    def hyper_parameters(self):
        return {}
