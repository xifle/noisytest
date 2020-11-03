import tensorflow as tf
import numpy as np

from noisytest.preprocessor.base import ImportPreprocessor
from noisytest.preprocessor.data import InputTargetData


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
            raise ValueError("Preprocessor requires valid label keywords. Please check your noisytest config file!")

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
