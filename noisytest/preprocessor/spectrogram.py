import tensorflow as tf

from noisytest.preprocessor.input_only import InputOnlyPreprocessor
from noisytest.tunable import HyperParameterRange


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
