import tensorflow as tf

from noisytest.preprocessor.input_only import InputOnlyPreprocessor


class Mag2Log(InputOnlyPreprocessor):
    """Preprocessor for conversion to logarithmic representation"""

    def process(self, data):
        return tf.math.log(data + 1e-06)

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
