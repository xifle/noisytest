import tensorflow as tf

from noisytest.preprocessor.input_only import InputOnlyPreprocessor


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
