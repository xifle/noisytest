import tensorflow as tf
import ac_config


class MeanSubtraction:

    def __init__(self):
        self.mean_data = None
        self.std_dev = None

    def learn(self, data, scale=False):
        if scale:
            self.std_dev = tf.math.reduce_std(data, axis=0)
        else:
            self.std_dev = 1.0
        self.mean_data = tf.math.reduce_mean(data, axis=0)

    def center_data(self, data):
        return (data - self.mean_data) / self.std_dev

    def save(self, filename):
        tf.io.write_file(filename, tf.io.serialize_tensor(self.mean_data))

    def load(self, filename):
        self.mean_data = tf.io.parse_tensor(tf.io.read_file(filename), tf.float32)


def averaging_matrix(original_size, compression_rate):
    if original_size % compression_rate > 0:
        print('originalSize:', original_size, 'compressionRate:', compression_rate)
        raise ValueError('original size must be a multiple of the compression rate!')
    averaging_block_line = tf.fill([1, int(compression_rate)], 1.0)
    padded = tf.pad(averaging_block_line, tf.constant([[0, 0], [0, int(original_size)]]))
    repeated = tf.tile(padded, multiples=[1, int(original_size / compression_rate)])
    result = tf.reshape(tf.slice(repeated, [0, 0], [1, int(original_size * original_size / compression_rate)]),
                        [original_size, int(original_size / compression_rate)])
    return result


def spectrogram(data):
    magnitudes = tf.abs(tf.signal.stft(data,
                                       ac_config.fftSampleLength,
                                       ac_config.fftStrideLength))
    return magnitudes


def mag2log(data):
    return tf.math.log(data + 1e-06)


def compress_spectrogram(spec, compression_rate):
    return tf.matmul(spec, averaging_matrix(tf.shape(spec)[-1], compression_rate))


def flatten(data):
    return tf.reshape(data, [-1, tf.shape(data)[-1] * tf.shape(data)[-2]])


def average_over_time_domain(data):
    '''Aggregate over time-domain (addition)'''
    compressed = tf.fill([tf.shape(data)[-3], tf.shape(data)[-1]], 0.0)
    for i in range(0, tf.shape(data)[-2]):
        compressed = compressed + data[:, i, :]

    return tf.reshape(compressed, [tf.shape(compressed)[-2], tf.shape(compressed)[-1]])


def add_feature_axis(data):
    return tf.reshape(data, [-1, tf.shape(data)[-2], tf.shape(data)[-1], 1])


def prepare_time_averaged(x):
    return mag2log(average_over_time_domain(compress_spectrogram(spectrogram(x), 3)))


def prepare_fft(x):
    return mag2log(compress_spectrogram(tf.abs(tf.signal.rfft(x))[:, 0:-1], 1))


def prepare_compressed_flat(x):
    return flatten(mag2log(compress_spectrogram(spectrogram(x), 3)))


def prepare_flat(x):
    return flatten(mag2log(spectrogram(x)))


def prepare_compressed(x):
    return mag2log(compress_spectrogram(spectrogram(x), 3))


def prepare_compressed_2d(x):
    return add_feature_axis(mag2log(compress_spectrogram(spectrogram(x), 3)))


def prepare_raw(x):
    return tf.reshape(x, [-1, tf.shape(x)[-1], 1])


def prepare_2d(x):
    return add_feature_axis(mag2log(spectrogram(x)[:, :, 0:-1]))
