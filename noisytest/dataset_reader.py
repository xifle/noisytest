import glob
import os

import noisytest.experiment_reader


class DataSetReader(noisytest.ExperimentReader):

    def read_data_set(self, path):
        data = self._preprocessor.create_empty_input_target_data()

        for filename in glob.iglob(os.path.join(path, "*" + noisytest.NoiseReader.file_extension())):
            exp_name = os.path.splitext(os.path.basename(filename))[0]

            experiment = self.read_experiment(path, exp_name)
            data = self._preprocessor.concat_input_target_data(data, experiment)

        return data


"""
    def read_training_data(binary_labels=False, sparse_labels=True):
        input = tf.fill([0, ac_config.samplesPerFrame], 0.0)

        total_number_of_time_samples = 0
        total_number_of_chunks = 0

        if binary_labels or sparse_labels:
            target = tf.fill([0], 0.0)
        else:
            target = tf.fill([0, num_categories], 0.0)

        for filename in glob.iglob('data/training/*.log'):
            exp_name = os.path.splitext(os.path.basename(filename))[0]
            exp_data = read_experiment('training/' + exp_name, binary_labels, sparse_labels)

            input = tf.concat([input, exp_data[0]], 0)
            target = tf.concat([target, exp_data[1]], 0)
            total_number_of_time_samples += exp_data[2]
            total_number_of_chunks += exp_data[3]

        print('The total input data tensor shape is:', tf.shape(input))
        print('The total target data tensor shape is:', tf.shape(target))
        print('The total training data time samples are: ', total_number_of_time_samples)
        print('The total number of annotated training chunks is: ', total_number_of_chunks)
        print('-------------------')

        return input, target

    def read_validation_data(binary_labels=False, sparse_labels=True):
        input = tf.fill([0, ac_config.samplesPerFrame], 0.0)

        total_number_of_time_samples = 0
        total_number_of_chunks = 0

        if binary_labels or sparse_labels:
            target = tf.fill([0], 0.0)
        else:
            target = tf.fill([0, num_categories], 0.0)

        for filename in glob.iglob('data/validation/*.log'):
            exp_name = os.path.splitext(os.path.basename(filename))[0]
            exp_data = read_experiment('validation/' + exp_name, binary_labels, sparse_labels, True)

            input = tf.concat([input, exp_data[0]], 0)
            target = tf.concat([target, exp_data[1]], 0)
            total_number_of_time_samples += exp_data[2]
            total_number_of_chunks += exp_data[3]

        print('The total validation data x tensor shape is:', tf.shape(input))
        print('The total validation data y tensor shape is:', tf.shape(target))
        print('The total validation data time samples are: ', total_number_of_time_samples)
        print('The total number of annotated validation chunks is: ', total_number_of_chunks)
        print('-------------------')

        return input, target

    def read_experiment33(experiment_name, binary_labels, sparse_labels, always_full_frame=False):
        '''Returns acoustic signal data tensors (sample_size samples per element) and
        expected result tensors as tuple (signals, results) for the given filename.
        Applies framing where possible to generate more data frames'''

        preprocessor = noisytest.Preprocessor({
            'ok': 0,
            'impact': 1,
            'highaccelerations': 2,
            'oscillations': 3
        })

        reader = noisytest.ExperimentReader(preprocessor)

        reader.do_pad_data = not always_full_frame
        exp = reader.read_experiment('data', experiment_name)

        return exp.input, exp.target, exp.no_of_time_samples, exp.no_of_annotated_blocks
"""
