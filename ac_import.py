'''
Read in acoustic data from a simulation file
'''
import tensorflow as tf
import numpy as np
import csv
import toml
import glob
import os
import ac_config
import noisytest


def read_acoustics_log(filename):
    reader = noisytest.NoiseReader(filename)
    return reader.data()


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


def read_experiment(experiment_name, binary_labels, sparse_labels, always_full_frame=False):
    '''Returns acoustic signal data tensors (sample_size samples per element) and
    expected result tensors as tuple (signals, results) for the given filename.
    Applies framing where possible to generate more data frames'''

    reader = noisytest.ExperimentReader(noisytest.PreprocessorParameters())

    reader.do_pad_data = not always_full_frame
    exp = reader.read('data', experiment_name)

    return exp.input, exp.target, exp.no_of_samples, exp.no_of_annotated_blocks
