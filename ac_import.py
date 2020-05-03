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

binary_label_map = {
    'ok': 0,
    'oscillations': 1,
    'impact': 1,
    'highaccelerations': 1,
}

label_map = {
    'ok': 0,
    'impact': 1,
    'highaccelerations': 2,
    'oscillations': 3
}
num_categories = max(label_map.values()) + 1


def read_acoustics_log(filename):
    with open(filename, newline='') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=' ')

        # skip all header lines
        while str(next(data_reader)[0]).lstrip().startswith('#'):
            pass

        result = np.array(list(data_reader), np.float32)
        return result[:, 0], result[:, 1]


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
    acoustics_log_filename = 'data/' + experiment_name + '.log'
    meta_data_filename = 'data/' + experiment_name + '.toml'

    (time, data) = read_acoustics_log(acoustics_log_filename)
    meta_data = toml.load(meta_data_filename)

    input = tf.fill([0, ac_config.samplesPerFrame], 0.0)
    if binary_labels or sparse_labels:
        target = tf.fill([0], 0.0)
    else:
        target = tf.fill([0, num_categories], 0.0)

    assert len(meta_data['data']['chunk_ranges']) == len(meta_data['data']['chunk_labels']), \
        "%r: Number of chunk ranges must match the number of chunk labels" % experiment_name

    total_number_of_time_samples = 0

    for range, labels in zip(meta_data['data']['chunk_ranges'], meta_data['data']['chunk_labels']):
        start_time = range[0]
        end_time = range[1]

        if end_time == -1:
            end_time = time[-1] + 0.1

        if always_full_frame:
            # Don't pad - use a broader chunk than specified instead
            frame_duration = ac_config.samplesPerFrame / ac_config.inputDataSampleRate
            chunk_duration = end_time - start_time
            if chunk_duration < frame_duration:
                start_time -= (frame_duration - chunk_duration) / 2.0
                end_time = start_time + frame_duration + 1.0 / ac_config.inputDataSampleRate

        assert (start_time < end_time)
        # print("Processing chunk from t=", start_time, "to t=", end_time)

        chunk_data = data[(time >= start_time) & (time <= end_time)]
        if chunk_data.size < ac_config.samplesPerFrame:
            chunk_data = np.pad(chunk_data, (0, ac_config.samplesPerFrame - chunk_data.size),
                                mode='reflect', reflect_type='odd')

        total_number_of_time_samples += chunk_data.size

        # determine stride
        stride = ac_config.timeFrameStride
        if meta_data['data'].get('stride') == -1:
            stride = ac_config.samplesPerFrame

        input_data = tf.signal.frame(chunk_data, ac_config.samplesPerFrame, stride)

        if binary_labels:
            label_data = convert_labels_to_binary_result_tensor(labels, tf.shape(input_data)[0])
        elif sparse_labels:
            label_data = convert_labels_to_categorical_result_tensor_sparse(labels, tf.shape(input_data)[0])
        else:
            label_data = convert_labels_to_categorical_result_tensor(labels, tf.shape(input_data)[0], 1)

        input = tf.concat([input, input_data], 0)
        target = tf.concat([target, label_data], 0)

    return input, target, total_number_of_time_samples, len(meta_data['data']['chunk_ranges'])


def read_unlabeled_data(filename):
    (time, data) = read_acoustics_log(filename)

    return tf.signal.frame((time, data), ac_config.samplesPerFrame, ac_config.timeFrameStride)


def convert_labels_to_categorical_result_tensor(labels, expected_size, max_labels=num_categories):
    result = np.zeros(num_categories, dtype=np.float32)

    for label, idx in zip(labels, range(max_labels)):
        index = label_map.get(label)
        if index is not None:
            result[index] = 1.0

    return tf.reshape(tf.tile(result, [expected_size]), [expected_size, num_categories])


def convert_labels_to_categorical_result_tensor_sparse(labels, expected_size):
    result = float(label_map[labels[0]])
    return tf.fill([expected_size], result)


def convert_labels_to_binary_result_tensor(labels, expected_size):
    result = 1.0

    for label in labels:
        num = binary_label_map[label]
        if num == 0:
            result = 0.0
            break

    return tf.fill([expected_size], result)
