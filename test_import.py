from ac_import import *

data = read_training_data(binary_labels=False, sparse_labels=False)

print('training data percentage over class', tf.math.reduce_mean(data[1], 0))
print('training data samples over class', tf.math.reduce_sum(data[1], 0))

validation = read_validation_data(binary_labels=False, sparse_labels=False)

print('validation data percentage over class', tf.math.reduce_mean(validation[1], 0))
print('validation data samples over class', tf.math.reduce_sum(validation[1], 0))
