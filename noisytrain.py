from sklearn import svm

# Helper libraries

from joblib import dump
from ac_preprocess import *
from ac_import import *
from ac_preprocess import MeanSubtraction

from noisytest import experiment_reader


def calc_error_per_class(y, y_val):
    false_positives = np.zeros(experiment_reader.num_categories)
    false_negatives = np.zeros(experiment_reader.num_categories)

    for (pred, exp) in zip(y, y_val.numpy()):
        if pred != exp:
            false_positives[int(pred + 0.1)] += 1.0
            false_negatives[int(exp + 0.1)] += 1.0

    false_positives = false_positives / len(y)
    false_negatives = false_negatives / len(y)
    return false_negatives, false_positives


def train(x_raw, y, x_raw_val, y_val):
    # Calculate balanced weights
    weights = float(tf.size(y)) / (np.max(y.numpy()) + 1) / np.bincount((y.numpy() + 0.1).astype(int))

    mdl = svm.SVC(C=ac_config.svmRegularization, kernel=ac_config.svmKernel, gamma=ac_config.rbfKernelGamma,
                  class_weight={0: weights[0], 1: weights[1], 2: weights[2], 3: weights[3]})

    x = prepare_compressed_flat(x_raw)
    x_val = prepare_compressed_flat(x_raw_val)

    normalizer = MeanSubtraction()
    normalizer.learn(x, True)
    # x = normalizer.center_data(x)
    # x_val = normalizer.center_data(x_val)

    mdl.fit(x, y)

    accuracy_ = mdl.score(x, y), mdl.score(x_val, y_val)
    error_per_class = calc_error_per_class(mdl.predict(x_val), y_val)

    return accuracy_, error_per_class, mdl, normalizer


def load_data():
    training_data = read_training_data(False, True)
    validation_data = read_validation_data(False, True)

    return training_data, validation_data


def train_dry_run():
    t, v = load_data()
    (acc, err_per_class, mdl, norm) = train(t[0], t[1], v[0], v[1])
    print('Class weights: ', mdl.class_weight)
    print('Overall Accuracy:', acc, 'Per-Class Error (false negative, false positive):', err_per_class)
    return mdl, norm, acc


def train_and_store_model():
    model, normalizer, accuracy = train_dry_run()

    noisytest_model = {
        "svm": model,
        "normalizer": normalizer,
        "accuracy": accuracy
    }
    dump(noisytest_model, 'noisytest.model')


def find_parameter(t, v, parameter_array, default_value, set_parameter_function):
    accuracy_result = []
    for param_value in parameter_array:
        set_parameter_function(param_value)
        (acc, _, mdl, _) = train(t[0], t[1], v[0], v[1])
        accuracy_result.append(acc[1])

    highest_accuracy = np.amax(accuracy_result)
    indices_best = np.asarray(accuracy_result == highest_accuracy).nonzero()[0]

    if np.size(indices_best) == np.size(accuracy_result):
        # indefinite result
        best_value = default_value
    else:
        # use mean value
        if np.issubdtype(type(parameter_array[0]), np.integer):
            # print('info: Choosing a sample in the mid of all optimal values')
            mid = int(np.size(indices_best) / 2)
            best_value = parameter_array[indices_best[mid]]
        else:
            # print('info: Using mean value')
            best_value = np.mean(parameter_array[indices_best])

    set_parameter_function(best_value)

    return best_value, highest_accuracy


def set_regularization(value):
    ac_config.svmRegularization = value


def set_fft_stride_length(value):
    ac_config.fftStrideLength = value


def set_kernel_gamma(value):
    ac_config.rbfKernelGamma = value


def parameter_line_search(iterations=2):
    t, v = load_data()

    for i in range(1, iterations + 1):
        print('Starting Iteration', i)

        hat_c, hat_c_accuracy = find_parameter(t, v, np.arange(0.3, 3.0, 0.1),
                                               ac_config.svmRegularization,
                                               set_regularization)
        print('best hatc:', hat_c, 'accuracy:', hat_c_accuracy)

        stride_fft, stride_fft_accuracy = find_parameter(t, v,
                                                         np.arange(224, ac_config.fftSampleLength + 1, 3, np.int32),
                                                         ac_config.fftStrideLength,
                                                         set_fft_stride_length)
        print('best stride:', stride_fft, 'accuracy:', stride_fft_accuracy)

        if ac_config.svmKernel == 'rbf':
            gamma, gamma_accuracy = find_parameter(t, v, np.arange(4.0e-04, 6.0e-04, .5e-05),
                                                   ac_config.rbfKernelGamma,
                                                   set_kernel_gamma)
            print('best gamma:', gamma, 'accuracy:', gamma_accuracy)

    train_dry_run()


def parameter_grid_search():
    t, v = load_data()

    bestaccuracy = 0
    besthatc = 0
    beststride = 0
    bestgamma = 0

    for c in np.arange(0.5, 2.5, 0.1):
        set_regularization(c)

        if ac_config.svmKernel == 'rbf':
            for stride in np.arange(224, ac_config.fftSampleLength + 1, 10, np.int32):
                set_fft_stride_length(stride)

                # print('c:', c, 'stride:', stride)
                gamma, accuracy = find_parameter(t, v, np.arange(4.5e-04, 6.0e-04, 1.0e-05),
                                                 ac_config.rbfKernelGamma,
                                                 set_kernel_gamma)
                if accuracy > bestaccuracy:
                    bestaccuracy = accuracy
                    besthatc = c
                    beststride = stride
                    bestgamma = gamma

                    print('new optimum with accuracy:', accuracy)
                    print(besthatc)
                    print(beststride)
                    print(bestgamma)

        else:
            stride_fft, accuracy = find_parameter(t, v,
                                                  np.arange(224, ac_config.fftSampleLength + 1, 3, np.int32),
                                                  ac_config.fftStrideLength,
                                                  set_fft_stride_length)

            if accuracy > bestaccuracy:
                bestaccuracy = accuracy
                besthatc = c
                beststride = stride_fft

                print('new optimum with accuracy:', accuracy)
                print(besthatc)
                print(beststride)


#parameter_grid_search()

# parameter_search(3)
model, normalizer, accuracy = train_dry_run()
# train_and_store_model()
