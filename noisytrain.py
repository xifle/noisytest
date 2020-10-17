from sklearn import svm

# Helper libraries

from joblib import dump
import tensorflow as tf
import numpy as np
import noisytest
import ac_config


def train(training, validation, preprocessor):
    model = noisytest.Model(preprocessor)

    acc = model.train(training)
    result = model.validate(validation)

    accuracy_ = acc, result.subset_accuracy
    error_per_class = result.class_false_negatives, result.class_false_positives
    mdl = model.svm

    return accuracy_, error_per_class, mdl


def load_data():
    preprocessor = noisytest.TimeDataFramer({
        'ok': 0,
        'impact': 1,
        'highaccelerations': 2,
        'oscillations': 3
    })
    reader = noisytest.DataSetReader(preprocessor)
    reader.do_pad_data = True

    training_data = reader.read_data_set('data/training')

    reader.do_pad_data = False
    validation_data = reader.read_data_set('data/validation')

    return training_data, validation_data


def train_dry_run():
    t, v = load_data()

    preprocessor = noisytest.Flatten(noisytest.DiscreteCosineTransform(
        noisytest.Mag2Log(noisytest.SpectrogramCompressor(noisytest.Spectrogram(None)))))
    (acc, err_per_class, mdl) = train(t, v, preprocessor)
    print('Class weights: ', mdl.class_weight)
    print('Overall Accuracy:', acc, 'Per-Class Error (false negative, false positive):', err_per_class)
    return mdl, acc


def train_and_store_model():
    model, accuracy = train_dry_run()

    noisytest_model = {
        "svm": model,
        "accuracy": accuracy
    }
    dump(noisytest_model, 'noisytest.model')


# parameter_grid_search()

# parameter_search(3)
# model, accuracy = train_dry_run()
# train_and_store_model()

preprocessor = noisytest.Flatten(noisytest.DiscreteCosineTransform(
    noisytest.Mag2Log(noisytest.SpectrogramCompressor(noisytest.Spectrogram(None)))))

model = noisytest.Model(preprocessor)

t, v = load_data()
opt = noisytest.Optimizer(t, v, model)
opt.grid_search()
