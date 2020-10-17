import argparse
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import noisytest


def _parse_arguments():
    parser = argparse.ArgumentParser(description='This is NoisyTest ' + noisytest.__version__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', type=argparse.FileType('rb'),
                        help='the noisytest model to use.',
                        default='default.noisymdl', required=False, metavar='NOISYTEST_MODEL')
    parser.add_argument('--noise-file', type=str,
                        help='a noise estimation file to test',
                        default='noise.log', required=False, metavar='NOISE_FILE')

    parser.set_defaults(func=_test)

    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train', help='train a model from given data using default parameters',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_parser.add_argument('--output', '-o', type=argparse.FileType('wb'), help='output model filename',
                              default='default.noisymdl', required=False, metavar='OUTPUT_FILENAME')

    train_parser.add_argument('--training-data', type=str, help='training data directory',
                              default='data/training', required=False, metavar='DIRECTORY')

    train_parser.add_argument('--validation-data', type=str, help='validation data directory',
                              default='data/validation', required=False, metavar='DIRECTORY')

    train_parser.add_argument('--load-parameters', action='store_true',
                              help='start from existing model parameters given via --model', required=False)

    train_parser.add_argument('--kernel', type=str,
                              help='svm kernel type. Overwrites parameter loaded from model if given',
                              choices=['rbf', 'linear'], default='rbf', required=False)

    train_parser.add_argument('--optimize', help='optimize the model parameters using a grid-based search',
                              action='store_true', required=False)
    train_parser.set_defaults(func=_train)

    return parser.parse_args()


def _default_input_preprocessor():
    return noisytest.TimeDataFramer({
        'ok': 0,
        'impact': 1,
        'highaccelerations': 2,
        'oscillations': 3
    })


def _default_preprocessor(parent=None):
    return noisytest.Flatten(noisytest.DiscreteCosineTransform(
        noisytest.Mag2Log(noisytest.SpectrogramCompressor(noisytest.Spectrogram(parent)))))


def _load_data(training_dir, validation_dir, input_preprocessor):
    reader = noisytest.DataSetReader(input_preprocessor)
    reader.do_pad_data = True

    training_data = reader.read_data_set(training_dir)

    reader.do_pad_data = False
    validation_data = reader.read_data_set(validation_dir)

    return training_data, validation_data


def _test(args):
    print(args)
    model = pickle.load(args['model'])

    reader = noisytest.NoiseReader(args['noise_file'])

    time_frames, failure_prediction = model.predict(reader.data())
    for t, label in zip(time_frames, failure_prediction):
        if label > 0:
            print("Possible", model.input_preprocessor.target_data_to_keywords[label],
                  "in time region", t[0], "-", t[-1])


def _train(args):
    if args['load_parameters']:
        model = pickle.load(args['model'])
    else:
        model = noisytest.Model(_default_input_preprocessor(), _default_preprocessor(), kernel=args['kernel'])

    training_data, validation_data = _load_data(args['training_data'], args['validation_data'],
                                                model.input_preprocessor)

    if args['optimize']:
        opt = noisytest.Optimizer(training_data, validation_data, model)
        opt.grid_search()
    else:
        model.train(training_data)
        error = model.validate(validation_data)
        print('Validation subset accuracy:', error.subset_accuracy,
              ', per-class error (false negative, false positive):',
              error.class_false_negatives, error.class_false_positives)

    pickle.dump(model, args['output'])


def run():
    args = _parse_arguments()
    args.func(vars(args))


if __name__ == "__main__":
    run()
