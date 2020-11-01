import argparse
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import noisytest


def _parse_arguments():
    parser = argparse.ArgumentParser(description='This is NoisyTest ' + noisytest.__version__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pipeline', type=argparse.FileType('rb'),
                        help='the noisytest pipeline to use.',
                        default='default.noisy', required=False, metavar='NOISYTEST_PIPELINE')

    parser.add_argument('--noise-file', type=str,
                        help='a noise estimation file to test',
                        default='noise.log', required=False, metavar='NOISE_FILE')

    parser.add_argument('--config', type=argparse.FileType('rt'),
                        help='noisytest config file name',
                        default='noisytest-config.json', required=False, metavar='NOISYTEST_CONFIG')

    parser.set_defaults(func=_test)

    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train', help='train a model from given data using default parameters',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_parser.add_argument('--output', '-o', type=argparse.FileType('wb'), help='output pipeline filename',
                              default='default.noisy', required=False, metavar='OUTPUT_FILENAME')

    train_parser.add_argument('--training-data', type=str, help='training data directory',
                              default='data/training', required=False, metavar='DIRECTORY')

    train_parser.add_argument('--validation-data', type=str, help='validation data directory',
                              default='data/validation', required=False, metavar='DIRECTORY')

    train_parser.add_argument('--load-parameters', action='store_true',
                              help='start from existing model parameters given via --pipeline', required=False)

    train_parser.add_argument('--optimize', help='optimize the model parameters using a grid-based search',
                              action='store_true', required=False)
    train_parser.set_defaults(func=_train)

    return parser.parse_args()


def _test(args):
    pipeline = pickle.load(args['pipeline'])

    time_frames, failure_prediction = pipeline.test(args['noise_file'])
    for t, label in zip(time_frames, failure_prediction):
        if label > 0:
            print("<!>:possible", pipeline.import_preprocessor.target_data_to_keywords[label],
                  "in time region", t[0], "-", t[-1])


def _train(args):
    if args['load_parameters']:
        pipeline = pickle.load(args['pipeline'])
    else:
        pipeline = noisytest.DefaultPipeline(args['config'])

    training_data, validation_data = pipeline.load_training_data(args['training_data'], args['validation_data'])

    if args['optimize']:
        pipeline.optimize(training_data, validation_data)
    else:
        error = pipeline.learn(training_data, validation_data)
        print('Validation subset accuracy:', error.subset_accuracy,
              ', per-class error (false negative, false positive):',
              error.class_false_negatives, error.class_false_positives)

    pickle.dump(pipeline, args['output'])


def run():
    args = _parse_arguments()
    args.func(vars(args))


if __name__ == "__main__":
    run()
