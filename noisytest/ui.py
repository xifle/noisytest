import argparse
import pickle
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from noisytest.pipeline import DefaultPipeline
from noisytest import __version__


def _parse_arguments():
    parser = argparse.ArgumentParser(description='This is NoisyTest ' + __version__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pipeline', type=argparse.FileType('rb'),
                        help='the noisytest pipeline to use.',
                        default='default.noisy', required=False, metavar='NOISYTEST_PIPELINE')

    parser.add_argument('--config', type=argparse.FileType('rt'),
                        help='noisytest config file name',
                        default='noisytest-config.json', required=False, metavar='NOISYTEST_CONFIG')

    parser.add_argument("-v", "--verbosity", action="count",
                        help="console output verbosity")

    parser.set_defaults(func=lambda x: parser.print_usage())

    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train', help='Train a model from given data using default parameters',
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

    run_parser = subparsers.add_parser('run', help='Run test on specified noise file',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    run_parser.add_argument('noisefile', type=str,
                            help='a noise estimation file to test',
                            metavar='NOISE_FILE')
    run_parser.set_defaults(func=_test)

    return parser.parse_args()


def _test(args):
    pipeline = pickle.load(args['pipeline'])

    time_frames, failure_prediction = pipeline.test(args['noisefile'])
    for t, label in zip(time_frames, failure_prediction):
        if label > 0:
            logging.warning(f"possible {pipeline.import_preprocessor.target_data_to_keywords[label]}"
                            f" in time region {t[0]:.2}-{t[-1]:.2}")


def _train(args):
    if args['load_parameters']:
        pipeline = pickle.load(args['pipeline'])
    else:
        pipeline = DefaultPipeline(args['config'])

    training_data, validation_data = pipeline.load_training_data(args['training_data'], args['validation_data'])

    if args['optimize']:
        pipeline.optimize(training_data, validation_data)
    else:
        error = pipeline.learn(training_data, validation_data)
        logging.info(f'Validation subset accuracy: {error.subset_accuracy}')
        logging.info('Per-class error (false negative, false positive): '
                     f'{error.class_false_negatives}, {error.class_false_positives}')

    pickle.dump(pipeline, args['output'])


def _configure_logger(args):
    if args['verbosity']:
        level = max(logging.DEBUG, logging.CRITICAL + 10 - args['verbosity'] * 10)
    else:
        level = logging.INFO

    logging.basicConfig(format='%(levelname)s:%(message)s', level=level)


def run():
    args = _parse_arguments()
    _configure_logger(vars(args))
    args.func(vars(args))


if __name__ == "__main__":
    run()
