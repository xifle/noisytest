import json

from noisytest.model import Model
from noisytest.optimizer import Optimizer
from noisytest.reader import NoiseReader, DataSetReader
from noisytest.preprocessor import Mag2Log, Spectrogram, SpectrogramCompressor, Flatten, TimeDataFramer, \
    DiscreteCosineTransform
from noisytest.pipeline.base import Pipeline


class DefaultPipeline(Pipeline):
    """Default noisytest pipeline consisting of model, (import/feature) preprocessor

       Derive from this class to create a new pipeline with different model / preprocessors
    """

    def __init__(self, config_file_handle):
        """Construct pipeline from JSON config file handle"""
        params = json.load(config_file_handle)

        self._import_preprocessor = self.create_import_preprocessor(**params['import_preprocessor']['arguments'])
        self._feature_preprocessor = self.create_feature_preprocessor(**params['feature_preprocessor']['arguments'])
        self._model = self.create_model(**params['model']['arguments'])

    def test(self, test_file: str):
        """Run noisytest on given file. Returns time-frame and model output as tuple"""

        imported_data = self.load_prediction_data(test_file)
        failure_prediction = self._model.predict(self._feature_preprocessor.prepare_data(imported_data.noise_estimate))
        return imported_data.time.numpy(), failure_prediction

    def learn(self, training_data, validation_data):
        """Run training and validation on this pipeline"""
        self._model.train(self._feature_preprocessor.prepare_input_target_data(training_data))
        return self._model.validate(self._feature_preprocessor.prepare_input_target_data(validation_data))

    def optimize(self, training_data, validation_data):
        """Run hyper-parameter search on this pipeline's parameters"""
        opt = Optimizer(training_data, validation_data, self)
        opt.grid_search()

    def create_import_preprocessor(self, **kwargs):
        """Factory method for import preprocessor chain"""
        return TimeDataFramer(**kwargs)

    def create_feature_preprocessor(self, **kwargs):
        """Factory method for feature preprocessor chain"""

        # creates the chain of preprocessors used for the default feature processing
        # execution order is inner first.
        return Flatten(DiscreteCosineTransform(
            Mag2Log(SpectrogramCompressor(Spectrogram(**kwargs)))))

    def create_model(self, **kwargs):
        """Factory method for noisytest model"""
        return Model(**kwargs)

    def load_prediction_data(self, filename: str):
        """Load noise data from given file"""
        return NoiseReader(filename, self._import_preprocessor).data()

    def load_training_data(self, training_data_directory, validation_data_directory):
        """Loads data from specified directory using the import preprocessor

        Returns a tuple (training_data, validation_data)
        """
        reader = DataSetReader(self._import_preprocessor)
        training_data = reader.read_data_set(training_data_directory)

        # Validation data must not be padded to avoid learning the symmetry / properties of padded
        # data samples. (for real tests there are no padded data frames)
        reader.do_pad_data = False
        validation_data = reader.read_data_set(validation_data_directory)

        return training_data, validation_data

    @property
    def import_preprocessor(self):
        return self._import_preprocessor

    @property
    def feature_preprocessor(self):
        return self._feature_preprocessor

    @property
    def model(self):
        return self._model
