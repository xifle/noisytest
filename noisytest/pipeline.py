from abc import ABC, abstractmethod
import json

import noisytest.model
import noisytest.optimizer
import noisytest.preprocessor


class Pipeline(ABC):

    @abstractmethod
    def create_import_preprocessor(self, **kwargs):
        pass

    @abstractmethod
    def create_feature_preprocessor(self, **kwargs):
        pass

    @abstractmethod
    def create_model(self, **kwargs):
        pass

    @abstractmethod
    def load_training_data(self, training_data_directory, validation_data_directory):
        pass

    @abstractmethod
    def test(self, test_file: str):
        pass

    @abstractmethod
    def learn(self, training_data, validation_data):
        pass

    @abstractmethod
    def optimize(self, training_data, validation_data):
        pass


class DefaultPipeline(Pipeline):
    """Default noisytest pipeline consisting of model, (import/feature) preprocessor

       Derive from this class to create a new pipeline with different model / preprocessors
    """

    def __init__(self, config_file_handle):
        params = json.load(config_file_handle)

        self._import_preprocessor = self.create_import_preprocessor(**params['import_preprocessor']['arguments'])
        self._feature_preprocessor = self.create_feature_preprocessor(**params['feature_preprocessor']['arguments'])
        self._model = self.create_model(**params['model']['arguments'])

    def test(self, test_file: str):
        """Run noisytest on given file. Returns time-frame and model output as tuple"""

        imported_data = self.load_data(test_file)
        failure_prediction = self._model.predict(self._feature_preprocessor.prepare_data(imported_data.noise_estimate))
        return imported_data.time.numpy(), failure_prediction

    def learn(self, training_data, validation_data):
        self._model.train(self._feature_preprocessor.prepare_input_target_data(training_data))
        return self._model.validate(self._feature_preprocessor.prepare_input_target_data(validation_data))

    def optimize(self, training_data, validation_data):
        opt = noisytest.Optimizer(training_data, validation_data, self)
        opt.grid_search()

    def create_import_preprocessor(self, **kwargs):
        return noisytest.TimeDataFramer(**kwargs)

    def create_feature_preprocessor(self, **kwargs):
        return noisytest.Flatten(noisytest.DiscreteCosineTransform(
            noisytest.Mag2Log(noisytest.SpectrogramCompressor(noisytest.Spectrogram(**kwargs)))))

    def create_model(self, **kwargs):
        return noisytest.Model(**kwargs)

    def load_data(self, filename: str):
        """Load noise data from given file"""
        return noisytest.NoiseReader(filename, self._import_preprocessor).data()

    def load_training_data(self, training_data_directory, validation_data_directory):
        """Loads data from specified directory using the import preprocessor

        Returns a tuple (training_data, validation_data)
        """
        reader = noisytest.DataSetReader(self._import_preprocessor)
        reader.do_pad_data = True

        training_data = reader.read_data_set(training_data_directory)

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
