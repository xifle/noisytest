from abc import ABC, abstractmethod


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
