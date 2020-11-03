from abc import abstractmethod

from noisytest.preprocessor.base import Preprocessor


class InputOnlyPreprocessor(Preprocessor):
    """Preprocessor operating only on the noise data (leaving samples dimensions untouched)"""

    def prepare_data(self, data):
        result = super().prepare_data(data)
        result = self.process(result)
        return result

    def prepare_input_target_data(self, data):
        result = super().prepare_input_target_data(data)
        result.input = self.process(result.input)
        return result

    @abstractmethod
    def process(self, data):
        pass
