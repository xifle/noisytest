from dataclasses import dataclass
from typing import Any
import csv

from noisytest.preprocessor import NullPreprocessor
import numpy as np


@dataclass
class Noise:
    time: Any
    noise_estimate: Any


class NoiseReader:
    """Reads in column-based data format with time and noise estimate"""

    def __init__(self, filename: str, preprocessor=NullPreprocessor()):
        """Construct a noise data reader from given filename

        The file extension is automatically appended if none is supplied
        """
        self._preprocessor = preprocessor

        if not filename.endswith(self.file_extension()):
            filename = filename + self.file_extension()

        with open(filename, newline='') as file_handle:
            reader = csv.reader(file_handle, delimiter=' ')
            self._read_header(reader)
            self._data = np.array(list(reader), np.float32)

    @staticmethod
    def _read_header(csv_reader):
        # skip all header lines
        while str(next(csv_reader)[0]).lstrip().startswith('#'):
            pass

    def data(self):
        """Returns simulation time and noise estimate arrays"""
        return Noise(self._preprocessor.prepare_data(self._data[:, 0]),
                     self._preprocessor.prepare_data(self._data[:, 1]))

    @staticmethod
    def file_extension():
        return ".log"

    @property
    def sample_time(self):
        """Returns the sample time in seconds"""
        return np.mean(np.diff(self._data[:, 0]))
