from dataclasses import dataclass
from typing import Any

import toml


@dataclass
class MetaData:
    """Represents meta data for training/validation experiment"""
    block_ranges: Any
    block_labels: Any
    stride: int


class MetaDataReader:
    """TOML-based reader for training/validation data annotations"""

    def __init__(self, filename: str):
        """Construct a meta data reader from given filename

        The file extension is automatically appended if none is supplied
        """
        if not filename.endswith(self.file_extension()):
            filename = filename + self.file_extension()

        self._reader = toml.load(filename)

        number_of_chunks = len(self._reader['data']['chunk_ranges'])
        number_of_labels = len(self._reader['data']['chunk_labels'])

        if number_of_chunks != number_of_labels:
            raise ValueError(f"{filename}: Number of chunk ranges must match the number of chunk labels")

    @staticmethod
    def file_extension():
        return ".toml"

    def data(self):
        """Return MetaData object"""
        return MetaData(self._reader['data']['chunk_ranges'], self._reader['data']['chunk_labels'],
                        self._reader['data'].get('stride', 0))
