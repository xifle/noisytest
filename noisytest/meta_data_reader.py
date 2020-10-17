from dataclasses import dataclass
from typing import Any

import toml


@dataclass
class MetaData:
    """Represents meta data for training/validation experiment"""
    block_ranges: Any
    block_labels: Any
    stride: int


class MetaDataReader(MetaData):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __init__(self, filename: str):
        """Construct a meta data reader from given filename

        The file extension is automatically appended if none is supplied
        """
        if not filename.endswith(self.file_extension()):
            filename = filename + self.file_extension()

        self._reader = toml.load(filename)

        assert len(self._reader['data']['chunk_ranges']) == len(self._reader['data']['chunk_labels']), \
            "%r: Number of chunk ranges must match the number of chunk labels" % filename

    @staticmethod
    def file_extension():
        return ".toml"

    def data(self):
        """Return MetaData object"""
        return MetaData(self._reader['data']['chunk_ranges'], self._reader['data']['chunk_labels'],
                        self._reader['data'].get('stride', 0))
