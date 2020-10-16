from dataclasses import dataclass
from typing import Any


@dataclass
class PreprocessorParameters:
    # The number of time samples per frame
    samples_per_frame: int = 4000

    # The stride for generating time-sample frames from original data
    time_frame_stride: int = 1000

