from dataclasses import dataclass
from typing import Any

@dataclass
class InputTargetData:
    """Holds input/target pairs for training/validation"""
    input: Any
    target: Any
    no_of_time_samples: int
    no_of_annotated_blocks: int
