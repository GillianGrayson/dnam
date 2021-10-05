from dataclasses import dataclass
from typing import Any


@dataclass
class Field:
    label: str
    column: Any
