from abc import ABC, abstractmethod
from typing import Any, Tuple


class Plant(ABC):
    def __init__(self, y: float):
        self.y = y

    @abstractmethod
    def update(self, u: float, d: float):
        pass
