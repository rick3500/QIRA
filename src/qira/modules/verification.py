from __future__ import annotations
from abc import ABC, abstractmethod
from ..path_state import PathState


class BaseVerifier(ABC):
    @abstractmethod
    def check(self, path: PathState) -> float:
        """Return 1.0 if verified, 0.0 if not."""


class GoalVerifier(BaseVerifier):
    """Checks whether a target conclusion appears anywhere in the path trace."""

    def __init__(self, goal: str) -> None:
        self._goal = goal.lower()

    def check(self, path: PathState) -> float:
        trace_text = " ".join(path.trace).lower()
        return 1.0 if self._goal in trace_text else 0.0
