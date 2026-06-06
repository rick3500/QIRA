from __future__ import annotations
from abc import ABC, abstractmethod
from ..path_state import PathState


class BaseConstraint(ABC):
    @abstractmethod
    def evaluate(self, path: PathState) -> float:
        """Return a score in [0, 1]. Higher is better."""


class LengthConstraint(BaseConstraint):
    """Rejects paths that exceed a maximum number of reasoning steps."""

    def __init__(self, max_steps: int = 10) -> None:
        self.max_steps = max_steps

    def evaluate(self, path: PathState) -> float:
        return 0.0 if len(path.trace) > self.max_steps else 1.0


class AssumptionConsistencyConstraint(BaseConstraint):
    """Penalises paths whose trace explicitly negates a stated assumption."""

    def evaluate(self, path: PathState) -> float:
        trace_text = " ".join(path.trace).lower()
        for assumption in path.assumptions:
            negated = f"not {assumption.lower()}"
            if negated in trace_text:
                return 0.5
        return 1.0


class CompositeConstraint(BaseConstraint):
    """Averages scores from multiple constraints."""

    def __init__(self, constraints: list[BaseConstraint]) -> None:
        self._constraints = constraints

    def evaluate(self, path: PathState) -> float:
        if not self._constraints:
            return 1.0
        scores = [c.evaluate(path) for c in self._constraints]
        return sum(scores) / len(scores)
