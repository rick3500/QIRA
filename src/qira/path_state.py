from __future__ import annotations
import copy


class PathState:
    def __init__(self) -> None:
        self.trace: list[str] = []
        self.assumptions: list[str] = []
        self.score: float = 0.0
        self.phase: float = 0.0
        self.metadata: dict = {}

    def add_step(self, step: str) -> None:
        self.trace.append(step)

    def update_score(self, delta: float) -> None:
        self.score += delta

    def clone(self) -> PathState:
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"PathState(steps={len(self.trace)}, score={self.score:.3f})"
