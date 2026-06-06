from __future__ import annotations
from .path_state import PathState


class ThoughtState:
    def __init__(self) -> None:
        self.paths: list[PathState] = []
        self.amplitudes: list[float] = []

    def add_path(self, path: PathState, amplitude: float = 1.0) -> None:
        self.paths.append(path)
        self.amplitudes.append(amplitude)

    def normalize(self) -> None:
        total = sum(abs(a) for a in self.amplitudes)
        if total == 0:
            return
        self.amplitudes = [a / total for a in self.amplitudes]

    def collapse(self) -> PathState:
        idx = max(range(len(self.amplitudes)), key=lambda i: abs(self.amplitudes[i]))
        return self.paths[idx]

    def __repr__(self) -> str:
        return f"ThoughtState(paths={len(self.paths)}, max_amp={max(self.amplitudes, default=0):.3f})"
