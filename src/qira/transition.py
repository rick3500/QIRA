from __future__ import annotations
from typing import Callable
from .path_state import PathState
from .thought_state import ThoughtState


class RuleEngine:
    """Deterministic rule-based path expander.

    Each rule is a (condition, action) pair:
      - condition(path) -> bool   — fires when True
      - action(path) -> str       — produces the next reasoning step
    """

    def __init__(self) -> None:
        self._rules: list[tuple[Callable[[PathState], bool], Callable[[PathState], str]]] = []

    def add_rule(
        self,
        condition: Callable[[PathState], bool],
        action: Callable[[PathState], str],
    ) -> None:
        self._rules.append((condition, action))

    def propose_next_steps(self, path: PathState) -> list[PathState]:
        proposals: list[PathState] = []
        for condition, action in self._rules:
            if condition(path):
                new_path = path.clone()
                new_path.add_step(action(path))
                proposals.append(new_path)
        # if no rules fire, carry the path forward unchanged so it isn't dropped
        if not proposals:
            proposals.append(path.clone())
        return proposals


class TransitionOperator:
    def __init__(self, constraint_module, verification_module, rule_engine: RuleEngine) -> None:
        self.constraint_module = constraint_module
        self.verification_module = verification_module
        self.rule_engine = rule_engine

    def update(self, thought_state: ThoughtState) -> None:
        new_paths: list[PathState] = []
        new_amplitudes: list[float] = []

        for path, amp in zip(thought_state.paths, thought_state.amplitudes):
            proposals = self.rule_engine.propose_next_steps(path)

            for p in proposals:
                score = self.constraint_module.evaluate(p)
                verified = self.verification_module.check(p)

                p.score = score + verified
                new_amp = amp * p.score
                new_paths.append(p)
                new_amplitudes.append(new_amp)

        thought_state.paths = new_paths
        thought_state.amplitudes = new_amplitudes
        thought_state.normalize()
