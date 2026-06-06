"""
True QIRA for symbolic math.

States are sympy expressions — not text.
Transitions are algebraic operators (expand, simplify, factor, collect).
Interference is symbolic equivalence — no LLM.
Deduplication merges paths with the same canonical form, summing amplitudes.
FFT smooths the amplitude distribution across all live paths.
Amplitude tracks simplification progress.
Collapse selects the most dominant (simplest) surviving path.
"""

from __future__ import annotations
import numpy as np
from sympy import symbols, expand, collect, factor, simplify, count_ops, cancel
from sympy.core.expr import Expr

x, y, z = symbols("x y z")

_TRANSFORMS: list[tuple[str, callable]] = [
    ("expand",    expand),
    ("simplify",  simplify),
    ("factor",    factor),
    ("collect_x", lambda e: collect(e, x)),
    ("cancel",    cancel),
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SymbolicState:
    def __init__(self, expr: Expr, history: list[str] | None = None) -> None:
        self.expr = expr
        self.history: list[str] = history if history is not None else [str(expr)]
        self.score: float = 0.0

    def complexity(self) -> int:
        return count_ops(self.expr)

    def clone(self) -> SymbolicState:
        return SymbolicState(self.expr, self.history[:])

    def __repr__(self) -> str:
        return f"SymbolicState({self.expr}, ops={self.complexity()})"


# ---------------------------------------------------------------------------
# Superposition
# ---------------------------------------------------------------------------

class SymbolicSuperposition:
    def __init__(self) -> None:
        self.states: list[SymbolicState] = []
        self.amplitudes: list[float] = []

    def add_state(self, state: SymbolicState, amplitude: float = 1.0) -> None:
        self.states.append(state)
        self.amplitudes.append(amplitude)

    def normalize(self) -> None:
        total = sum(abs(a) for a in self.amplitudes)
        if total:
            self.amplitudes = [a / total for a in self.amplitudes]

    def deduplicate(self) -> None:
        """Merge paths whose expressions share the same expanded canonical form.

        Multiple routes to the same mathematical truth sum their amplitudes
        rather than proliferating as separate paths.
        """
        canonical: dict[str, tuple[SymbolicState, float]] = {}
        for state, amp in zip(self.states, self.amplitudes):
            key = str(expand(state.expr))
            if key in canonical:
                existing, existing_amp = canonical[key]
                # Keep the simpler derivation, combine amplitudes
                winner = state if state.complexity() < existing.complexity() else existing
                canonical[key] = (winner, existing_amp + amp)
            else:
                canonical[key] = (state, amp)
        self.states = [s for s, _ in canonical.values()]
        self.amplitudes = [a for _, a in canonical.values()]
        self.normalize()

    def prune(self, min_amplitude: float = 0.05, max_paths: int = 15) -> None:
        """Drop low-amplitude paths and cap total count to prevent explosion."""
        pairs = sorted(
            zip(self.states, self.amplitudes),
            key=lambda p: abs(p[1]),
            reverse=True,
        )[:max_paths]
        # Keep paths above threshold; always keep at least the top one
        filtered = [(s, a) for s, a in pairs if abs(a) >= min_amplitude]
        pairs = filtered if filtered else [pairs[0]]
        self.states, self.amplitudes = map(list, zip(*pairs))
        self.normalize()

    def spectral_mix(self) -> None:
        if len(self.amplitudes) < 2:
            return
        amps = np.array(self.amplitudes, dtype=float)
        kernel = np.array([1 / (1 + i) for i in range(len(amps))])
        smoothed = np.fft.ifft(np.fft.fft(amps) * kernel)
        self.amplitudes = smoothed.real.tolist()
        self.normalize()

    def collapse(self) -> SymbolicState:
        idx = max(range(len(self.amplitudes)), key=lambda i: abs(self.amplitudes[i]))
        return self.states[idx]

    def is_stable(self, threshold: float = 0.80) -> bool:
        return bool(self.amplitudes) and max(abs(a) for a in self.amplitudes) >= threshold

    def __repr__(self) -> str:
        best = min(self.states, key=lambda s: s.complexity(), default=None)
        return (
            f"SymbolicSuperposition(paths={len(self.states)}, "
            f"max_amp={max(self.amplitudes, default=0):.3f}, "
            f"simplest={best.expr if best else '?'})"
        )


# ---------------------------------------------------------------------------
# Markov Transition Operator
# ---------------------------------------------------------------------------

class AlgebraTransitionOperator:
    def update(self, sup: SymbolicSuperposition) -> None:
        new_states: list[SymbolicState] = []
        new_amplitudes: list[float] = []

        for state, amp in zip(sup.states, sup.amplitudes):
            for proposal in self._propose(state):
                gain = max(0, state.complexity() - proposal.complexity())
                score = 1.0 + gain * 0.3
                proposal.score = score
                new_states.append(proposal)
                new_amplitudes.append(amp * score)

        sup.states = new_states
        sup.amplitudes = new_amplitudes
        sup.normalize()
        sup.deduplicate()   # collapse equivalent paths before interference

    def _propose(self, state: SymbolicState) -> list[SymbolicState]:
        proposals = []
        for name, fn in _TRANSFORMS:
            try:
                new_expr = fn(state.expr)
                if new_expr != state.expr:
                    proposals.append(SymbolicState(
                        new_expr,
                        state.history + [f"{name:12s}  ->  {new_expr}"],
                    ))
            except Exception:
                pass
        return proposals or [state.clone()]


# ---------------------------------------------------------------------------
# Symbolic Interference (no LLM)
# ---------------------------------------------------------------------------

class SymbolicInterference:
    def apply(self, sup: SymbolicSuperposition) -> None:
        n = len(sup.states)
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = sup.states[i], sup.states[j]
                try:
                    equivalent = simplify(si.expr - sj.expr) == 0
                except Exception:
                    equivalent = False

                if equivalent:
                    sup.amplitudes[i] *= 1.2
                    sup.amplitudes[j] *= 1.2
                else:
                    ci, cj = si.complexity(), sj.complexity()
                    if ci > cj * 2:
                        sup.amplitudes[i] *= 0.5
                    elif cj > ci * 2:
                        sup.amplitudes[j] *= 0.5
        sup.normalize()


# ---------------------------------------------------------------------------
# Reasoning Cycle
# ---------------------------------------------------------------------------

def symbolic_reasoning_cycle(
    sup: SymbolicSuperposition,
    transition: AlgebraTransitionOperator,
    interference: SymbolicInterference,
    max_iters: int = 10,
) -> SymbolicState:
    for iteration in range(max_iters):
        transition.update(sup)   # expand paths + deduplicate
        interference.apply(sup)  # adjust amplitudes by equivalence
        sup.spectral_mix()       # FFT smoothing
        sup.prune()              # drop low-amplitude paths, cap at 15

        print(f"  [cycle {iteration + 1}] {sup}")

        if sup.is_stable():
            print(f"  -> stable after {iteration + 1} iteration(s)")
            break

    return sup.collapse()
