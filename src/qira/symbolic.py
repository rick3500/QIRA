"""
True QIRA for symbolic math.

States are sympy expressions — not text.
Transitions are algebraic operators (expand, simplify, factor, collect).
Interference is symbolic equivalence — no LLM.
FFT smooths the amplitude distribution across all live paths.
Amplitude tracks simplification progress.
Collapse selects the most simplified form.

This is the Markov chain + quantum-inspired architecture the blueprint describes.
"""

from __future__ import annotations
import numpy as np
from sympy import symbols, expand, collect, factor, simplify, count_ops, cancel, pretty
from sympy.core.expr import Expr

x, y, z = symbols("x y z")

# Algebraic transformation operators — each is a Markov transition candidate
_TRANSFORMS: list[tuple[str, callable]] = [
    ("expand",     expand),
    ("simplify",   simplify),
    ("factor",     factor),
    ("collect_x",  lambda e: collect(e, x)),
    ("cancel",     cancel),
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SymbolicState:
    """A single reasoning path — an algebraic expression plus its derivation history."""

    def __init__(self, expr: Expr, history: list[str] | None = None) -> None:
        self.expr = expr
        self.history: list[str] = history if history is not None else [str(expr)]
        self.score: float = 0.0

    def complexity(self) -> int:
        """Number of operations in the expression — lower is more simplified."""
        return count_ops(self.expr)

    def clone(self) -> SymbolicState:
        return SymbolicState(self.expr, self.history[:])

    def __repr__(self) -> str:
        return f"SymbolicState({self.expr}, ops={self.complexity()})"


# ---------------------------------------------------------------------------
# Superposition
# ---------------------------------------------------------------------------

class SymbolicSuperposition:
    """Superposition of candidate algebraic states with associated amplitudes."""

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

    def prune(self, min_amplitude: float = 0.05) -> None:
        pairs = [(s, a) for s, a in zip(self.states, self.amplitudes) if abs(a) >= min_amplitude]
        if pairs:
            self.states, self.amplitudes = map(list, zip(*pairs))
            self.normalize()

    def spectral_mix(self) -> None:
        """FFT low-pass filter smooths amplitude noise across the path ensemble."""
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
    """
    Applies every algebraic transformation to every live state.
    Amplitude is boosted proportional to the complexity reduction achieved —
    simpler results naturally dominate.
    """

    def update(self, sup: SymbolicSuperposition) -> None:
        new_states: list[SymbolicState] = []
        new_amplitudes: list[float] = []

        for state, amp in zip(sup.states, sup.amplitudes):
            proposals = self._propose(state)
            for proposal in proposals:
                gain = max(0, state.complexity() - proposal.complexity())
                score = 1.0 + gain * 0.3   # reward simplification
                proposal.score = score
                new_states.append(proposal)
                new_amplitudes.append(amp * score)

        sup.states = new_states
        sup.amplitudes = new_amplitudes
        sup.normalize()

    def _propose(self, state: SymbolicState) -> list[SymbolicState]:
        proposals = []
        for name, fn in _TRANSFORMS:
            try:
                new_expr = fn(state.expr)
                if new_expr != state.expr:
                    proposals.append(SymbolicState(
                        new_expr,
                        state.history + [f"{name:12s}  →  {pretty(new_expr, use_unicode=True)}"],
                    ))
            except Exception:
                pass
        return proposals or [state.clone()]  # carry forward if no transform applies


# ---------------------------------------------------------------------------
# Symbolic Interference (no LLM — pure sympy)
# ---------------------------------------------------------------------------

class SymbolicInterference:
    """
    Constructive interference: two paths that are mathematically equivalent
    amplify each other — they're two routes to the same truth.
    Destructive interference: a path that is significantly more complex than
    another is suppressed — it's going the wrong direction.
    """

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
                        sup.amplitudes[i] *= 0.5   # i is much more complex — suppress
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
        transition.update(sup)
        interference.apply(sup)
        sup.spectral_mix()
        sup.prune()

        print(f"  [cycle {iteration + 1}] {sup}")

        if sup.is_stable():
            print(f"  -> stable after {iteration + 1} iteration(s)")
            break

    return sup.collapse()
