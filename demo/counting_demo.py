"""
QIRA Counting Puzzle Demo -- no LLM.

Problem:
  How many pets does Mary have?
    - All are dogs except 2
    - All are cats except 2
    - All are rabbits except 2

States:       candidate integers for total pet count (N = 1..10)
              Each is a simultaneous hypothesis held in superposition.
Transitions:  arithmetic constraint operators that set amplitude=0
              for candidates that violate a clue.
Interference: invalid candidates cancel (destructive); the valid one
              is boosted each cycle it survives (constructive).
FFT:          smooths amplitude distribution across the candidate ensemble.
Collapse:     the surviving candidate is the answer.

This is the quantum-inspired architecture applied to arithmetic reasoning:
a superposition of all possible answers, with each constraint acting as a
measurement that collapses the space.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Hypothesis state
# ---------------------------------------------------------------------------

class CountingState:
    """A single candidate answer: N total pets, with derived counts."""

    def __init__(self, n: int, history: list[str] | None = None) -> None:
        self.n = n
        self.dogs    = n - 2   # "all are dogs except 2"
        self.cats    = n - 2   # "all are cats except 2"
        self.rabbits = n - 2   # "all are rabbits except 2"
        self.history = history or [f"Hypothesis: total = {n}"]
        self.score   = 0.0

    def is_valid(self) -> bool:
        return (self.dogs >= 0 and self.cats >= 0 and self.rabbits >= 0 and
                self.dogs + self.cats + self.rabbits == self.n)

    def __repr__(self) -> str:
        return f"N={self.n}(dogs={self.dogs}, cats={self.cats}, rabbits={self.rabbits})"


# ---------------------------------------------------------------------------
# Superposition of counting hypotheses
# ---------------------------------------------------------------------------

class CountingSuperposition:
    def __init__(self) -> None:
        self.states: list[CountingState] = []
        self.amplitudes: list[float] = []
        self._eliminated: set[int] = set()   # hard mask — survives FFT diffusion

    def add(self, state: CountingState, amplitude: float = 1.0) -> None:
        self.states.append(state)
        self.amplitudes.append(amplitude)

    def eliminate(self, i: int) -> None:
        self._eliminated.add(i)
        self.amplitudes[i] = 0.0

    def normalize(self) -> None:
        total = sum(abs(a) for a in self.amplitudes)
        if total:
            self.amplitudes = [a / total for a in self.amplitudes]

    def spectral_mix(self) -> None:
        if len(self.amplitudes) < 2:
            return
        amps = np.array(self.amplitudes, dtype=float)
        kernel = np.array([1 / (1 + i) for i in range(len(amps))])
        smoothed = np.fft.ifft(np.fft.fft(amps) * kernel)
        self.amplitudes = smoothed.real.tolist()
        # Re-apply hard mask — eliminated candidates cannot recover from FFT diffusion
        for i in self._eliminated:
            self.amplitudes[i] = 0.0
        self.normalize()

    def collapse(self) -> CountingState:
        idx = max(range(len(self.amplitudes)), key=lambda i: abs(self.amplitudes[i]))
        return self.states[idx]

    def snapshot(self) -> str:
        parts = []
        for s, a in sorted(zip(self.states, self.amplitudes), key=lambda x: x[0].n):
            bar = "#" * int(abs(a) * 40)
            parts.append(f"    N={s.n:2d}  {bar:<40}  {a:.3f}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Constraint operators (Markov transitions)
# ---------------------------------------------------------------------------

class CountingConstraint:
    def __init__(self, name: str, check, reason: str) -> None:
        self.name   = name
        self.check  = check    # check(state) -> bool
        self.reason = reason

    def apply(self, sup: CountingSuperposition) -> None:
        for i, state in enumerate(sup.states):
            if not self.check(state):
                sup.eliminate(i)   # hard mask — survives FFT diffusion
                if self.reason not in state.history:
                    state.history.append(f"ELIMINATED by {self.name}: {self.reason}")
            else:
                sup.amplitudes[i] *= 1.2   # constructive: survived this constraint
        sup.normalize()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("QIRA Counting Puzzle Demo")
    print("=" * 60)
    print()
    print("Problem:")
    print("  How many pets does Mary have?")
    print("  - All are dogs except 2")
    print("  - All are cats except 2")
    print("  - All are rabbits except 2")
    print()
    print("Architecture:")
    print("  States      = candidate integers N = 1..10 (uniform superposition)")
    print("  Transitions = arithmetic constraint operators")
    print("  Interference= invalid candidates -> amplitude 0 (destructive)")
    print("                valid candidates   -> amplitude x1.2 (constructive)")
    print("  FFT         = amplitude smoothing across candidate ensemble")
    print("  Collapse    = highest-amplitude surviving candidate")
    print()

    # Uniform superposition of all candidate answers
    sup = CountingSuperposition()
    for n in range(1, 11):
        sup.add(CountingState(n), amplitude=1.0)
    sup.normalize()

    print("Initial superposition (uniform):")
    print(sup.snapshot())
    print()

    # Constraints — each is a Markov transition operator
    constraints = [
        CountingConstraint(
            "non-negative dogs",
            lambda s: s.dogs >= 0,
            "dogs = N-2 < 0  (N must be >= 2)"
        ),
        CountingConstraint(
            "non-negative cats",
            lambda s: s.cats >= 0,
            "cats = N-2 < 0  (N must be >= 2)"
        ),
        CountingConstraint(
            "non-negative rabbits",
            lambda s: s.rabbits >= 0,
            "rabbits = N-2 < 0  (N must be >= 2)"
        ),
        CountingConstraint(
            "total conservation",
            lambda s: s.dogs + s.cats + s.rabbits == s.n,
            f"dogs+cats+rabbits = 3*(N-2) must equal N  =>  N=3 only"
        ),
    ]

    print("Applying constraints:\n")
    for constraint in constraints:
        constraint.apply(sup)
        sup.spectral_mix()

        surviving = [s.n for s, a in zip(sup.states, sup.amplitudes) if abs(a) > 1e-6]
        print(f"  After '{constraint.name}':")
        print(f"    Surviving candidates: {surviving}")
        print(sup.snapshot())
        print()

    result = sup.collapse()

    print("=" * 60)
    print("ANSWER (collapsed state)")
    print("=" * 60)
    print()
    print(f"  Mary has {result.n} pets:")
    print(f"    Dogs:    {result.dogs}")
    print(f"    Cats:    {result.cats}")
    print(f"    Rabbits: {result.rabbits}")
    print(f"    Total:   {result.dogs + result.cats + result.rabbits}")
    print()
    print("Derivation:")
    for step in result.history:
        print(f"  {step}")
    print()
    print("Verification:")
    print(f"  All are dogs    except {result.n - result.dogs} = 2  {'OK' if result.n - result.dogs == 2 else 'FAIL'}")
    print(f"  All are cats    except {result.n - result.cats} = 2  {'OK' if result.n - result.cats == 2 else 'FAIL'}")
    print(f"  All are rabbits except {result.n - result.rabbits} = 2  {'OK' if result.n - result.rabbits == 2 else 'FAIL'}")


if __name__ == "__main__":
    main()
