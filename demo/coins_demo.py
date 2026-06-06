"""
QIRA Coin Puzzle Demo -- no LLM.

Problem:
  Alex has 20 coins consisting of dimes and nickels.
  Altogether the coins add up to $1.35.
  How many of each coin type does he have?

States:       candidate dime counts d = 0..27 (nickels derived as 20 - d)
              Each is a simultaneous hypothesis held in superposition.
Transitions:  arithmetic constraint operators that hard-eliminate
              candidates that violate a clue.
Interference: invalid candidates cancel (destructive); the valid one
              is boosted each cycle it survives (constructive).
FFT:          smooths amplitude distribution across the candidate ensemble.
Collapse:     the surviving candidate is the answer.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Hypothesis state
# ---------------------------------------------------------------------------

class CoinState:
    """A single candidate: d dimes with derived nickels and total value."""

    def __init__(self, dimes: int) -> None:
        self.dimes   = dimes
        self.nickels = 20 - dimes          # required to make 20 total
        self.value   = 10 * dimes + 5 * (20 - dimes)   # cents
        self.history = [f"Hypothesis: {dimes} dimes, {20 - dimes} nickels"]
        self.score   = 0.0

    def __repr__(self) -> str:
        return (f"CoinState(dimes={self.dimes}, nickels={self.nickels}, "
                f"value={self.value}c)")


# ---------------------------------------------------------------------------
# Superposition of coin hypotheses
# ---------------------------------------------------------------------------

class CoinSuperposition:
    def __init__(self) -> None:
        self.states: list[CoinState] = []
        self.amplitudes: list[float] = []
        self._eliminated: set[int] = set()   # hard mask — survives FFT diffusion

    def add(self, state: CoinState, amplitude: float = 1.0) -> None:
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
        for i in self._eliminated:          # re-apply hard mask after FFT
            self.amplitudes[i] = 0.0
        self.normalize()

    def collapse(self) -> CoinState:
        idx = max(range(len(self.amplitudes)), key=lambda i: abs(self.amplitudes[i]))
        return self.states[idx]

    def snapshot(self) -> str:
        parts = []
        for s, a in zip(self.states, self.amplitudes):
            bar = "#" * int(abs(a) * 40)
            parts.append(f"    d={s.dimes:2d}  {bar:<40}  {a:.3f}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Constraint operators (Markov transitions)
# ---------------------------------------------------------------------------

class CoinConstraint:
    def __init__(self, name: str, check, reason: str) -> None:
        self.name   = name
        self.check  = check
        self.reason = reason

    def apply(self, sup: CoinSuperposition) -> None:
        for i, state in enumerate(sup.states):
            if not self.check(state):
                sup.eliminate(i)
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
    print("QIRA Coin Puzzle Demo")
    print("=" * 60)
    print()
    print("Problem:")
    print("  Alex has 20 coins (dimes and nickels)")
    print("  Total value = $1.35")
    print("  How many of each?")
    print()
    print("Architecture:")
    print("  States      = candidate dime counts d = 0..27 (superposition)")
    print("  Transitions = arithmetic constraint operators")
    print("  Interference= invalid candidates -> amplitude 0 (destructive)")
    print("                valid candidates   -> amplitude x1.2 (constructive)")
    print("  FFT         = amplitude smoothing across candidate ensemble")
    print("  Collapse    = highest-amplitude surviving candidate")
    print()

    # Superposition over all plausible dime counts (up to 27 dimes = $2.70)
    sup = CoinSuperposition()
    for d in range(28):
        sup.add(CoinState(d), amplitude=1.0)
    sup.normalize()

    print("Initial superposition (28 candidates, uniform):")
    print(sup.snapshot())
    print()

    constraints = [
        CoinConstraint(
            "non-negative nickels",
            lambda s: s.nickels >= 0,
            "nickels = 20 - d < 0  (d must be <= 20)"
        ),
        CoinConstraint(
            "value = $1.35",
            lambda s: s.value == 135,
            "10*d + 5*(20-d) must equal 135  =>  5*d = 35  =>  d = 7"
        ),
    ]

    print("Applying constraints:\n")
    for constraint in constraints:
        constraint.apply(sup)
        sup.spectral_mix()

        surviving = [s.dimes for s, a in zip(sup.states, sup.amplitudes) if abs(a) > 1e-6]
        print(f"  After '{constraint.name}':")
        print(f"    Surviving candidates (dimes): {surviving}")
        print(sup.snapshot())
        print()

    result = sup.collapse()

    print("=" * 60)
    print("ANSWER (collapsed state)")
    print("=" * 60)
    print()
    print(f"  Dimes:   {result.dimes:2d}  x $0.10 = ${result.dimes * 10 / 100:.2f}")
    print(f"  Nickels: {result.nickels:2d}  x $0.05 = ${result.nickels * 5 / 100:.2f}")
    print(f"  Total coins: {result.dimes + result.nickels}")
    print(f"  Total value: ${result.value / 100:.2f}")
    print()
    print("Verification:")
    print(f"  Coin count  = {result.dimes} + {result.nickels} = {result.dimes + result.nickels}"
          f"  {'OK' if result.dimes + result.nickels == 20 else 'FAIL'}")
    print(f"  Total value = {result.dimes}*10 + {result.nickels}*5 = {result.value}c"
          f"  {'OK' if result.value == 135 else 'FAIL'}")


if __name__ == "__main__":
    main()
