"""
QIRA Murder Mystery Demo -- no LLM.

Problem:
  5 people were questioned by the police. One murdered a romantic rival.
  Statements:
    EILEEN:  "Earl is the murderer"
    EMMA:    "Earl is the murderer"
    EUGENIA: "Earl is, without a shadow of a doubt, the murderer"
    EDWIN:   "Emma lied"
    EARL:    "Emma told the truth"
  4 statements are lies, 1 is true -- and strangely, the true one was
  made by the murderer.

States:       one hypothesis per suspect (5 candidates in superposition)
Transitions:  logical constraint operators that evaluate each statement's
              truth value under the assumption that a given suspect is guilty
Interference: inconsistent hypotheses cancel (destructive); the consistent
              one is boosted (constructive)
FFT:          smooths amplitude distribution across the candidate ensemble
Collapse:     the surviving candidate is the murderer
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SUSPECTS = ["Eileen", "Emma", "Eugenia", "Edwin", "Earl"]


# ---------------------------------------------------------------------------
# Hypothesis state
# ---------------------------------------------------------------------------

class MurderState:
    """
    One hypothesis: a specific suspect is the murderer.

    Given that assumption, every statement's truth value is determined
    mechanically -- no LLM, no lookup table.
    """

    def __init__(self, suspect: str) -> None:
        self.suspect = suspect
        self.history = [f"Hypothesis: {suspect} is the murderer"]
        self.score   = 0.0

        # Derive truth value of each statement from the hypothesis
        earl_guilty     = (suspect == "Earl")
        emma_told_truth = earl_guilty   # Emma said "Earl is the murderer"

        self.truths: dict[str, bool] = {
            "Eileen" : earl_guilty,           # "Earl is the murderer"
            "Emma"   : earl_guilty,           # "Earl is the murderer"
            "Eugenia": earl_guilty,           # "Earl is the murderer"
            "Edwin"  : not emma_told_truth,   # "Emma lied"
            "Earl"   : emma_told_truth,       # "Emma told the truth"
        }

    @property
    def murderer_told_truth(self) -> bool:
        return self.truths[self.suspect]

    @property
    def truth_count(self) -> int:
        return sum(self.truths.values())

    def __repr__(self) -> str:
        truth_tellers = [p for p, t in self.truths.items() if t]
        return f"MurderState({self.suspect}, truth_tellers={truth_tellers})"


# ---------------------------------------------------------------------------
# Superposition of murder hypotheses
# ---------------------------------------------------------------------------

class MurderSuperposition:
    def __init__(self) -> None:
        self.states: list[MurderState] = []
        self.amplitudes: list[float]   = []
        self._eliminated: set[int]     = set()

    def add(self, state: MurderState, amplitude: float = 1.0) -> None:
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
        for i in self._eliminated:      # re-apply hard mask after FFT diffusion
            self.amplitudes[i] = 0.0
        self.normalize()

    def collapse(self) -> MurderState:
        idx = max(range(len(self.amplitudes)), key=lambda i: abs(self.amplitudes[i]))
        return self.states[idx]

    def snapshot(self) -> str:
        parts = []
        for s, a in zip(self.states, self.amplitudes):
            bar = "#" * int(abs(a) * 40)
            parts.append(f"    {s.suspect:8s}  {bar:<40}  {a:.3f}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Constraint operators
# ---------------------------------------------------------------------------

class MurderConstraint:
    def __init__(self, name: str, check, reason: str) -> None:
        self.name   = name
        self.check  = check
        self.reason = reason

    def apply(self, sup: MurderSuperposition) -> None:
        for i, state in enumerate(sup.states):
            if not self.check(state):
                sup.eliminate(i)
                state.history.append(f"ELIMINATED by '{self.name}': {self.reason}")
            else:
                sup.amplitudes[i] *= 1.3   # constructive: hypothesis survived
        sup.normalize()


# ---------------------------------------------------------------------------
# Truth-table display
# ---------------------------------------------------------------------------

STATEMENTS = {
    "Eileen" : 'Eileen  says: "Earl is the murderer"',
    "Emma"   : 'Emma    says: "Earl is the murderer"',
    "Eugenia": 'Eugenia says: "Earl is, without a shadow of a doubt, the murderer"',
    "Edwin"  : 'Edwin   says: "Emma lied"',
    "Earl"   : 'Earl    says: "Emma told the truth"',
}

def print_truth_table(state: MurderState) -> None:
    print(f"  Assuming {state.suspect} is the murderer:")
    for person, stmt in STATEMENTS.items():
        verdict = "TRUTH" if state.truths[person] else "lie  "
        marker  = " <-- murderer's statement" if person == state.suspect else ""
        print(f"    [{verdict}]  {stmt}{marker}")
    print(f"  Truth count = {state.truth_count}  |  "
          f"Murderer told truth = {state.murderer_told_truth}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("QIRA Murder Mystery Demo")
    print("=" * 65)
    print()
    print("Statements:")
    for stmt in STATEMENTS.values():
        print(f"  {stmt}")
    print()
    print("Rules:")
    print("  - Exactly 1 statement is true, 4 are lies")
    print("  - The true statement was made by the murderer")
    print()
    print("Architecture:")
    print("  States      = 5 suspect hypotheses (uniform superposition)")
    print("  Transitions = constraint operators that evaluate statement")
    print("                truth values under each hypothesis")
    print("  Interference= inconsistent hypotheses -> amplitude 0")
    print("                consistent hypothesis   -> amplitude x1.3")
    print("  FFT         = amplitude smoothing across candidates")
    print("  Collapse    = highest-amplitude surviving hypothesis")
    print()

    # Uniform superposition over all 5 suspects
    sup = MurderSuperposition()
    for suspect in SUSPECTS:
        sup.add(MurderState(suspect), amplitude=1.0)
    sup.normalize()

    print("Initial superposition (all suspects equally likely):")
    print(sup.snapshot())
    print()

    print("Truth table under each hypothesis:")
    for s in sup.states:
        print_truth_table(s)

    constraints = [
        MurderConstraint(
            "murderer told the truth",
            lambda s: s.murderer_told_truth,
            "the murderer's own statement must be the true one"
        ),
        MurderConstraint(
            "exactly 1 truth",
            lambda s: s.truth_count == 1,
            "exactly 4 lies and 1 truth total"
        ),
    ]

    print("Applying constraints:\n")
    for constraint in constraints:
        constraint.apply(sup)
        sup.spectral_mix()

        surviving = [s.suspect for s, a in zip(sup.states, sup.amplitudes) if abs(a) > 1e-6]
        print(f"  After '{constraint.name}':")
        print(f"    Surviving suspects: {surviving}")
        print(sup.snapshot())
        print()

    result = sup.collapse()

    print("=" * 65)
    print("VERDICT (collapsed state)")
    print("=" * 65)
    print()
    print_truth_table(result)
    print(f"  The murderer is: {result.suspect.upper()}")
    print()
    print("Reasoning:")
    for step in result.history:
        print(f"  {step}")


if __name__ == "__main__":
    main()
