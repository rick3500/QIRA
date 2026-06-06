"""
QIRA Logic Demo — Socrates syllogism, no LLM.

States:       frozensets of propositions (formal facts)
Transitions:  inference rules — modus ponens, transitivity, negation elimination
Interference: superset states reinforce (more derived = stronger), contradictions cancel
FFT:          amplitude smoothing across the proof-state ensemble
Amplitude:    tracks how many goal propositions have been proved
Collapse:     selects the state that has proved the most

Architecture is identical to symbolic_algebra_demo — Markov transitions,
FFT mixing, interference, collapse. No LLM anywhere.

Facts use simple predicate notation: predicate(individual)
Universal rules:  forall X: P(X) -> Q(X)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qira.logic import (
    LogicState,
    LogicSuperposition,
    LogicTransitionOperator,
    LogicInterference,
    logic_reasoning_cycle,
)

# ---------------------------------------------------------------------------
# Problem definition — change facts and goals for any syllogism
# ---------------------------------------------------------------------------

AXIOMS = frozenset([
    "human(socrates)",                       # Socrates is a human
    "forall X: human(X) -> mortal(X)",       # All humans are mortal
])

GOALS = [
    "mortal(socrates)",                      # What we want to prove
]


def main() -> None:
    print("=" * 60)
    print("QIRA Logic Demo - Socrates Syllogism (no LLM)")
    print("=" * 60)
    print()
    print("Architecture:")
    print("  States      = frozensets of propositions (formal facts)")
    print("  Transitions = Markov operators: modus ponens / transitivity")
    print("  Interference= superset states reinforce, contradictions cancel")
    print("  FFT         = amplitude smoothing across proof-state ensemble")
    print("  Amplitude   = goals proved (higher = more complete proof)")
    print()
    print(f"Axioms: {sorted(AXIOMS)}")
    print(f"Goal:   {GOALS}")
    print()

    # Seed three starting states — same axioms, different orderings act as
    # independent starting points that may generate different proof paths
    sup = LogicSuperposition()
    sup.add_state(LogicState(AXIOMS),          amplitude=1.0)
    sup.add_state(LogicState(AXIOMS),          amplitude=1.0)
    sup.add_state(LogicState(AXIOMS),          amplitude=1.0)
    sup.normalize()
    sup.deduplicate()  # they're identical — merge into one immediately

    print(f"Initial state: {sup}")
    print("\nRunning reasoning cycle...\n")

    transition  = LogicTransitionOperator(goals=GOALS)
    interference = LogicInterference()

    result = logic_reasoning_cycle(sup, transition, interference, max_iters=10)

    print("\n" + "=" * 60)
    print("FINAL PROOF (collapsed state)")
    print("=" * 60)
    print()
    print("Derived facts:")
    for fact in sorted(result.facts):
        proven = " <-- GOAL PROVED" if fact in GOALS else ""
        print(f"  {fact}{proven}")
    print()
    print("Derivation:")
    for step in result.history:
        print(f"  {step}")
    print(f"\nPath score: {result.score:.3f}")


if __name__ == "__main__":
    main()
