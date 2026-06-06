"""
QIRA Demo — Syllogism: "All humans are mortal. Socrates is a human. Prove Socrates is mortal."

Three seed paths explore the same problem from different starting angles.
Four forward-chaining rules expand each path deterministically.
The InterferenceModule (LLM-judged) prunes contradictions and boosts agreement.
The cycle collapses to the highest-amplitude path as the final answer.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qira import PathState, ThoughtState, TransitionOperator, RuleEngine, InterferenceModule, SpectralMixer, reasoning_cycle
from qira.modules import CompositeConstraint, LengthConstraint, AssumptionConsistencyConstraint, GoalVerifier


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def _has_step(keyword: str):
    return lambda path: any(keyword.lower() in s.lower() for s in path.trace)

def _lacks_step(keyword: str):
    return lambda path: not any(keyword.lower() in s.lower() for s in path.trace)


def build_rule_engine() -> RuleEngine:
    engine = RuleEngine()

    # Rule 1: if we know the universal, assert the category membership
    engine.add_rule(
        condition=lambda p: _has_step("all humans are mortal")(p) and _lacks_step("socrates is a human")(p),
        action=lambda p: "Socrates is a human (given premise).",
    )

    # Rule 2: if we know Socrates is human and the universal holds, apply modus ponens
    engine.add_rule(
        condition=lambda p: _has_step("socrates is a human")(p) and _has_step("all humans are mortal")(p) and _lacks_step("socrates is mortal")(p),
        action=lambda p: "By modus ponens: Socrates is mortal.",
    )

    # Rule 3: if we reached the conclusion, state it explicitly
    engine.add_rule(
        condition=lambda p: _has_step("socrates is mortal")(p) and _lacks_step("therefore")(p),
        action=lambda p: "Therefore, Socrates is mortal. QED.",
    )

    # Rule 4: inductive backup — observe the pattern from the universal
    engine.add_rule(
        condition=lambda p: _lacks_step("all humans are mortal")(p) and _has_step("socrates is a human")(p),
        action=lambda p: "All humans are mortal (universal premise).",
    )

    return engine


# ---------------------------------------------------------------------------
# Seed paths
# ---------------------------------------------------------------------------

def build_thought_state() -> ThoughtState:
    ts = ThoughtState()

    # Path A: deductive — start with the universal
    a = PathState()
    a.assumptions = ["All humans are mortal", "Socrates is a human"]
    a.add_step("All humans are mortal (universal premise).")
    ts.add_path(a, amplitude=1.0)

    # Path B: subject-first — start with Socrates
    b = PathState()
    b.assumptions = ["All humans are mortal", "Socrates is a human"]
    b.add_step("Socrates is a human (given premise).")
    ts.add_path(b, amplitude=1.0)

    # Path C: assumption-first — state both premises together
    c = PathState()
    c.assumptions = ["All humans are mortal", "Socrates is a human"]
    c.add_step("All humans are mortal (universal premise).")
    c.add_step("Socrates is a human (given premise).")
    ts.add_path(c, amplitude=1.0)

    ts.normalize()
    return ts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. Export it before running this demo.")
        sys.exit(1)

    print("=" * 60)
    print("QIRA Demo — Socrates Syllogism")
    print("=" * 60)

    thought_state = build_thought_state()
    rule_engine = build_rule_engine()

    constraint = CompositeConstraint([
        LengthConstraint(max_steps=8),
        AssumptionConsistencyConstraint(),
    ])
    verifier = GoalVerifier(goal="socrates is mortal")

    transition = TransitionOperator(constraint, verifier, rule_engine)
    interference = InterferenceModule()
    mixer = SpectralMixer()

    print(f"\nInitial state: {thought_state}")
    print("\nRunning reasoning cycle...\n")

    result = reasoning_cycle(thought_state, transition, interference, mixer, max_iters=10)

    print("\n" + "=" * 60)
    print("FINAL REASONING TRACE (collapsed path)")
    print("=" * 60)
    for i, step in enumerate(result.trace, 1):
        print(f"  {i}. {step}")
    print(f"\nPath score: {result.score:.3f}")
    print(f"Assumptions: {result.assumptions}")


if __name__ == "__main__":
    main()
