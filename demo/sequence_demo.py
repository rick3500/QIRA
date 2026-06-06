"""
QIRA Demo — Sequence: "Find the next number: 1, 4, 9, 16, 25, ?"

Three seed paths attack the problem differently:
  Path A: finite-difference method (second differences)
  Path B: perfect-square pattern recognition
  Path C: algebraic formula search f(n) = n²

All three should converge on 36.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qira import PathState, ThoughtState, TransitionOperator, RuleEngine, InterferenceModule, SpectralMixer, reasoning_cycle
from qira.modules import CompositeConstraint, LengthConstraint, AssumptionConsistencyConstraint, GoalVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has(keyword: str):
    return lambda path: any(keyword.lower() in s.lower() for s in path.trace)

def _lacks(keyword: str):
    return lambda path: not any(keyword.lower() in s.lower() for s in path.trace)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def build_rule_engine() -> RuleEngine:
    engine = RuleEngine()

    # --- Difference path rules ---

    # Rule 1: compute second differences to show they are constant
    engine.add_rule(
        condition=lambda p: _has("first differences")(p) and _lacks("second differences")(p),
        action=lambda p: "Second differences: 5-3=2, 7-5=2, 9-7=2. Constant second difference of 2 — quadratic growth.",
    )

    # Rule 2: use constant second difference to find next first difference, then next term
    engine.add_rule(
        condition=lambda p: _has("second differences")(p) and _lacks("next difference")(p),
        action=lambda p: "Next first difference: 9+2=11. Next term: 25+11=36.",
    )

    # --- Square recognition path rules ---

    # Rule 3: name the index pattern once squares are spotted
    engine.add_rule(
        condition=lambda p: _has("1², 2², 3², 4², 5²")(p) and _lacks("n=6")(p),
        action=lambda p: "The next index is n=6. Therefore the next term is 6²=36.",
    )

    # --- Formula path rules ---

    # Rule 4: verify the formula holds for all known terms
    engine.add_rule(
        condition=lambda p: _has("f(n) = n²")(p) and _lacks("verified")(p),
        action=lambda p: "Verify: f(1)=1 ✓, f(2)=4 ✓, f(3)=9 ✓, f(4)=16 ✓, f(5)=25 ✓. Formula verified.",
    )

    # Rule 5: apply the formula at n=6
    engine.add_rule(
        condition=lambda p: _has("formula verified")(p) and _lacks("f(6)")(p),
        action=lambda p: "Apply: f(6) = 6² = 36.",
    )

    # --- Shared conclusion rule ---

    # Rule 6: state the final answer once 36 is in the trace
    engine.add_rule(
        condition=lambda p: _has("36")(p) and _lacks("answer is 36")(p),
        action=lambda p: "The answer is 36.",
    )

    return engine


# ---------------------------------------------------------------------------
# Seed paths
# ---------------------------------------------------------------------------

def build_thought_state() -> ThoughtState:
    ts = ThoughtState()

    # Path A: finite-difference method
    a = PathState()
    a.assumptions = ["Sequence: 1, 4, 9, 16, 25"]
    a.add_step("First differences: 4-1=3, 9-4=5, 16-9=7, 25-16=9.")
    ts.add_path(a, amplitude=1.0)

    # Path B: perfect-square recognition
    b = PathState()
    b.assumptions = ["Sequence: 1, 4, 9, 16, 25"]
    b.add_step("Observe: 1=1², 4=2², 9=3², 16=4², 25=5². These are perfect squares.")
    b.add_step("Pattern confirmed: term n = 1², 2², 3², 4², 5².")
    ts.add_path(b, amplitude=1.0)

    # Path C: algebraic formula search
    c = PathState()
    c.assumptions = ["Sequence: 1, 4, 9, 16, 25"]
    c.add_step("Hypothesis: f(n) = n².")
    ts.add_path(c, amplitude=1.0)

    ts.normalize()
    return ts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    print("=" * 60)
    print("QIRA Demo — Sequence: 1, 4, 9, 16, 25, ?")
    print("=" * 60)

    thought_state = build_thought_state()
    rule_engine = build_rule_engine()

    constraint = CompositeConstraint([
        LengthConstraint(max_steps=10),
        AssumptionConsistencyConstraint(),
    ])
    verifier = GoalVerifier(goal="36")

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
