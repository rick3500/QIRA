"""
QIRA Demo — Algebra: "Simplify the expression: 2(x + 3) + 4x"

Three seed paths attack the problem differently:
  Path A: distribute first, then collect like terms
  Path B: identify like terms conceptually before expanding
  Path C: step-by-step term-by-term accounting

All three should converge on 6x + 6.
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

    # Rule 1: apply the distributive property to expand 2(x + 3)
    engine.add_rule(
        condition=lambda p: _has("distributive")(p) and _lacks("2x + 6")(p),
        action=lambda p: "Apply distributive property: 2(x + 3) = 2x + 6. Expression becomes: 2x + 6 + 4x.",
    )

    # Rule 2: after expanding, collect x terms
    engine.add_rule(
        condition=lambda p: _has("2x + 6 + 4x")(p) and _lacks("6x")(p),
        action=lambda p: "Collect like terms: 2x + 4x = 6x.",
    )

    # Rule 3: after collecting x terms, write the constant
    engine.add_rule(
        condition=lambda p: _has("6x")(p) and _lacks("6x + 6")(p),
        action=lambda p: "Simplified expression: 6x + 6.",
    )

    # Rule 4: group x-terms conceptually before expanding (Path B approach)
    engine.add_rule(
        condition=lambda p: _has("x terms")(p) and _lacks("2x + 6")(p),
        action=lambda p: "Expand 2(x + 3) = 2x + 6, giving x-terms: 2x + 4x = 6x and constant: 6.",
    )

    # Rule 5: term-by-term accounting (Path C approach)
    engine.add_rule(
        condition=lambda p: _has("term-by-term")(p) and _lacks("constant term")(p),
        action=lambda p: "Constant term from 2(x+3): 2×3 = 6. Variable terms: 2×x + 4x = 2x + 4x.",
    )

    engine.add_rule(
        condition=lambda p: _has("2x + 4x")(p) and _lacks("6x")(p),
        action=lambda p: "Combine variable terms: 2x + 4x = 6x. Simplified expression: 6x + 6.",
    )

    # Rule 6: state the final answer once simplified form is in the trace
    engine.add_rule(
        condition=lambda p: _has("6x + 6")(p) and _lacks("the answer is")(p),
        action=lambda p: "The answer is 6x + 6.",
    )

    return engine


# ---------------------------------------------------------------------------
# Seed paths
# ---------------------------------------------------------------------------

def build_thought_state() -> ThoughtState:
    ts = ThoughtState()

    # Path A: distribute first
    a = PathState()
    a.assumptions = ["Expression: 2(x + 3) + 4x"]
    a.add_step("Strategy: apply the distributive property first, then collect like terms.")
    a.add_step("Identify: 2(x + 3) requires the distributive property.")
    ts.add_path(a, amplitude=1.0)

    # Path B: group like terms conceptually before expanding
    b = PathState()
    b.assumptions = ["Expression: 2(x + 3) + 4x"]
    b.add_step("Strategy: identify x terms and constant terms before expanding.")
    b.add_step("The expression has x terms from 2(x+3) and 4x, plus a constant from 2(x+3).")
    ts.add_path(b, amplitude=1.0)

    # Path C: term-by-term accounting
    c = PathState()
    c.assumptions = ["Expression: 2(x + 3) + 4x"]
    c.add_step("Strategy: process term-by-term — variable part and constant part separately.")
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
    print("QIRA Demo — Algebra: Simplify 2(x + 3) + 4x")
    print("=" * 60)

    thought_state = build_thought_state()
    rule_engine = build_rule_engine()

    constraint = CompositeConstraint([
        LengthConstraint(max_steps=10),
        AssumptionConsistencyConstraint(),
    ])
    verifier = GoalVerifier(goal="6x + 6")

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
