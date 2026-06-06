"""
QIRA Dynamic Demo — no pre-coded rules or answers.

The LLMExpander generates reasoning steps on the fly.
The LLMVerifier checks completeness without knowing the answer.
The InterferenceModule prunes contradicting paths and boosts agreement.

Change PROBLEM to anything — the system figures it out.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qira import (
    PathState, ThoughtState, TransitionOperator,
    InterferenceModule, SpectralMixer, reasoning_cycle,
    LLMExpander, LLMVerifier,
)
from qira.modules import LengthConstraint, AssumptionConsistencyConstraint, CompositeConstraint

# ---------------------------------------------------------------------------
# Change this to any problem — no other code changes needed
# ---------------------------------------------------------------------------
PROBLEM = "Simplify the expression: 2(x + 3) + 4x"


def build_thought_state(problem: str) -> ThoughtState:
    """Seed three paths with different strategic starting points — no answer pre-coded."""
    ts = ThoughtState()

    approaches = [
        "I will work inside-out, handling the parentheses first.",
        "I will identify and group all like terms before simplifying.",
        "I will expand every term individually then collect results.",
    ]
    for approach in approaches:
        p = PathState()
        p.assumptions = [f"Problem: {problem}"]
        p.add_step(approach)
        ts.add_path(p, amplitude=1.0)

    ts.normalize()
    return ts


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    print("=" * 60)
    print(f"QIRA Dynamic Demo")
    print(f"Problem: {PROBLEM}")
    print("=" * 60)

    thought_state = build_thought_state(PROBLEM)
    expander = LLMExpander(problem=PROBLEM, n_proposals=2)
    verifier = LLMVerifier(problem=PROBLEM)

    constraint = CompositeConstraint([
        LengthConstraint(max_steps=8),
        AssumptionConsistencyConstraint(),
    ])

    transition = TransitionOperator(constraint, verifier, expander)
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


if __name__ == "__main__":
    main()
