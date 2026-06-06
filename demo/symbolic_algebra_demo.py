"""
QIRA Symbolic Algebra Demo — no LLM, no pre-coded answers.

States:       sympy expressions (the actual math)
Transitions:  algebraic operators — expand, simplify, factor, collect, cancel
Interference: symbolic equivalence via sympy — equivalent paths amplify each other
FFT:          smooths amplitude distribution across the path ensemble
Amplitude:    tracks simplification progress — lower complexity = higher weight
Collapse:     selects the most dominant (simplest) surviving path

This is the Markov chain + quantum-inspired architecture from the blueprint.
Change PROBLEM to any algebraic expression.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy import expand, factor, simplify, pretty

from qira.symbolic import (
    SymbolicState,
    SymbolicSuperposition,
    AlgebraTransitionOperator,
    SymbolicInterference,
    symbolic_reasoning_cycle,
)

# ---------------------------------------------------------------------------
# Change this — no other code changes needed
# ---------------------------------------------------------------------------
PROBLEM = "2*(x + 3) + 4*x"


def main() -> None:
    print("=" * 60)
    print(f"QIRA Symbolic Demo")
    print(f"Problem: {PROBLEM}")
    print("=" * 60)
    print()
    print("Architecture:")
    print("  States      = sympy expressions (real math, not text)")
    print("  Transitions = Markov operators: expand / simplify / factor / collect / cancel")
    print("  Interference= symbolic equivalence (no LLM)")
    print("  FFT         = amplitude smoothing across path ensemble")
    print("  Amplitude   = simplification progress (complexity reduction)")
    print()

    transforms = standard_transformations + (implicit_multiplication_application,)
    expr = parse_expr(PROBLEM, transformations=transforms)

    # Seed the superposition with three starting points:
    # the raw expression, an expanded form, and a factored form.
    # Each is a valid entry point — the system finds the simplest.
    sup = SymbolicSuperposition()
    sup.add_state(SymbolicState(expr,         [f"initial       ->  {expr}"]),         amplitude=1.0)
    sup.add_state(SymbolicState(expand(expr), [f"expand        ->  {expand(expr)}"]), amplitude=1.0)
    sup.add_state(SymbolicState(factor(expr), [f"factor        ->  {factor(expr)}"]), amplitude=1.0)
    sup.normalize()

    print(f"Initial state: {sup}")
    print("\nRunning reasoning cycle...\n")

    transition  = AlgebraTransitionOperator()
    interference = SymbolicInterference()

    result = symbolic_reasoning_cycle(sup, transition, interference, max_iters=10)

    print("\n" + "=" * 60)
    print("FINAL ANSWER (collapsed path)")
    print("=" * 60)
    print(f"\n  {pretty(result.expr, use_unicode=False)}\n")
    print("Derivation history:")
    for step in result.history:
        print(f"  {step}")
    print(f"\nComplexity (operation count): {result.complexity()}")
    print(f"Path score: {result.score:.3f}")


if __name__ == "__main__":
    main()
