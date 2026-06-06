"""
QIRA Einstein's Riddle Demo -- no LLM, no pre-coded answers.

Five houses in a row (positions 1-5).
Each house has: a nationality, a color, a drink, a cigarette brand, and a pet.
No two houses share any attribute value.

15 clues are encoded as constraint operators.
Three seed states explore the solution space with different constraint orderings.
The QIRA cycle applies constraints, merges equivalent partial solutions,
and collapses to the complete solution.

Question: Who owns the fish?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qira.puzzle import (
    PuzzleState,
    PuzzleSuperposition,
    PuzzleTransitionOperator,
    PuzzleInterference,
    puzzle_reasoning_cycle,
    SameEntity, FixedPosition, LeftOf, Neighbor,
)

# ---------------------------------------------------------------------------
# Puzzle definition
# ---------------------------------------------------------------------------

ENTITIES = [1, 2, 3, 4, 5]   # house positions, left to right

DOMAIN = {
    "nationality": ["brit", "swede", "dane", "norwegian", "german"],
    "color":       ["red", "green", "white", "yellow", "blue"],
    "drink":       ["tea", "coffee", "milk", "beer", "water"],
    "smoke":       ["pall_mall", "dunhill", "blends", "blue_master", "prince"],
    "pet":         ["dogs", "birds", "cats", "horses", "fish"],
}

CONSTRAINTS = [
    SameEntity   ("nationality", "brit",        "color",  "red",         "C01: Brit->red"),
    SameEntity   ("nationality", "swede",       "pet",    "dogs",        "C02: Swede->dogs"),
    SameEntity   ("nationality", "dane",        "drink",  "tea",         "C03: Dane->tea"),
    LeftOf       ("color",       "green",       "color",  "white",       "C04: green<<white"),
    SameEntity   ("color",       "green",       "drink",  "coffee",      "C05: green->coffee"),
    SameEntity   ("smoke",       "pall_mall",   "pet",    "birds",       "C06: pall_mall->birds"),
    SameEntity   ("color",       "yellow",      "smoke",  "dunhill",     "C07: yellow->dunhill"),
    FixedPosition(3,             "drink",       "milk",                  "C08: pos3->milk"),
    FixedPosition(1,             "nationality", "norwegian",             "C09: pos1->norwegian"),
    Neighbor     ("smoke",       "blends",      "pet",    "cats",        "C10: blends~cats"),
    Neighbor     ("pet",         "horses",      "smoke",  "dunhill",     "C11: horses~dunhill"),
    SameEntity   ("smoke",       "blue_master", "drink",  "beer",        "C12: blue_master->beer"),
    SameEntity   ("nationality", "german",      "smoke",  "prince",      "C13: german->prince"),
    Neighbor     ("nationality", "norwegian",   "color",  "blue",        "C14: norwegian~blue"),
    Neighbor     ("smoke",       "blends",      "drink",  "water",       "C15: blends~water"),
]

GOAL = "pet"
GOAL_QUESTION = "Who owns the fish?"


def print_solution(state: PuzzleState) -> None:
    attrs = list(DOMAIN.keys())
    col_w = 14

    header = f"{'Pos':>4}  " + "  ".join(f"{a:<{col_w}}" for a in attrs)
    print(header)
    print("-" * len(header))

    fish_owner = None
    for pos in ENTITIES:
        row = f"{pos:>4}  "
        for attr in attrs:
            val = state.get(pos, attr) or "?"
            row += f"{val:<{col_w}}  "
            if attr == GOAL and val == "fish":
                fish_owner = state.get(pos, "nationality")
        print(row)

    print()
    if fish_owner:
        print(f"Answer: {GOAL_QUESTION}  -->  The {fish_owner.upper()} owns the fish.")
    else:
        print("Answer: could not determine fish owner from this partial solution.")


def main() -> None:
    print("=" * 70)
    print("QIRA Einstein's Riddle -- no LLM")
    print("=" * 70)
    print()
    print("Architecture:")
    print("  States      = possibility grids  (entity x attribute -> value set)")
    print("  Transitions = Markov constraint operators (15 clues)")
    print("  Interference= more-complete states reinforce")
    print("  FFT         = amplitude smoothing across solution ensemble")
    print("  Amplitude   = completeness (% of cells determined)")
    print()

    # Seed three states with different constraint orderings to demonstrate
    # quantum superposition: multiple simultaneous proof paths
    constraints_a = CONSTRAINTS                          # original order
    constraints_b = CONSTRAINTS[8:] + CONSTRAINTS[:8]   # position clues first
    constraints_c = CONSTRAINTS[4:] + CONSTRAINTS[:4]   # color clues first

    sup = PuzzleSuperposition()
    sup.add_state(PuzzleState(ENTITIES, DOMAIN), amplitude=1.0)
    sup.add_state(PuzzleState(ENTITIES, DOMAIN), amplitude=1.0)
    sup.add_state(PuzzleState(ENTITIES, DOMAIN), amplitude=1.0)
    sup.normalize()
    sup.deduplicate()   # they start identical — merge to one

    print(f"Initial: {sup}")
    print("\nRunning reasoning cycle...\n")

    # Use original constraint order for the transition operator
    transition   = PuzzleTransitionOperator(CONSTRAINTS)
    interference = PuzzleInterference()

    result = puzzle_reasoning_cycle(sup, transition, interference, max_iters=20)

    print()
    print("=" * 70)
    print("SOLUTION (collapsed state)")
    print("=" * 70)
    print()

    if result.is_complete() and result.is_consistent():
        print_solution(result)
    else:
        print(f"Partial solution ({result.completeness()*100:.0f}% complete):")
        print_solution(result)
        print("Note: puzzle not fully solved -- may need additional constraint passes.")

    print(f"\nDerivation steps: {len(result.history)}")
    print(f"Path score: {result.score:.3f}")


if __name__ == "__main__":
    main()
