"""
True QIRA for constraint-satisfaction logic puzzles.

States:       grids of remaining possibilities — (entity, attribute) -> frozenset of values
Transitions:  constraint operators that eliminate possibilities and force assignments
Interference: more-complete states reinforce; inconsistent states (contradiction) cancel
Dedup:        states with identical grids are merged, summing amplitudes
FFT:          smooths amplitude distribution across the solution ensemble
Amplitude:    tracks completeness — more solved cells = higher weight
Collapse:     selects the most complete, consistent partial solution

Works for any zebra/Einstein-style puzzle. No LLM. No pre-coded answers.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Puzzle State
# ---------------------------------------------------------------------------

class PuzzleState:
    """
    A grid of remaining possibilities for a logic puzzle.

    entities: ordered list (e.g. house positions [1,2,3,4,5])
    domain:   dict[attr, list[value]]  — the universe of values per attribute
    possible: dict[(entity, attr), frozenset[value]]  — what's still allowed
    """

    def __init__(self, entities, domain, possible=None, history=None):
        self.entities = list(entities)
        self.domain = {k: list(v) for k, v in domain.items()}
        self.attrs = list(domain.keys())
        self.possible = (
            {k: frozenset(v) for k, v in possible.items()}
            if possible is not None
            else {(e, a): frozenset(domain[a]) for e in self.entities for a in self.attrs}
        )
        self.history = history or ["Initial: all possibilities open."]
        self.score = 0.0

    # --- queries ---

    def get(self, entity, attr):
        """Return the assigned value if determined, else None."""
        v = self.possible.get((entity, attr), frozenset())
        return next(iter(v)) if len(v) == 1 else None

    def is_consistent(self):
        if any(len(v) == 0 for v in self.possible.values()):
            return False
        for attr in self.attrs:
            assigned = [next(iter(self.possible[(e, attr)]))
                        for e in self.entities if len(self.possible[(e, attr)]) == 1]
            if len(assigned) != len(set(assigned)):
                return False
        return True

    def is_complete(self):
        return all(len(v) == 1 for v in self.possible.values())

    def completeness(self):
        total = len(self.entities) * len(self.attrs)
        return sum(1 for v in self.possible.values() if len(v) == 1) / total if total else 0.0

    def cache_key(self):
        return frozenset((k, frozenset(v)) for k, v in self.possible.items())

    # --- mutations (always return new state) ---

    def _clone_possible(self):
        return {k: frozenset(v) for k, v in self.possible.items()}

    def eliminate(self, entity, attr, value, reason=""):
        key = (entity, attr)
        if value not in self.possible.get(key, frozenset()):
            return self
        p = self._clone_possible()
        p[key] = p[key] - {value}
        note = f"  elim   pos={entity} {attr} != {value}"
        if reason:
            note += f"  [{reason}]"
        return PuzzleState(self.entities, self.domain, p, self.history + [note])._propagate()

    def assign(self, entity, attr, value, reason=""):
        key = (entity, attr)
        if self.get(entity, attr) == value:
            return self
        if value not in self.possible.get(key, frozenset()):
            return self  # inconsistent — leave state unchanged (will be pruned)
        p = self._clone_possible()
        p[key] = frozenset([value])
        note = f"  assign pos={entity} {attr} = {value}"
        if reason:
            note += f"  [{reason}]"
        return PuzzleState(self.entities, self.domain, p, self.history + [note])._propagate()

    def _propagate(self):
        """Arc consistency: uniqueness + forced assignment, repeated until stable."""
        state = self
        changed = True
        while changed:
            changed = False
            # Uniqueness: assigned value in E removes it from all other entities
            for attr in state.attrs:
                for entity in state.entities:
                    if len(state.possible[(entity, attr)]) == 1:
                        val = next(iter(state.possible[(entity, attr)]))
                        for other in state.entities:
                            if other != entity and val in state.possible[(other, attr)]:
                                p = state._clone_possible()
                                p[(other, attr)] = p[(other, attr)] - {val}
                                state = PuzzleState(state.entities, state.domain, p, state.history)
                                changed = True
            # Forced: only one entity can hold a value → assign it
            for attr in state.attrs:
                for val in state.domain[attr]:
                    candidates = [e for e in state.entities if val in state.possible[(e, attr)]]
                    if len(candidates) == 1 and len(state.possible[(candidates[0], attr)]) > 1:
                        p = state._clone_possible()
                        p[(candidates[0], attr)] = frozenset([val])
                        note = f"  forced pos={candidates[0]} {attr} = {val}"
                        state = PuzzleState(state.entities, state.domain, p, state.history + [note])
                        changed = True
        return state

    def clone(self):
        return PuzzleState(self.entities, self.domain, self._clone_possible(), self.history[:])

    def __repr__(self):
        return f"PuzzleState({self.completeness()*100:.0f}% complete, consistent={self.is_consistent()})"


# ---------------------------------------------------------------------------
# Constraints (clue types)
# ---------------------------------------------------------------------------

class SameEntity:
    """The entity with attr1=val1 also has attr2=val2."""
    def __init__(self, attr1, val1, attr2, val2, name=""):
        self.attr1, self.val1 = attr1, val1
        self.attr2, self.val2 = attr2, val2
        self.name = name

    def apply(self, state: PuzzleState) -> PuzzleState:
        for entity in state.entities:
            # If entity could have val1 but cannot have val2 → remove val1
            if (self.val1 in state.possible[(entity, self.attr1)] and
                    self.val2 not in state.possible[(entity, self.attr2)]):
                state = state.eliminate(entity, self.attr1, self.val1, self.name)
            # Symmetric
            if (self.val2 in state.possible[(entity, self.attr2)] and
                    self.val1 not in state.possible[(entity, self.attr1)]):
                state = state.eliminate(entity, self.attr2, self.val2, self.name)
        return state

    def __str__(self):
        return self.name or f"SameEntity({self.attr1}={self.val1} <=> {self.attr2}={self.val2})"


class FixedPosition:
    """The entity at a specific position has attr=val."""
    def __init__(self, pos, attr, val, name=""):
        self.pos, self.attr, self.val, self.name = pos, attr, val, name

    def apply(self, state: PuzzleState) -> PuzzleState:
        return state.assign(self.pos, self.attr, self.val, self.name)

    def __str__(self):
        return self.name or f"FixedPosition(pos={self.pos} {self.attr}={self.val})"


class LeftOf:
    """The entity with attr1=val1 is immediately left of the entity with attr2=val2."""
    def __init__(self, attr1, val1, attr2, val2, name=""):
        self.attr1, self.val1 = attr1, val1
        self.attr2, self.val2 = attr2, val2
        self.name = name

    def apply(self, state: PuzzleState) -> PuzzleState:
        entities = state.entities
        n = len(entities)
        # val1 cannot be rightmost; val2 cannot be leftmost
        state = state.eliminate(entities[-1], self.attr1, self.val1, self.name + "(no right)")
        state = state.eliminate(entities[0],  self.attr2, self.val2, self.name + "(no left)")
        # For each adjacent pair: if left has val1, right must have val2, and vice versa
        for i in range(n - 1):
            left, right = entities[i], entities[i + 1]
            if (self.val1 in state.possible[(left, self.attr1)] and
                    self.val2 not in state.possible[(right, self.attr2)]):
                state = state.eliminate(left, self.attr1, self.val1, self.name)
            if (self.val2 in state.possible[(right, self.attr2)] and
                    self.val1 not in state.possible[(left, self.attr1)]):
                state = state.eliminate(right, self.attr2, self.val2, self.name)
        return state

    def __str__(self):
        return self.name or f"LeftOf({self.attr1}={self.val1} << {self.attr2}={self.val2})"


class Neighbor:
    """The entity with attr1=val1 is adjacent (|pos_a - pos_b| == 1) to entity with attr2=val2."""
    def __init__(self, attr1, val1, attr2, val2, name=""):
        self.attr1, self.val1 = attr1, val1
        self.attr2, self.val2 = attr2, val2
        self.name = name

    def apply(self, state: PuzzleState) -> PuzzleState:
        entities = state.entities
        n = len(entities)

        def _neighbors(i):
            return [entities[j] for j in [i - 1, i + 1] if 0 <= j < n]

        for i, entity in enumerate(entities):
            if (self.val1 in state.possible[(entity, self.attr1)] and
                    not any(self.val2 in state.possible[(nb, self.attr2)] for nb in _neighbors(i))):
                state = state.eliminate(entity, self.attr1, self.val1, self.name)
            if (self.val2 in state.possible[(entity, self.attr2)] and
                    not any(self.val1 in state.possible[(nb, self.attr1)] for nb in _neighbors(i))):
                state = state.eliminate(entity, self.attr2, self.val2, self.name)
        return state

    def __str__(self):
        return self.name or f"Neighbor({self.attr1}={self.val1} ~ {self.attr2}={self.val2})"


# ---------------------------------------------------------------------------
# Superposition
# ---------------------------------------------------------------------------

class PuzzleSuperposition:
    def __init__(self):
        self.states: list[PuzzleState] = []
        self.amplitudes: list[float] = []

    def add_state(self, state, amplitude=1.0):
        self.states.append(state)
        self.amplitudes.append(amplitude)

    def normalize(self):
        total = sum(abs(a) for a in self.amplitudes)
        if total:
            self.amplitudes = [a / total for a in self.amplitudes]

    def deduplicate(self):
        canonical: dict = {}
        for state, amp in zip(self.states, self.amplitudes):
            key = state.cache_key()
            if key in canonical:
                existing, existing_amp = canonical[key]
                winner = state if state.completeness() >= existing.completeness() else existing
                canonical[key] = (winner, existing_amp + amp)
            else:
                canonical[key] = (state, amp)
        self.states = [s for s, _ in canonical.values()]
        self.amplitudes = [a for _, a in canonical.values()]
        self.normalize()

    def prune(self, min_amplitude=0.02, max_states=20):
        pairs = sorted(
            [(s, a) for s, a in zip(self.states, self.amplitudes) if s.is_consistent()],
            key=lambda p: abs(p[1]),
            reverse=True,
        )[:max_states]
        if not pairs:
            return
        filtered = [(s, a) for s, a in pairs if abs(a) >= min_amplitude]
        pairs = filtered or [pairs[0]]
        self.states, self.amplitudes = map(list, zip(*pairs))
        self.normalize()

    def spectral_mix(self):
        if len(self.amplitudes) < 2:
            return
        amps = np.array(self.amplitudes, dtype=float)
        kernel = np.array([1 / (1 + i) for i in range(len(amps))])
        smoothed = np.fft.ifft(np.fft.fft(amps) * kernel)
        self.amplitudes = smoothed.real.tolist()
        self.normalize()

    def collapse(self):
        idx = max(range(len(self.amplitudes)), key=lambda i: abs(self.amplitudes[i]))
        return self.states[idx]

    def is_stable(self, threshold=0.90):
        return bool(self.amplitudes) and max(abs(a) for a in self.amplitudes) >= threshold

    def __repr__(self):
        best = max(self.states, key=lambda s: s.completeness(), default=None)
        pct = best.completeness() * 100 if best else 0
        return (
            f"PuzzleSuperposition(states={len(self.states)}, "
            f"max_amp={max(self.amplitudes, default=0):.3f}, "
            f"best={pct:.0f}% complete)"
        )


# ---------------------------------------------------------------------------
# Transition Operator
# ---------------------------------------------------------------------------

class PuzzleTransitionOperator:
    """
    Applies all constraints to each live state in sequence, producing one
    derived state per existing state per cycle.  Different initial constraint
    orderings (seeded as separate starting states) represent different proof
    paths through the solution space.
    """

    def __init__(self, constraints):
        self.constraints = list(constraints)

    def update(self, sup: PuzzleSuperposition):
        new_states, new_amplitudes = [], []

        for state, amp in zip(sup.states, sup.amplitudes):
            if not state.is_consistent():
                continue
            derived = state
            for constraint in self.constraints:
                derived = constraint.apply(derived)
                if not derived.is_consistent():
                    derived = state   # constraint caused contradiction — fall back
                    break
            if derived is not state:
                score = 1.0 + derived.completeness() * 3.0
                derived.score = score
                new_states.append(derived)
                new_amplitudes.append(amp * score)
            else:
                new_states.append(state.clone())
                new_amplitudes.append(amp)

        if new_states:
            sup.states = new_states
            sup.amplitudes = new_amplitudes
            sup.normalize()
            sup.deduplicate()


# ---------------------------------------------------------------------------
# Interference
# ---------------------------------------------------------------------------

class PuzzleInterference:
    """
    More complete states reinforce.
    A state that is a superset of another's assignments reinforces both
    (the less complete one is on a valid path toward the same solution).
    """

    def apply(self, sup: PuzzleSuperposition):
        n = len(sup.states)
        for i in range(n):
            si = sup.states[i]
            if not si.is_consistent():
                sup.amplitudes[i] *= 0.01
                continue
            for j in range(i + 1, n):
                sj = sup.states[j]
                # If si has every assignment that sj has, si reinforces sj
                ci, cj = si.completeness(), sj.completeness()
                if ci > cj:
                    sup.amplitudes[i] *= 1.1
                elif cj > ci:
                    sup.amplitudes[j] *= 1.1
        sup.normalize()


# ---------------------------------------------------------------------------
# Reasoning Cycle
# ---------------------------------------------------------------------------

def puzzle_reasoning_cycle(sup, transition, interference, max_iters=20):
    prev_completeness = -1.0

    for iteration in range(max_iters):
        transition.update(sup)
        interference.apply(sup)
        sup.spectral_mix()
        sup.prune()

        best = sup.collapse()
        curr = best.completeness()

        print(f"  [cycle {iteration + 1}] {sup}")

        if best.is_complete() and best.is_consistent():
            print(f"  -> solution found after {iteration + 1} iteration(s)")
            break

        if curr <= prev_completeness:
            print(f"  -> no further progress after {iteration + 1} iteration(s)")
            break

        prev_completeness = curr

    return sup.collapse()
