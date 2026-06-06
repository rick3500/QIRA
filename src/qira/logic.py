"""
True QIRA for formal logic.

States are frozensets of propositions — not text strings, not LLM outputs.
Transitions are inference rules (modus ponens, universal instantiation).
Interference is fact-set based — same derived facts reinforce, contradictions cancel.
Amplitude tracks how many goal propositions have been derived.
Collapse selects the state that has proved the most.

No LLM. The inference engine does the reasoning.
"""

from __future__ import annotations
import re
import numpy as np


# ---------------------------------------------------------------------------
# Fact parsing
# ---------------------------------------------------------------------------

def _parse_universal(fact: str):
    """Parse 'forall X: P(X) -> Q(X)', return (antecedent, consequent) or None."""
    m = re.match(r"forall\s+\w+\s*:\s*(\w+)\((\w+)\)\s*->\s*(\w+)\((\w+)\)", fact.strip())
    if m:
        return m.group(1), m.group(3)  # antecedent predicate, consequent predicate
    return None

def _parse_ground(fact: str):
    """Parse 'P(a)', return (predicate, individual) or None."""
    m = re.match(r"(\w+)\((\w+)\)$", fact.strip())
    return (m.group(1), m.group(2)) if m else None


# ---------------------------------------------------------------------------
# Inference rules — domain-general, not problem-specific
# ---------------------------------------------------------------------------

def _modus_ponens(facts: frozenset[str]) -> set[str]:
    """forall X: P(X)->Q(X)  +  P(a)  =>  Q(a)  for any a found in facts."""
    derived = set()
    for fact in facts:
        rule = _parse_universal(fact)
        if rule:
            antecedent, consequent = rule
            for other in facts:
                ground = _parse_ground(other)
                if ground and ground[0] == antecedent:
                    new_fact = f"{consequent}({ground[1]})"
                    if new_fact not in facts:
                        derived.add(new_fact)
    return derived

def _transitivity(facts: frozenset[str]) -> set[str]:
    """P->Q and Q->R  =>  P->R  (chain universal rules)."""
    derived = set()
    universals = [(_parse_universal(f), f) for f in facts]
    universals = [(r, f) for r, f in universals if r]
    for (p1, q1), _ in universals:
        for (p2, q2), _ in universals:
            if q1 == p2 and p1 != q2:
                chained = f"forall X: {p1}(X) -> {q2}(X)"
                if chained not in facts:
                    derived.add(chained)
    return derived

def _negation_elimination(facts: frozenset[str]) -> set[str]:
    """not(not(P(a))) => P(a)."""
    derived = set()
    for fact in facts:
        m = re.match(r"not\(not\((.+)\)\)$", fact.strip())
        if m:
            inner = m.group(1)
            if inner not in facts:
                derived.add(inner)
    return derived

_INFERENCE_RULES: list[tuple[str, callable]] = [
    ("modus_ponens",        _modus_ponens),
    ("transitivity",        _transitivity),
    ("negation_elim",       _negation_elimination),
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class LogicState:
    """A knowledge base — a frozenset of derived propositions + history."""

    def __init__(self, facts: frozenset[str], history: list[str] | None = None) -> None:
        self.facts = facts
        self.history: list[str] = history if history is not None else [
            "Axioms: " + ", ".join(sorted(facts))
        ]
        self.score: float = 0.0

    def clone(self) -> LogicState:
        return LogicState(self.facts, self.history[:])

    def has_contradiction(self) -> bool:
        for fact in self.facts:
            ground = _parse_ground(fact)
            if ground and f"not_{ground[0]}({ground[1]})" in self.facts:
                return True
        return False

    def __repr__(self) -> str:
        derived = [f for f in self.facts if not f.startswith("forall")]
        return f"LogicState(derived={sorted(derived)})"


# ---------------------------------------------------------------------------
# Superposition
# ---------------------------------------------------------------------------

class LogicSuperposition:
    def __init__(self) -> None:
        self.states: list[LogicState] = []
        self.amplitudes: list[float] = []

    def add_state(self, state: LogicState, amplitude: float = 1.0) -> None:
        self.states.append(state)
        self.amplitudes.append(amplitude)

    def normalize(self) -> None:
        total = sum(abs(a) for a in self.amplitudes)
        if total:
            self.amplitudes = [a / total for a in self.amplitudes]

    def deduplicate(self) -> None:
        """Merge states with identical fact sets, summing their amplitudes."""
        canonical: dict[frozenset, tuple[LogicState, float]] = {}
        for state, amp in zip(self.states, self.amplitudes):
            key = state.facts
            if key in canonical:
                existing, existing_amp = canonical[key]
                canonical[key] = (existing, existing_amp + amp)
            else:
                canonical[key] = (state, amp)
        self.states = [s for s, _ in canonical.values()]
        self.amplitudes = [a for _, a in canonical.values()]
        self.normalize()

    def prune(self, min_amplitude: float = 0.05, max_states: int = 15) -> None:
        pairs = sorted(
            zip(self.states, self.amplitudes),
            key=lambda p: abs(p[1]),
            reverse=True,
        )[:max_states]
        filtered = [(s, a) for s, a in pairs if abs(a) >= min_amplitude]
        pairs = filtered if filtered else [pairs[0]]
        self.states, self.amplitudes = map(list, zip(*pairs))
        self.normalize()

    def spectral_mix(self) -> None:
        if len(self.amplitudes) < 2:
            return
        amps = np.array(self.amplitudes, dtype=float)
        kernel = np.array([1 / (1 + i) for i in range(len(amps))])
        smoothed = np.fft.ifft(np.fft.fft(amps) * kernel)
        self.amplitudes = smoothed.real.tolist()
        self.normalize()

    def collapse(self) -> LogicState:
        idx = max(range(len(self.amplitudes)), key=lambda i: abs(self.amplitudes[i]))
        return self.states[idx]

    def is_stable(self, threshold: float = 0.80) -> bool:
        return bool(self.amplitudes) and max(abs(a) for a in self.amplitudes) >= threshold

    def __repr__(self) -> str:
        return (
            f"LogicSuperposition(states={len(self.states)}, "
            f"max_amp={max(self.amplitudes, default=0):.3f})"
        )


# ---------------------------------------------------------------------------
# Transition Operator — applies inference rules as Markov transitions
# ---------------------------------------------------------------------------

class LogicTransitionOperator:
    def __init__(self, goals: list[str]) -> None:
        self.goals = set(goals)

    def update(self, sup: LogicSuperposition) -> None:
        new_states: list[LogicState] = []
        new_amplitudes: list[float] = []

        for state, amp in zip(sup.states, sup.amplitudes):
            for proposal in self._propose(state):
                score = self._score(proposal)
                proposal.score = score
                new_states.append(proposal)
                new_amplitudes.append(amp * score)

        sup.states = new_states
        sup.amplitudes = new_amplitudes
        sup.normalize()
        sup.deduplicate()

    def _propose(self, state: LogicState) -> list[LogicState]:
        proposals = []
        for name, rule_fn in _INFERENCE_RULES:
            new_facts = rule_fn(state.facts)
            if new_facts:
                extended = state.facts | frozenset(new_facts)
                step = f"{name:20s} =>  {', '.join(sorted(new_facts))}"
                proposals.append(LogicState(extended, state.history + [step]))
        return proposals or [state.clone()]

    def _score(self, state: LogicState) -> float:
        if state.has_contradiction():
            return 0.1   # heavily penalise contradictions
        goals_proved = sum(1 for g in self.goals if g in state.facts)
        return 1.0 + goals_proved * 0.5


# ---------------------------------------------------------------------------
# Interference — fact-set based, no LLM
# ---------------------------------------------------------------------------

class LogicInterference:
    def apply(self, sup: LogicSuperposition) -> None:
        n = len(sup.states)
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = sup.states[i], sup.states[j]

                if si.has_contradiction():
                    sup.amplitudes[i] *= 0.1
                    continue
                if sj.has_contradiction():
                    sup.amplitudes[j] *= 0.1
                    continue

                # Constructive: one state's facts are a superset of the other's
                if si.facts >= sj.facts:
                    sup.amplitudes[i] *= 1.2   # si knows everything sj knows, and more
                elif sj.facts >= si.facts:
                    sup.amplitudes[j] *= 1.2
        sup.normalize()


# ---------------------------------------------------------------------------
# Reasoning Cycle
# ---------------------------------------------------------------------------

def logic_reasoning_cycle(
    sup: LogicSuperposition,
    transition: LogicTransitionOperator,
    interference: LogicInterference,
    max_iters: int = 10,
) -> LogicState:
    for iteration in range(max_iters):
        transition.update(sup)
        interference.apply(sup)
        sup.spectral_mix()
        sup.prune()

        print(f"  [cycle {iteration + 1}] {sup}")

        if sup.is_stable():
            print(f"  -> stable after {iteration + 1} iteration(s)")
            break

    return sup.collapse()
