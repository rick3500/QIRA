from __future__ import annotations
import hashlib
import os
import anthropic
from .path_state import PathState

_MODEL = "claude-haiku-4-5-20251001"


def _trace_key(path: PathState) -> str:
    return hashlib.md5("\n".join(path.trace).encode()).hexdigest()


class LLMExpander:
    """Generates next reasoning steps via LLM — no pre-coded answers."""

    def __init__(self, problem: str, n_proposals: int = 2) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.problem = problem
        self.n_proposals = n_proposals
        self._cache: dict[str, list[str]] = {}

    def propose_next_steps(self, path: PathState) -> list[PathState]:
        key = _trace_key(path)
        if key not in self._cache:
            self._cache[key] = self._generate(path)

        steps = self._cache[key]
        if not steps:
            return [path.clone()]  # path is complete — carry forward unchanged

        proposals = []
        for step in steps:
            new_path = path.clone()
            new_path.add_step(step)
            proposals.append(new_path)
        return proposals

    def _generate(self, path: PathState) -> list[str]:
        trace_text = (
            "\n".join(f"{i+1}. {s}" for i, s in enumerate(path.trace))
            or "(none yet — start from scratch)"
        )
        prompt = (
            f"You are a step-by-step reasoning assistant.\n\n"
            f"Problem: {self.problem}\n\n"
            f"Reasoning so far:\n{trace_text}\n\n"
            f"If the problem is fully solved in the trace above, reply with exactly: COMPLETE\n\n"
            f"Otherwise, propose {self.n_proposals} distinct alternative next single reasoning "
            f"steps that each advance toward the solution. Be concise — one step per line.\n\n"
            f"Format:\n"
            f"STEP: <step text>\n"
            f"STEP: <step text>"
        )
        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        if text.upper().startswith("COMPLETE"):
            return []

        steps = []
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("STEP:"):
                step = line[5:].strip()
                if step:
                    steps.append(step)
        return steps[: self.n_proposals]


class LLMVerifier:
    """Checks whether a path has reached a complete answer — without knowing it in advance."""

    def __init__(self, problem: str) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.problem = problem
        self._cache: dict[str, float] = {}

    def check(self, path: PathState) -> float:
        if not path.trace:
            return 0.0
        key = _trace_key(path)
        if key in self._cache:
            return self._cache[key]

        trace_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(path.trace))
        prompt = (
            f"Problem: {self.problem}\n\n"
            f"Reasoning trace:\n{trace_text}\n\n"
            f"Has this trace reached a complete and correct answer? "
            f"Answer YES or NO only."
        )
        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        result = 1.0 if response.content[0].text.strip().upper().startswith("YES") else 0.0
        self._cache[key] = result
        return result
