from __future__ import annotations
import hashlib
import os
import anthropic
from .path_state import PathState
from .thought_state import ThoughtState

_MODEL = "claude-haiku-4-5-20251001"


def _trace_text(path: PathState) -> str:
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(path.trace)) or "(empty)"


def _trace_hash(path: PathState) -> str:
    return hashlib.md5("\n".join(path.trace).encode()).hexdigest()


# Three-way judgment: CONFLICT | REINFORCE | NEITHER
_JUDGMENT = {"CONFLICT", "REINFORCE", "NEITHER"}


class InterferenceModule:
    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic(api_key=api_key)
        # cache: (hash_i, hash_j) -> "CONFLICT" | "REINFORCE" | "NEITHER"
        self._cache: dict[tuple[str, str], str] = {}

    def _judge(self, path_i: PathState, path_j: PathState) -> str:
        """Single API call returning CONFLICT, REINFORCE, or NEITHER."""
        key = (_trace_hash(path_i), _trace_hash(path_j))
        if key in self._cache:
            return self._cache[key]

        prompt = (
            "You are a logic evaluator. Compare two reasoning traces and "
            "respond with exactly one word:\n"
            "  CONFLICT   — the traces reach contradictory conclusions\n"
            "  REINFORCE  — the traces mutually support the same conclusion\n"
            "  NEITHER    — the traces are unrelated or inconclusive\n\n"
            f"Trace A:\n{_trace_text(path_i)}\n\n"
            f"Trace B:\n{_trace_text(path_j)}\n\n"
            "Your one-word answer:"
        )
        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        word = response.content[0].text.strip().upper().split()[0]
        result = word if word in _JUDGMENT else "NEITHER"
        self._cache[key] = result
        return result

    def apply(self, thought_state: ThoughtState) -> None:
        n = len(thought_state.paths)
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = thought_state.paths[i], thought_state.paths[j]
                if not pi.trace or not pj.trace:
                    continue
                judgment = self._judge(pi, pj)
                if judgment == "CONFLICT":
                    thought_state.amplitudes[i] *= 0.5
                    thought_state.amplitudes[j] *= 0.5
                elif judgment == "REINFORCE":
                    thought_state.amplitudes[i] *= 1.1
                    thought_state.amplitudes[j] *= 1.1
        thought_state.normalize()
