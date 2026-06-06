from __future__ import annotations
import os
import anthropic
from .path_state import PathState
from .thought_state import ThoughtState

_MODEL = "claude-haiku-4-5-20251001"


def _trace_text(path: PathState) -> str:
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(path.trace)) or "(empty)"


class InterferenceModule:
    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic(api_key=api_key)

    def _ask(self, prompt: str) -> bool:
        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.content[0].text.strip().upper()
        return answer.startswith("YES")

    def conflict(self, path_i: PathState, path_j: PathState) -> bool:
        prompt = (
            "You are a logic evaluator. Given two reasoning traces, "
            "answer only YES or NO.\n\n"
            f"Trace A:\n{_trace_text(path_i)}\n\n"
            f"Trace B:\n{_trace_text(path_j)}\n\n"
            "Do these two traces contradict each other?"
        )
        return self._ask(prompt)

    def reinforce(self, path_i: PathState, path_j: PathState) -> bool:
        prompt = (
            "You are a logic evaluator. Given two reasoning traces, "
            "answer only YES or NO.\n\n"
            f"Trace A:\n{_trace_text(path_i)}\n\n"
            f"Trace B:\n{_trace_text(path_j)}\n\n"
            "Do these two traces mutually support the same conclusion?"
        )
        return self._ask(prompt)

    def apply(self, thought_state: ThoughtState) -> None:
        n = len(thought_state.paths)
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = thought_state.paths[i], thought_state.paths[j]
                if len(pi.trace) == 0 or len(pj.trace) == 0:
                    continue
                if self.conflict(pi, pj):
                    thought_state.amplitudes[i] *= 0.5
                    thought_state.amplitudes[j] *= 0.5
                elif self.reinforce(pi, pj):
                    thought_state.amplitudes[i] *= 1.1
                    thought_state.amplitudes[j] *= 1.1
        thought_state.normalize()
