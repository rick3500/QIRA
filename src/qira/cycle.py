from __future__ import annotations
from .path_state import PathState
from .thought_state import ThoughtState
from .transition import TransitionOperator
from .interference import InterferenceModule
from .spectral import SpectralMixer

_STABILITY_THRESHOLD = 0.85
_CONVERGENCE_DELTA = 1e-4


def is_stable(thought_state: ThoughtState, prev_amplitudes: list[float] | None = None) -> bool:
    if not thought_state.amplitudes:
        return True
    max_amp = max(abs(a) for a in thought_state.amplitudes)
    if max_amp >= _STABILITY_THRESHOLD:
        return True
    if prev_amplitudes and len(prev_amplitudes) == len(thought_state.amplitudes):
        delta = sum(abs(a - b) for a, b in zip(thought_state.amplitudes, prev_amplitudes))
        if delta < _CONVERGENCE_DELTA:
            return True
    return False


def reasoning_cycle(
    thought_state: ThoughtState,
    transition: TransitionOperator,
    interference: InterferenceModule,
    mixer: SpectralMixer,
    max_iters: int = 10,
) -> PathState:
    prev_amps: list[float] | None = None

    for iteration in range(max_iters):
        prev_amps = thought_state.amplitudes[:]
        transition.update(thought_state)
        interference.apply(thought_state)
        mixer.mix(thought_state)
        thought_state.prune()

        print(f"  [cycle {iteration + 1}] {thought_state}")

        if is_stable(thought_state, prev_amps):
            print(f"  -> stable after {iteration + 1} iteration(s)")
            break

    return thought_state.collapse()
