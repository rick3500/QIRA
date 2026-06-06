from __future__ import annotations
import numpy as np
from .thought_state import ThoughtState


class SpectralMixer:
    def mix(self, thought_state: ThoughtState) -> None:
        if len(thought_state.amplitudes) < 2:
            return
        amps = np.array(thought_state.amplitudes, dtype=float)
        fft_vals = np.fft.fft(amps)
        smoothed = np.fft.ifft(fft_vals * self._kernel(len(amps)))
        thought_state.amplitudes = smoothed.real.tolist()
        thought_state.normalize()

    @staticmethod
    def _kernel(n: int) -> list[float]:
        # low-pass smoothing: attenuates high-frequency amplitude noise
        return [1 / (1 + i) for i in range(n)]
