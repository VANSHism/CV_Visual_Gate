import numpy as np

from src.config import DspConfig


class SpectralSubtractor:
    def __init__(self, config: DspConfig) -> None:
        self.config = config
        self._noise_mag: np.ndarray | None = None

    def _rfft(self, x: np.ndarray) -> np.ndarray:
        return np.fft.rfft(x, n=self.config.fft_size)

    def _irfft(self, x: np.ndarray, length: int) -> np.ndarray:
        return np.fft.irfft(x, n=self.config.fft_size)[:length]

    def maybe_update_noise(self, audio: np.ndarray, allow_update: bool) -> None:
        if not allow_update:
            return
        spectrum = self._rfft(audio)
        mag = np.abs(spectrum)
        if self._noise_mag is None:
            self._noise_mag = mag
            return
        alpha = self.config.noise_alpha
        self._noise_mag = (1.0 - alpha) * self._noise_mag + alpha * mag

    def process(self, audio: np.ndarray) -> np.ndarray:
        if self._noise_mag is None:
            return audio

        spectrum = self._rfft(audio)
        mag = np.abs(spectrum)
        phase = np.exp(1j * np.angle(spectrum))

        cleaned_mag = mag - self.config.oversubtraction * self._noise_mag
        floor = self.config.floor_ratio * self._noise_mag
        cleaned_mag = np.maximum(cleaned_mag, floor)
        cleaned = cleaned_mag * phase

        out = self._irfft(cleaned, len(audio))
        return out.astype(np.float32)

    @property
    def has_noise_profile(self) -> bool:
        return self._noise_mag is not None
