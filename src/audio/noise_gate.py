import math

import numpy as np

from src.config import GateConfig


class SoftKneeGate:
    def __init__(self, config: GateConfig, sample_rate: int) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self._gain = 1.0
        self._hold_samples_remaining = 0

    @staticmethod
    def _db_to_linear(db: float) -> float:
        return 10.0 ** (db / 20.0)

    def _target_gain_from_level_db(self, level_db: float) -> float:
        threshold = self.config.threshold_db
        knee = max(self.config.knee_width_db, 1e-6)
        min_gain = self._db_to_linear(-self.config.attenuation_db)
        knee_low = threshold - knee / 2.0
        knee_high = threshold + knee / 2.0

        if level_db <= knee_low:
            return min_gain
        if level_db >= knee_high:
            return 1.0

        t = (level_db - knee_low) / knee
        eased = t * t * (3.0 - 2.0 * t)
        return min_gain + (1.0 - min_gain) * eased

    def process(self, audio: np.ndarray, is_visual_speaking: bool) -> tuple[np.ndarray, float]:
        rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
        level_db = 20.0 * math.log10(max(rms, 1e-8))

        if is_visual_speaking:
            target_gain = self._target_gain_from_level_db(level_db + 10.0)
            self._hold_samples_remaining = int(self.config.hold_ms * self.sample_rate / 1000.0)
        else:
            if self._hold_samples_remaining > 0:
                target_gain = self._gain
                self._hold_samples_remaining = max(0, self._hold_samples_remaining - len(audio))
            else:
                target_gain = self._target_gain_from_level_db(level_db - 8.0)

        attack_coeff = math.exp(-1.0 / max(1.0, (self.config.attack_ms / 1000.0) * self.sample_rate))
        release_coeff = math.exp(-1.0 / max(1.0, (self.config.release_ms / 1000.0) * self.sample_rate))
        coeff = attack_coeff if target_gain > self._gain else release_coeff
        self._gain = coeff * self._gain + (1.0 - coeff) * target_gain

        return audio * self._gain, self._gain
