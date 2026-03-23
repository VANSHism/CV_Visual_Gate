import threading
import time
from dataclasses import dataclass


@dataclass
class VisualStateSnapshot:
    is_speaking: bool
    normalized_lip_gap: float
    smoothed_lip_gap: float
    last_update_ts: float
    closed_since_ts: float


@dataclass
class AudioTelemetrySnapshot:
    gate_gain: float
    allow_noise_update: bool
    has_noise_profile: bool
    processor_mode: str
    last_update_ts: float


class SharedVisualState:
    def __init__(self) -> None:
        now = time.time()
        self._is_speaking = False
        self._normalized_lip_gap = 0.0
        self._smoothed_lip_gap = 0.0
        self._last_update_ts = now
        self._closed_since_ts = now
        self._gate_gain = 1.0
        self._allow_noise_update = False
        self._has_noise_profile = False
        self._processor_mode = "legacy"
        self._audio_last_update_ts = now
        self._lock = threading.Lock()

    def update(self, *, is_speaking: bool, normalized_lip_gap: float, smoothed_lip_gap: float) -> None:
        now = time.time()
        with self._lock:
            if self._is_speaking and not is_speaking:
                self._closed_since_ts = now
            self._is_speaking = is_speaking
            self._normalized_lip_gap = normalized_lip_gap
            self._smoothed_lip_gap = smoothed_lip_gap
            self._last_update_ts = now

    def snapshot(self) -> VisualStateSnapshot:
        with self._lock:
            return VisualStateSnapshot(
                is_speaking=self._is_speaking,
                normalized_lip_gap=self._normalized_lip_gap,
                smoothed_lip_gap=self._smoothed_lip_gap,
                last_update_ts=self._last_update_ts,
                closed_since_ts=self._closed_since_ts,
            )

    def closed_duration_seconds(self, now_ts: float | None = None) -> float:
        now_ts = time.time() if now_ts is None else now_ts
        with self._lock:
            if self._is_speaking:
                return 0.0
            return max(0.0, now_ts - self._closed_since_ts)

    def update_audio_telemetry(
        self,
        *,
        gate_gain: float,
        allow_noise_update: bool,
        has_noise_profile: bool,
        processor_mode: str,
    ) -> None:
        with self._lock:
            self._gate_gain = gate_gain
            self._allow_noise_update = allow_noise_update
            self._has_noise_profile = has_noise_profile
            self._processor_mode = processor_mode
            self._audio_last_update_ts = time.time()

    def audio_snapshot(self) -> AudioTelemetrySnapshot:
        with self._lock:
            return AudioTelemetrySnapshot(
                gate_gain=self._gate_gain,
                allow_noise_update=self._allow_noise_update,
                has_noise_profile=self._has_noise_profile,
                processor_mode=self._processor_mode,
                last_update_ts=self._audio_last_update_ts,
            )
