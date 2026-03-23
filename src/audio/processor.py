from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.audio.noise_gate import SoftKneeGate
from src.config import AppConfig
from src.dsp.spectral_subtract import SpectralSubtractor


@dataclass
class AudioProcessTelemetry:
    gate_gain: float
    allow_noise_update: bool
    has_noise_profile: bool
    processor_mode: str


@dataclass
class AudioProcessResult:
    audio: np.ndarray
    telemetry: AudioProcessTelemetry


class AudioProcessor:
    mode_name = "unknown"

    def process(self, audio: np.ndarray, *, is_speaking: bool, allow_noise_update: bool) -> AudioProcessResult:
        raise NotImplementedError


class LegacyProcessor(AudioProcessor):
    mode_name = "legacy"

    def __init__(self, config: AppConfig) -> None:
        self._subtractor = SpectralSubtractor(config.dsp)
        self._gate = SoftKneeGate(config.gate, config.audio.sample_rate)

    def process(self, audio: np.ndarray, *, is_speaking: bool, allow_noise_update: bool) -> AudioProcessResult:
        self._subtractor.maybe_update_noise(audio, allow_update=allow_noise_update)
        denoised = self._subtractor.process(audio)
        gated, gate_gain = self._gate.process(denoised, is_speaking)
        return AudioProcessResult(
            audio=gated.astype(np.float32, copy=False),
            telemetry=AudioProcessTelemetry(
                gate_gain=gate_gain,
                allow_noise_update=allow_noise_update,
                has_noise_profile=self._subtractor.has_noise_profile,
                processor_mode=self.mode_name,
            ),
        )


class RnnoiseProcessor(AudioProcessor):
    mode_name = "rnnoise"

    def __init__(self, config: AppConfig, backend: Any) -> None:
        self._backend = backend
        self._wet_mix = float(np.clip(config.denoiser.wet_mix, 0.0, 1.0))

    def process(self, audio: np.ndarray, *, is_speaking: bool, allow_noise_update: bool) -> AudioProcessResult:
        del is_speaking
        enhanced = self._backend.process_frame(audio)
        if len(enhanced) < len(audio):
            enhanced = np.pad(enhanced, (0, len(audio) - len(enhanced)))
        enhanced = enhanced[: len(audio)]
        mixed = ((1.0 - self._wet_mix) * audio) + (self._wet_mix * enhanced)
        return AudioProcessResult(
            audio=np.clip(mixed, -1.0, 1.0).astype(np.float32),
            telemetry=AudioProcessTelemetry(
                gate_gain=1.0,
                allow_noise_update=allow_noise_update,
                has_noise_profile=True,
                processor_mode=self.mode_name,
            ),
        )


class _RnnoiseBackend:
    def __init__(self, runtime: Any, target_rate: int) -> None:
        self._runtime = runtime
        self._target_rate = target_rate

    def process_frame(self, audio: np.ndarray) -> np.ndarray:
        if hasattr(self._runtime, "denoise_chunk"):
            outputs: list[np.ndarray] = []
            for _, frame in self._runtime.denoise_chunk(audio, partial=False):
                outputs.append(np.asarray(frame).reshape(-1))
            if not outputs:
                return np.zeros_like(audio, dtype=np.float32)
            out = np.concatenate(outputs)
            if np.issubdtype(outputs[0].dtype, np.integer):
                out = out.astype(np.float32) / 32768.0
            return out.astype(np.float32)
        if hasattr(self._runtime, "denoise_frame"):
            _, out = self._runtime.denoise_frame(audio, partial=False)
            out = np.asarray(out).reshape(-1)
            if np.issubdtype(out.dtype, np.integer):
                out = out.astype(np.float32) / 32768.0
            return out.astype(np.float32)
        if hasattr(self._runtime, "process_frame"):
            out = self._runtime.process_frame(audio)
            return np.asarray(out, dtype=np.float32)
        if hasattr(self._runtime, "process"):
            out = self._runtime.process(audio)
            return np.asarray(out, dtype=np.float32)
        if callable(self._runtime):
            out = self._runtime(audio)
            return np.asarray(out, dtype=np.float32)
        raise RuntimeError("RNNoise backend does not expose a supported processing method.")


def _load_rnnoise_backend(target_rate: int) -> _RnnoiseBackend:
    candidates = [
        ("rnnoise", "RNNoise"),
        ("pyrnnoise", "RNNoise"),
    ]
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        runtime = getattr(module, class_name, None)
        if runtime is not None:
            try:
                instance = runtime(sample_rate=target_rate)
            except TypeError:
                try:
                    instance = runtime(target_rate)
                except TypeError:
                    instance = runtime()
            return _RnnoiseBackend(instance, target_rate)

        factory = getattr(module, "create_denoiser", None)
        if callable(factory):
            try:
                instance = factory(sample_rate=target_rate)
            except TypeError:
                instance = factory()
            return _RnnoiseBackend(instance, target_rate)

    raise RuntimeError(
        "RNNoise mode requested, but no supported Python RNNoise package was found. "
        "Install an rnnoise-compatible package or switch VG_DENOISER_MODE to legacy."
    )


def create_audio_processor(config: AppConfig) -> tuple[AudioProcessor, str]:
    mode = config.denoiser.mode
    if mode == "legacy":
        return LegacyProcessor(config), ""
    if mode == "rnnoise":
        try:
            backend = _load_rnnoise_backend(config.audio.sample_rate)
            return RnnoiseProcessor(config, backend), ""
        except Exception as exc:
            fallback = LegacyProcessor(config)
            return fallback, str(exc)
    return LegacyProcessor(config), f"Unknown denoiser mode '{mode}', using legacy."
