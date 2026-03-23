import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class VisionConfig:
    camera_index: int = field(default_factory=lambda: _env_int("VG_CAMERA_INDEX", 0))
    max_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    smoothing_alpha: float = 0.2
    speech_open_threshold: float = field(default_factory=lambda: _env_float("VG_SPEECH_OPEN_THRESHOLD", 0.00637))
    speech_close_threshold: float = field(default_factory=lambda: _env_float("VG_SPEECH_CLOSE_THRESHOLD", 0.00382))
    min_closed_update_seconds: float = field(default_factory=lambda: _env_float("VG_MIN_CLOSED_UPDATE_SECONDS", 0.4))


@dataclass
class AudioConfig:
    sample_rate: int = field(default_factory=lambda: _env_int("VG_SAMPLE_RATE", 16000))
    channels: int = field(default_factory=lambda: _env_int("VG_CHANNELS", 1))
    chunk_size: int = field(default_factory=lambda: _env_int("VG_CHUNK_SIZE", 512))
    format_width_bytes: int = 2


@dataclass
class GateConfig:
    threshold_db: float = field(default_factory=lambda: _env_float("VG_GATE_THRESHOLD_DB", -28.0))
    knee_width_db: float = field(default_factory=lambda: _env_float("VG_GATE_KNEE_WIDTH_DB", 10.0))
    attenuation_db: float = field(default_factory=lambda: _env_float("VG_GATE_ATTENUATION_DB", 34.0))
    attack_ms: float = field(default_factory=lambda: _env_float("VG_GATE_ATTACK_MS", 20.0))
    release_ms: float = field(default_factory=lambda: _env_float("VG_GATE_RELEASE_MS", 180.0))
    hold_ms: float = field(default_factory=lambda: _env_float("VG_GATE_HOLD_MS", 60.0))


@dataclass
class DspConfig:
    fft_size: int = field(default_factory=lambda: _env_int("VG_FFT_SIZE", 512))
    hop_size: int = field(default_factory=lambda: _env_int("VG_HOP_SIZE", 256))
    noise_alpha: float = field(default_factory=lambda: _env_float("VG_NOISE_ALPHA", 0.12))
    oversubtraction: float = field(default_factory=lambda: _env_float("VG_OVERSUBTRACTION", 2.2))
    floor_ratio: float = field(default_factory=lambda: _env_float("VG_FLOOR_RATIO", 0.03))


@dataclass
class DenoiserConfig:
    mode: str = field(default_factory=lambda: os.getenv("VG_DENOISER_MODE", "legacy").strip().lower())
    wet_mix: float = field(default_factory=lambda: _env_float("VG_DENOISER_WET", 1.0))
    preferred_sample_rate: int = field(default_factory=lambda: _env_int("VG_DENOISER_SAMPLE_RATE", 48000))


@dataclass
class AppConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    dsp: DspConfig = field(default_factory=DspConfig)
    denoiser: DenoiserConfig = field(default_factory=DenoiserConfig)
