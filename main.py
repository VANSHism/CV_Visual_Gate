import queue
import threading
import time
import json
import uuid
import os
import wave
import shutil
import subprocess
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pyaudio

from src.audio.processor import LegacyProcessor, create_audio_processor
from src.config import AppConfig
from src.control.state import SharedVisualState
from src.vision.lip_activity import LipActivityDetector


def _normalize_windows_path(path: str) -> str:
    if os.name != "nt":
        return path
    if path.startswith("\\\\?\\UNC\\"):
        return "\\\\" + path[8:]
    if path.startswith("\\\\?\\"):
        return path[4:]
    return path


def _resolve_ffmpeg_binary() -> str | None:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _mux_audio_video(video_path: str, audio_path: str, output_path: str) -> tuple[bool, str]:
    video_path = _normalize_windows_path(video_path)
    audio_path = _normalize_windows_path(audio_path)
    output_path = _normalize_windows_path(output_path)
    ffmpeg_bin = _resolve_ffmpeg_binary()
    if ffmpeg_bin is None:
        return False, "FFmpeg not found. Install dependency 'imageio-ffmpeg' or system ffmpeg."
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, ""
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.strip() if exc.stderr else str(exc)
        return False, err


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "78c57a",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open("debug-78c57a.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def audio_worker(
    config: AppConfig,
    visual_state: SharedVisualState,
    stop_event: threading.Event,
    audio_output_path: str,
    raw_audio_output_path: str,
) -> None:
    # region agent log
    _debug_log(
        "pre-fix",
        "H2",
        "main.py:36",
        "audio_worker_entry",
        {"sampleRate": config.audio.sample_rate, "chunkSize": config.audio.chunk_size},
    )
    # endregion
    pa = pyaudio.PyAudio()
    processor, processor_warning = create_audio_processor(config)
    processor_failed = False
    if processor_warning:
        print(f"Audio processor warning: {processor_warning}")
    frames_in = queue.Queue(maxsize=20)

    def callback(in_data, frame_count, _time_info, _status):
        try:
            frames_in.put_nowait(in_data)
        except queue.Full:
            pass
        return (in_data, pyaudio.paContinue)

    try:
        stream_in = pa.open(
            format=pa.get_format_from_width(config.audio.format_width_bytes),
            channels=config.audio.channels,
            rate=config.audio.sample_rate,
            input=True,
            frames_per_buffer=config.audio.chunk_size,
            stream_callback=callback,
        )
    except Exception as exc:
        # region agent log
        _debug_log(
            "pre-fix",
            "H2",
            "main.py:61",
            "audio_stream_open_failed",
            {"errorType": type(exc).__name__, "error": str(exc)},
        )
        # endregion
        raise

    stream_in.start_stream()
    # region agent log
    _debug_log("pre-fix", "H2", "main.py:74", "audio_stream_started", {})
    # endregion

    wav_file = wave.open(audio_output_path, "wb")
    wav_file.setnchannels(config.audio.channels)
    wav_file.setsampwidth(config.audio.format_width_bytes)
    wav_file.setframerate(config.audio.sample_rate)
    raw_wav_file = wave.open(raw_audio_output_path, "wb")
    raw_wav_file.setnchannels(config.audio.channels)
    raw_wav_file.setsampwidth(config.audio.format_width_bytes)
    raw_wav_file.setframerate(config.audio.sample_rate)
    rms_log_interval_seconds = 1.0
    next_rms_log_time = time.time() + rms_log_interval_seconds

    try:
        while not stop_event.is_set():
            try:
                in_data = frames_in.get(timeout=0.1)
            except queue.Empty:
                continue

            audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            snapshot = visual_state.snapshot()
            closed_dur = visual_state.closed_duration_seconds()
            allow_noise_update = (not snapshot.is_speaking) and (
                closed_dur >= config.vision.min_closed_update_seconds
            )

            try:
                result = processor.process(
                    audio,
                    is_speaking=snapshot.is_speaking,
                    allow_noise_update=allow_noise_update,
                )
            except Exception as exc:
                if not processor_failed:
                    print(f"Audio processor failed, switching to legacy: {type(exc).__name__}: {exc}")
                    processor = LegacyProcessor(config)
                    processor_failed = True
                else:
                    raise
                result = processor.process(
                    audio,
                    is_speaking=snapshot.is_speaking,
                    allow_noise_update=allow_noise_update,
                )
            visual_state.update_audio_telemetry(
                gate_gain=result.telemetry.gate_gain,
                allow_noise_update=result.telemetry.allow_noise_update,
                has_noise_profile=result.telemetry.has_noise_profile,
                processor_mode=result.telemetry.processor_mode,
            )

            out = np.clip(result.audio * 32767.0, -32768, 32767).astype(np.int16).tobytes()
            wav_file.writeframes(out)
            raw_wav_file.writeframes(in_data)

            now = time.time()
            if now >= next_rms_log_time:
                raw_rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
                proc_rms = float(np.sqrt(np.mean(np.square(result.audio)) + 1e-12))
                print(
                    "[audio-debug] "
                    f"raw_rms={raw_rms:.6f} proc_rms={proc_rms:.6f} "
                    f"gate_gain={result.telemetry.gate_gain:.3f} "
                    f"processor={result.telemetry.processor_mode} "
                    f"allow_noise_update={result.telemetry.allow_noise_update} "
                    f"noise_profile={result.telemetry.has_noise_profile}"
                )
                next_rms_log_time = now + rms_log_interval_seconds
    finally:
        # region agent log
        _debug_log("pre-fix", "H5", "main.py:105", "audio_worker_finally", {})
        # endregion
        stream_in.stop_stream()
        stream_in.close()
        wav_file.close()
        raw_wav_file.close()
        pa.terminate()


def run() -> None:
    # region agent log
    _debug_log("pre-fix", "H1", "main.py:114", "run_entry", {})
    # endregion
    config = AppConfig()
    visual_state = SharedVisualState()
    stop_event = threading.Event()
    cwd = _normalize_windows_path(os.getcwd())
    print("Audio tip: stay silent with your mouth closed for about 1 second after startup so the noise profile can learn the room tone.")
    print(
        "Active audio settings: "
        f"mode={config.denoiser.mode} "
        f"wet_mix={config.denoiser.wet_mix:.2f} "
        f"gate_threshold_db={config.gate.threshold_db:.1f} "
        f"gate_attenuation_db={config.gate.attenuation_db:.1f} "
        f"oversubtraction={config.dsp.oversubtraction:.2f} "
        f"floor_ratio={config.dsp.floor_ratio:.2f}"
    )
    recordings_dir = os.path.join(cwd, "recordings")
    os.makedirs(recordings_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_output_path = os.path.join(recordings_dir, f"session_{ts}_processed.mp4")
    raw_output_path = os.path.join(recordings_dir, f"session_{ts}_raw.mp4")
    temp_dir = tempfile.TemporaryDirectory(prefix=f"session_{ts}_", dir=recordings_dir)
    temp_video_output_path = os.path.join(temp_dir.name, "video.mp4")
    temp_audio_output_path = os.path.join(temp_dir.name, "processed.wav")
    temp_raw_audio_output_path = os.path.join(temp_dir.name, "raw.wav")
    try:
        lip_detector = LipActivityDetector(config.vision)
    except Exception as exc:
        # region agent log
        _debug_log(
            "pre-fix",
            "H4",
            "main.py:122",
            "lip_detector_init_failed",
            {"errorType": type(exc).__name__, "error": str(exc)},
        )
        # endregion
        raise

    audio_thread = threading.Thread(
        target=audio_worker,
        args=(config, visual_state, stop_event, temp_audio_output_path, temp_raw_audio_output_path),
        daemon=True,
    )
    audio_thread.start()
    # region agent log
    _debug_log("pre-fix", "H5", "main.py:128", "audio_thread_started", {"isAlive": audio_thread.is_alive()})
    # endregion

    cap = cv2.VideoCapture(config.vision.camera_index)
    # region agent log
    _debug_log(
        "pre-fix",
        "H3",
        "main.py:132",
        "camera_open_attempt",
        {
            "cameraIndex": config.vision.camera_index,
            "isOpened": bool(cap.isOpened()),
            "backend": int(cap.get(cv2.CAP_PROP_BACKEND)),
        },
    )
    # endregion
    if not cap.isOpened():
        # region agent log
        _debug_log("pre-fix", "H1", "main.py:140", "camera_open_failed", {})
        # endregion
        raise RuntimeError("Unable to open camera.")

    video_writer = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # region agent log
                _debug_log("pre-fix", "H3", "main.py:170", "camera_read_failed", {"ok": bool(ok)})
                # endregion
                break

            result = lip_detector.process_frame(frame)
            if result is not None:
                visual_state.update(
                    is_speaking=result.is_speaking,
                    normalized_lip_gap=result.normalized_lip_gap,
                    smoothed_lip_gap=result.smoothed_lip_gap,
                )
                text = (
                    f"Speak:{result.is_speaking} "
                    f"gap={result.normalized_lip_gap:.4f} "
                    f"smooth={result.smoothed_lip_gap:.4f}"
                )
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Open>{config.vision.speech_open_threshold:.3f} Close<{config.vision.speech_close_threshold:.3f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    1,
                )
                audio_info = visual_state.audio_snapshot()
                cv2.putText(
                    frame,
                    f"Mode={audio_info.processor_mode} GateGain={audio_info.gate_gain:.2f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 255),
                    1,
                )
                cv2.putText(
                    frame,
                    f"NoiseProfile={audio_info.has_noise_profile} NoiseUpdateAllowed={audio_info.allow_noise_update}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 255),
                    1,
                )
                cv2.putText(
                    frame,
                    f"Wet={config.denoiser.wet_mix:.2f} GateThr={config.gate.threshold_db:.0f}dB Sub={config.dsp.oversubtraction:.1f}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 200, 255),
                    1,
                )

            cv2.imshow("CV Speech Gate", frame)
            if video_writer is None:
                height, width = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 1.0 or fps > 120.0:
                    fps = 20.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(temp_video_output_path, fourcc, fps, (width, height))
            video_writer.write(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        stop_event.set()
        audio_thread.join(timeout=2.0)
        time.sleep(0.1)
        cap.release()
        if video_writer is not None:
            video_writer.release()
        lip_detector.close()
        cv2.destroyAllWindows()
        ok_processed, mux_msg_processed = _mux_audio_video(
            temp_video_output_path,
            temp_audio_output_path,
            processed_output_path,
        )
        if ok_processed:
            print(f"Saved processed A/V recording: {processed_output_path}")
        else:
            print("Unable to create processed A/V recording.")
            print(f"Reason: {mux_msg_processed}")
        ok_raw, mux_msg_raw = _mux_audio_video(
            temp_video_output_path,
            temp_raw_audio_output_path,
            raw_output_path,
        )
        if ok_raw:
            print(f"Saved raw A/V recording: {raw_output_path}")
        else:
            print("Unable to create raw A/V recording.")
            print(f"Reason: {mux_msg_raw}")
        temp_dir.cleanup()


if __name__ == "__main__":
    run()
