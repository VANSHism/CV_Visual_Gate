# Visual Gate

Visual Gate is a Python prototype that combines webcam-based lip activity detection with audio denoising. It records video and microphone input at the same time, uses facial landmarks to estimate when you are speaking, and exports two final comparison videos:

- one video with the original microphone audio
- one video with processed audio

The project is best understood as an experiment in visually guided audio processing, not yet as a production-grade noise suppression system.

## Current Version

The current app supports **two processing modes**:

- `legacy`: spectral subtraction + soft-knee noise gate
- `rnnoise`: RNNoise-based denoising with adjustable wet mix

Important behavior in the current version:

- live recording is **silent**: the app no longer plays your processed voice back through the speakers during capture
- each run now exports exactly **two final `.mp4` files**
- temporary intermediate audio/video files are created internally and cleaned up after export
- if `rnnoise` is unavailable or fails during processing, the app falls back to `legacy`

## What The App Does

At a high level, the pipeline works like this:

1. The webcam tracks face landmarks and estimates whether your mouth looks open enough to count as speech.
2. The microphone captures audio in small chunks.
3. The selected audio processor (`legacy` or `rnnoise`) processes the audio.
4. The app saves two final videos:
   - raw video + raw audio
   - raw video + processed audio

This makes it easy to compare whether the processing is actually helping.

## Features

- **Lip Activity Detection**: MediaPipe-based facial landmark tracking for simple speech-state estimation
- **Two Audio Modes**: `legacy` and `rnnoise`
- **Silent Live Capture**: No local speaker playback while recording
- **Side-by-Side Export Strategy**: Raw and processed outputs are both saved for comparison
- **Environment-Driven Configuration**: Most important parameters can be controlled with environment variables
- **Automatic MP4 Export**: Final recordings are muxed into ready-to-review `.mp4` files

## Project Structure

```text
P10_Visual_Gate/
|-- main.py
|-- requirements.txt
|-- src/
|   |-- config.py
|   |-- app/
|   |   `-- calibrate_and_test.py
|   |-- audio/
|   |   |-- noise_gate.py
|   |   `-- processor.py
|   |-- control/
|   |   `-- state.py
|   |-- dsp/
|   |   `-- spectral_subtract.py
|   `-- vision/
|       `-- lip_activity.py
`-- recordings/
```

## Dependencies

- `opencv-python`: video capture and display
- `mediapipe`: face landmark detection
- `numpy`: array and signal math
- `pyaudio`: microphone capture
- `scipy`: signal utilities
- `imageio-ffmpeg`: FFmpeg binary fallback for muxing

Install dependencies with:

```bash
pip install -r requirements.txt
```

Optional but recommended:

```bash
# Windows
winget install FFmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

## Configuration

Configuration is controlled through environment variables.

### Vision

- `VG_CAMERA_INDEX` default `0`
- `VG_SPEECH_OPEN_THRESHOLD` default `0.00637`
- `VG_SPEECH_CLOSE_THRESHOLD` default `0.00382`
- `VG_MIN_CLOSED_UPDATE_SECONDS` default `0.4`

### Audio

- `VG_SAMPLE_RATE` default `16000`
- `VG_CHANNELS` default `1`
- `VG_CHUNK_SIZE` default `512`

### Legacy Mode

- `VG_GATE_THRESHOLD_DB` default `-28.0`
- `VG_GATE_KNEE_WIDTH_DB` default `10.0`
- `VG_GATE_ATTENUATION_DB` default `34.0`
- `VG_GATE_ATTACK_MS` default `20.0`
- `VG_GATE_RELEASE_MS` default `180.0`
- `VG_GATE_HOLD_MS` default `60.0`
- `VG_NOISE_ALPHA` default `0.12`
- `VG_OVERSUBTRACTION` default `2.2`
- `VG_FLOOR_RATIO` default `0.03`

### Denoiser Selection

- `VG_DENOISER_MODE` default `legacy`
  - valid values: `legacy`, `rnnoise`
- `VG_DENOISER_WET` default `1.0`
  - used by `rnnoise`
  - controls blend between original and denoised audio
  - `0.0` = only original audio
  - `1.0` = only denoised audio

Example:

```powershell
$env:VG_DENOISER_MODE="rnnoise"
$env:VG_DENOISER_WET="0.85"
python main.py
```

## Usage

Run the app:

```bash
python main.py
```

During a run, the app will:

1. open the webcam
2. capture microphone audio
3. estimate speech activity from lip movement
4. process audio with the selected mode
5. save exactly two final `.mp4` files

Press `q` in the OpenCV window to stop recording.

## Output Files

Each recording creates exactly two final files in `recordings/`:

- `session_<timestamp>_raw.mp4`
- `session_<timestamp>_processed.mp4`

Meaning:

- `raw.mp4` = webcam video + original microphone audio
- `processed.mp4` = webcam video + processed audio

Intermediate files are stored temporarily and cleaned up automatically.

## How The Two Modes Work

### `legacy`

This is the older hand-built DSP pipeline:

- estimates a background noise profile during non-speaking moments
- applies spectral subtraction
- applies a soft-knee noise gate

This mode is useful for experimentation, but it can sound muffled or choppy if tuned too aggressively.

### `rnnoise`

This mode uses RNNoise through the installed Python backend.

- aims to preserve speech better than simple gating/subtraction
- supports `VG_DENOISER_WET` so you can mix denoised audio with the original
- falls back to `legacy` if RNNoise cannot be initialized or fails during processing

## Known Limitations

This project is still a prototype. Current limitations include:

- noise suppression quality is not consistently better than raw microphone audio
- webcam lip activity is only a rough speech cue, not true speech understanding
- visually guided gating helps most with pauses, not with noise mixed under speech
- `legacy` mode can introduce muffling, pumping, or chopped speech
- `rnnoise` integration is functional, but still not heavily tuned for every microphone/environment

## Calibration

You can use the calibration helper to estimate better lip thresholds:

```bash
python src/app/calibrate_and_test.py
```

This helps the vision side decide more reliably when you are speaking.

## Notes Before Pushing

This README describes the current behavior of the repository:

- no `bypass` mode
- no live local audio playback during recording
- only two final MP4 outputs per run

If you change the export format or denoiser modes later, update this section first so the repo stays accurate.

## License

MIT License. This project is for educational and research purposes.

## Author

Vansh Pal

## Contributing

Contributions, experiments, and improvement ideas are welcome.
