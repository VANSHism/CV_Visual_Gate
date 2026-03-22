# Visual Gate

A Python application that intelligently gates audio based on visual speech detection. Using computer vision to detect lip activity and facial landmarks, Visual Gate automatically mutes or attenuates audio during periods of silence, perfect for podcast recording, video interviews, or live streaming.

## Features

- **Lip Activity Detection**: Uses MediaPipe Face Landmarker to detect speech-related mouth movements
- **Intelligent Audio Gating**: Soft-knee noise gate with configurable thresholds for smooth audio transitions
- **Spectral Subtraction**: Optional audio enhancement to reduce background noise
- **Real-time Processing**: Simultaneous video capture and audio input processing
- **Audio-Video Muxing**: Automatically synchronizes and combines processed audio with video output
- **Configurable Parameters**: Environment variables and dataclass-based configuration for all system parameters

## Project Structure

```
P10_Visual_Gate/
├── main.py                      # Main application entry point
├── requirements.txt             # Python dependencies
├── src/
│   ├── config.py               # Configuration dataclasses
│   ├── app/
│   │   └── calibrate_and_test.py # Calibration and testing utilities
│   ├── audio/
│   │   └── noise_gate.py        # Soft-knee noise gate implementation
│   ├── control/
│   │   └── state.py             # Shared state management
│   ├── dsp/
│   │   └── spectral_subtract.py # Spectral subtraction audio processing
│   └── vision/
│       └── lip_activity.py      # Lip activity detection using MediaPipe
└── recordings/                  # Output directory for processed videos
```

## Dependencies

- **opencv-python**: Video capture and processing
- **mediapipe**: Face landmark detection and lip activity recognition
- **numpy**: Numerical computations
- **pyaudio**: Audio input/output stream handling
- **scipy**: Scientific computing utilities
- **imageio-ffmpeg**: Audio-video muxing via FFmpeg

Install dependencies with:
```bash
pip install -r requirements.txt
```

Optionally, install system `ffmpeg` for improved muxing performance:
```bash
# Windows (via winget or chocolatey)
winget install FFmpeg
# or
choco install ffmpeg

# macOS (via Homebrew)
brew install ffmpeg

# Linux (via apt)
sudo apt-get install ffmpeg
```

## Configuration

Visual Gate is configured via environment variables that override defaults. Key configuration options:

### Vision Configuration
- `VG_CAMERA_INDEX` (default: 0): Camera device index
- `VG_SPEECH_OPEN_THRESHOLD` (default: 0.00637): Lip gap threshold to open gate
- `VG_SPEECH_CLOSE_THRESHOLD` (default: 0.00382): Lip gap threshold to close gate
- `VG_MIN_CLOSED_UPDATE_SECONDS` (default: 0.4): Minimum duration gate stays closed

### Audio Configuration
- `VG_SAMPLE_RATE` (default: 16000): Audio sample rate in Hz
- `VG_CHANNELS` (default: 1): Number of audio channels
- `VG_CHUNK_SIZE` (default: 512): Frames per buffer

### Gate Configuration
- `VG_GATE_THRESHOLD_DB` (default: -28.0): Gate threshold in dB
- `VG_GATE_KNEE_WIDTH_DB` (default: 10.0): Knee width for soft gating
- `VG_GATE_ATTENUATION_DB` (default: 34.0): Maximum attenuation when gate is closed
- `VG_GATE_ATTACK_MS` (default: 20.0): Gate attack time in milliseconds

Example: Set custom sample rate and gate threshold:
```bash
set VG_SAMPLE_RATE=48000 VG_GATE_THRESHOLD_DB=-30.0
python main.py
```

## Usage

Run the application:
```bash
python main.py
```

The application will:
1. Open your default camera
2. Capture audio and video simultaneously
3. Detect lip activity in real-time
4. Apply audio gating based on speech detection
5. Save processed video and audio files to the `recordings/` directory
6. Mix audio and video into a final output file

## Output Files

Processed recordings are saved with timestamps in the `recordings/` directory:
- `output_with_gate_YYYY-MM-DD_HH-MM-SS.mp4` - Final muxed video with gated audio
- `audio_processed_*.wav` - Processed audio track
- `raw_audio_*.wav` - Original audio for reference

## How It Works

1. **Vision Pipeline**: MediaPipe detects facial landmarks and measures the vertical gap between upper and lower lips
2. **Speech Detection**: Compares normalized lip gap against configurable thresholds to determine speech activity
3. **Audio Gating**: When speech is detected, the soft-knee noise gate opens; otherwise, it attenuates audio based on knee characteristics
4. **Output**: Gated audio is mixed with video and saved to file

## Calibration and Testing

Use the calibration module for fine-tuning speech detection thresholds:
```python
from src.app.calibrate_and_test import calibrate
```

## Docker Deployment

Visual Gate can be containerized for consistent deployment across different environments.

### Building the Docker Image

```bash
docker build -t visual-gate:latest .
```

### Running with Docker Compose

Docker Compose is the recommended way to run Visual Gate with proper device access and volume mounting:

```bash
docker-compose up
```

This will:
- Build the image if it doesn't exist
- Mount your camera device (`/dev/video0`)
- Mount audio devices for PulseAudio
- Mount the `recordings/` directory for persistent output
- Run the application with all configured environment variables

### Running with Docker CLI

If you prefer direct Docker commands:

```bash
docker run -it \
  --device /dev/video0:/dev/video0 \
  --device /dev/snd:/dev/snd \
  -v $(pwd)/recordings:/app/recordings \
  -v /run/user/1000/pulse:/run/user/1000/pulse:ro \
  -e VG_SAMPLE_RATE=16000 \
  -e VG_GATE_THRESHOLD_DB=-28.0 \
  visual-gate:latest
```

**Note on Audio**: 
- On **Linux**: PulseAudio socket is mounted from `/run/user/1000/pulse` (adjust UID 1000 if needed)
- On **macOS/Windows**: Use Docker Desktop with appropriate audio forwarding, or remove the PulseAudio volume mount and configure audio routing separately

### Environment Variables in Docker

All configuration environment variables work the same way in Docker. Override them in `docker-compose.yml` or via `-e` flags:

```bash
docker-compose up -e VG_SPEECH_OPEN_THRESHOLD=0.0070
```

### Troubleshooting Docker Setup

- **Camera not detected**: Verify `/dev/video0` exists and you have permissions
- **No audio**: Check PulseAudio socket path and ensure audio devices are accessible
- **Permission denied on recordings**: Ensure the host `recordings/` directory has proper permissions

## License

[Specify your license here]

## Author

[Your name or organization]

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
