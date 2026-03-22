# Use Python 3.11 slim image with system libraries support
FROM python:3.11-slim

# Install system dependencies for OpenCV, audio, FFmpeg, and camera access
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    ffmpeg \
    libopencv-dev \
    python3-opencv \
    libasound2 \
    alsa-utils \
    pulseaudio \
    libpulse0 \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create recordings directory
RUN mkdir -p recordings

# Set environment variables for headless operation (optional)
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "main.py"]
