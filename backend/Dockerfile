FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# System libs: OpenCV (libGL, libGLib), video (ffmpeg), HDF5 (TensorFlow .keras)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb1 \
    ffmpeg \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install PyTorch CPU-only first (avoids pulling CUDA variant)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Render injects $PORT at runtime — use shell form so the variable expands
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1