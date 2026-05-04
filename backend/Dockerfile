FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System libs for OpenCV + video processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install all dependencies (onnxruntime ~80MB vs PyTorch+TF ~2GB)
RUN pip install --no-cache-dir -r requirements.txt

# Render injects $PORT at runtime
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1