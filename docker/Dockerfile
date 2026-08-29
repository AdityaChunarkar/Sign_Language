FROM python:3.11-slim

# System libraries needed by opencv-python and mediapipe in a headless container
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces containers run as a non-root user by default
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Hugging Face Spaces (Docker SDK) expects the app to listen on 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "webapp.py"]
