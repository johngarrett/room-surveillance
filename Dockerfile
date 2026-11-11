FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# Install CPU wheels first to avoid CUDA runtime dependencies.
RUN pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
# Package weights in the image; no model download on application restarts.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
COPY main.py camera.py detector.py scheduler.py config.py server.py ./
COPY static ./static
RUN mkdir -p photos config
EXPOSE 8080
CMD ["python", "main.py"]
