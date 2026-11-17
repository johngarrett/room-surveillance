FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# Install CPU wheels first to avoid CUDA runtime dependencies.
RUN pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-docker.lock ./
RUN pip install -r requirements.txt -c requirements-docker.lock && pip check
# Package weights in the image; no model download on application restarts.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
RUN echo '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1  yolo11n.pt' | sha256sum -c -
COPY main.py camera.py detector.py scheduler.py config_store.py server.py ./
COPY static ./static
RUN mkdir -p photos config images
EXPOSE 5418
CMD ["python", "main.py"]
