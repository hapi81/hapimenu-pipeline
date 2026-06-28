FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git wget build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalare PyTorch cu CUDA
RUN pip3 install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir runpod requests Pillow numpy

COPY handler.py .

CMD ["python3", "handler.py"]
