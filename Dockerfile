FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git wget build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch 2.7 cu suport Blackwell (sm_120)
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

RUN pip3 install --no-cache-dir runpod requests Pillow numpy

COPY handler.py .

CMD ["python3", "handler.py"]
