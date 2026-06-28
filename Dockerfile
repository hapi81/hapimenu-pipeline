FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git wget build-essential blender \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch
RUN pip3 install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# SAM
RUN pip3 install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# Shap-E
RUN pip3 install --no-cache-dir git+https://github.com/openai/shap-e.git

# Restul
RUN pip3 install --no-cache-dir runpod requests Pillow numpy trimesh

# Model SAM
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O /app/sam_vit_h_4b8939.pth

COPY handler.py .
COPY blender_cleanup.py .
COPY sam_segment.py .

CMD ["python3", "handler.py"]
