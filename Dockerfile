FROM runpod/base:0.6.2-cuda12.2.0

# Dependențe sistem
RUN apt-get update && apt-get install -y \
    blender \
    python3-pip \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalare PyTorch cu CUDA
RUN pip3 install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Instalare torchmcubes separat (necesită PyTorch preinstalat)
RUN pip3 install --no-cache-dir git+https://github.com/tatsy/torchmcubes.git

# Instalare segment-anything
RUN pip3 install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# Restul dependențelor
RUN pip3 install --no-cache-dir runpod requests Pillow numpy trimesh omegaconf einops transformers huggingface_hub

# Clonare TripoSR (fără să reinstaleze dependențele cu pip)
RUN git clone https://github.com/VAST-AI-Research/TripoSR.git

# Copiere scripturi
COPY handler.py .
COPY blender_cleanup.py .
COPY sam_segment.py .

# Descărcare model SAM (~2.4GB)
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O /app/sam_vit_h_4b8939.pth

CMD ["python3", "handler.py"]
