FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git wget build-essential blender \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch cu128 (suport Blackwell sm_120)
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# OpenCV
RUN pip3 install --no-cache-dir opencv-python-headless

# SAM
RUN pip3 install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# Shap-E
RUN pip3 install --no-cache-dir git+https://github.com/openai/shap-e.git

# Restul
RUN pip3 install --no-cache-dir runpod requests Pillow numpy trimesh ipywidgets

# Model SAM (~2.4GB)
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O /app/sam_vit_h_4b8939.pth

# Instalare numpy în Python-ul intern al Blender (necesar pentru exportul glTF)
RUN BLENDER_PY=$(ls /usr/share/blender/*/python/bin/python3.* 2>/dev/null | head -1) && \
    echo "Blender python: $BLENDER_PY" && \
    $BLENDER_PY -m ensurepip && \
    $BLENDER_PY -m pip install numpy

COPY handler.py .
COPY blender_cleanup.py .
COPY sam_segment.py .

CMD ["python3", "handler.py"]
