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

# Debug: găsim structura exactă a Blender
RUN dpkg -L blender | grep -i python | head -20
RUN ls -la /usr/share/blender/ 2>/dev/null || true
RUN find /usr/share/blender -maxdepth 4 -type d 2>/dev/null | head -30

# Instalare numpy în Python-ul intern al Blender
RUN BLENDER_PY=$(find /usr/share/blender -name "python3.*" -type f 2>/dev/null | head -1) && \
    if [ -z "$BLENDER_PY" ]; then BLENDER_PY=$(find /opt -name "python3.*" -type f -path "*blender*" 2>/dev/null | head -1); fi && \
    echo "Blender python found at: $BLENDER_PY" && \
    test -n "$BLENDER_PY" && \
    "$BLENDER_PY" -m ensurepip --default-pip && \
    "$BLENDER_PY" -m pip install numpy

COPY handler.py .
COPY blender_cleanup.py .
COPY sam_segment.py .

CMD ["python3", "handler.py"]
