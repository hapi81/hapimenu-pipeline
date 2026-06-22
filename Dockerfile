FROM runpod/base:0.6.2-cuda12.2.0

# Dependențe sistem
RUN apt-get update && apt-get install -y \
    blender \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalare Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Clonare TripoSR
RUN git clone https://github.com/VAST-AI-Research/TripoSR.git && \
    cd TripoSR && pip3 install -r requirements.txt

# Copiere scripturi
COPY handler.py .
COPY blender_cleanup.py .
COPY sam_segment.py .

# Descărcare modele (la build time, nu la runtime)
RUN python3 -c "from segment_anything import sam_model_registry; print('SAM ok')" || true

CMD ["python3", "handler.py"]
