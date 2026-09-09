FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Toolkit CUDA 12.8 complet (nu doar runtime) — necesar ca nvcc să recunoască
# arhitectura Blackwell (compute_120/sm_120); un toolkit mai vechi (ex. 12.4,
# folosit anterior) nu listează deloc compute_120 și blochează compilarea
# extensiilor CUDA native (nvdiffrast) — confirmat empiric în teste anterioare.
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git wget build-essential ninja-build \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bug de resolver pe pip-ul vechi din imaginea de bază (AssertionError la
# instalarea dependințelor InstantMesh) — confirmat empiric, necesar înainte
# de orice alt pip install.
RUN pip3 install --upgrade pip

# PyTorch cu128 — de data asta peste un toolkit CUDA 12.8 complet (nu doar
# pachete pip peste un toolkit vechi), condiție confirmată necesară pentru ca
# extensiile CUDA compilate din sursă (nvdiffrast) să funcționeze pe Blackwell.
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# OpenCV
RUN pip3 install --no-cache-dir opencv-python-headless

# SAM (segmentare — folosit încă, pe o singură imagine per generare, vezi handler.py).
# Nu are extensii CUDA native compilate, doar operații PyTorch standard — compatibil
# cu noua bază fără nicio adaptare suplimentară.
RUN pip3 install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# Shap-E — păstrat ca fallback (run_shape_e_DEPRECATED în handler.py), nu mai e
# calea principală dar rămâne instalat/funcțional dacă InstantMesh eșuează la runtime.
RUN pip3 install --no-cache-dir git+https://github.com/openai/shap-e.git

# ── InstantMesh ──────────────────────────────────────────────────────────
RUN git clone https://github.com/TencentARC/InstantMesh.git /app/InstantMesh

# Dependințele InstantMesh, fără nvdiffrast (instalat separat mai jos —
# pip install din git/clonă produce pachet gol "UNKNOWN", confirmat empiric).
RUN grep -v nvdiffrast /app/InstantMesh/requirements.txt > /app/InstantMesh/requirements_no_nvdiffrast.txt \
    && pip3 install --no-cache-dir -r /app/InstantMesh/requirements_no_nvdiffrast.txt

# accelerate neancorat în requirements.txt ia ultima versiune, incompatibilă cu
# huggingface-hub==0.17.3 cerut de InstantMesh (ImportError la
# split_torch_state_dict_into_shards) — fixat explicit la versiunea confirmată
# funcțională în test.
RUN pip3 install --no-cache-dir 'accelerate==0.24.1'

# rembg (dependință InstantMesh, folosită implicit pentru eliminarea fundalului
# din imaginea de input) are nevoie de onnxruntime, neinclus automat.
RUN pip3 install --no-cache-dir onnxruntime

# nvdiffrast — NU instalat prin `pip install git+URL` (produce pachet gol/corupt
# "UNKNOWN-0.0.0", confirmat empiric: setup.py-ul oficial nu declară name= explicit,
# iar pip modern eșuează să-i deducă identitatea din pyproject.toml). Clonăm sursa
# și compilăm cu setup.py install (funcționează, doar packaging-ul e defect), apoi
# expunem modulul direct din sursă via PYTHONPATH, ocolind pachetul "UNKNOWN".
#
# TORCH_CUDA_ARCH_LIST setat explicit la 12.0 (Blackwell): build-ul de imagine
# Docker pe RunPod rulează probabil FĂRĂ GPU atașat, deci auto-detecția arhitecturii
# țintă (bazată pe GPU-ul vizibil la compile time) ar eșua sau ar ținti arhitectura
# greșită. Acest pas NU a fost testat empiric fără GPU la build — e o măsură
# defensivă bazată pe cum funcționează torch.utils.cpp_extension, nu o reproducere
# exactă a testului (care a compilat cu GPU-ul vizibil prin SSH). Semnalat în raport.
ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN git clone https://github.com/NVlabs/nvdiffrast.git /app/nvdiffrast_src \
    && cd /app/nvdiffrast_src && python3 setup.py install
ENV PYTHONPATH="/app/nvdiffrast_src:${PYTHONPATH}"

# Restul dependințelor pipeline-ului (neschimbate)
RUN pip3 install --no-cache-dir runpod requests Pillow numpy trimesh ipywidgets

# ── Checkpoint-uri, pre-descărcate la build time (ca sam_vit_h_4b8939.pth) ──
# Fără asta, primul request servit de fiecare worker RunPod nou (cold start)
# ar descărca ~6.7GB la runtime — confirmat empiric ~2-3 minute în plus doar
# pentru download, pe lângă generarea efectivă. Repo ID-urile exacte HuggingFace
# sunt verificate din codul sursă run.py (TencentARC/InstantMesh, sudo-ai/zero123plus-v1.2),
# nu presupuse. Rulează pe CPU la build (from_pretrained fără .to('cuda')), deci
# nu are nevoie de GPU la acest pas.
RUN python3 -c "\
from huggingface_hub import hf_hub_download; \
from diffusers import DiffusionPipeline; \
hf_hub_download(repo_id='TencentARC/InstantMesh', filename='diffusion_pytorch_model.bin', repo_type='model'); \
hf_hub_download(repo_id='TencentARC/InstantMesh', filename='instant_mesh_large.ckpt', repo_type='model'); \
DiffusionPipeline.from_pretrained('sudo-ai/zero123plus-v1.2', custom_pipeline='zero123plus')"

# Model SAM (~2.4GB)
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O /app/sam_vit_h_4b8939.pth

COPY handler.py .
COPY blender_cleanup.py .
COPY sam_segment.py .

CMD ["python3", "handler.py"]
