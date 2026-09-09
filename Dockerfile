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

# Fix pentru eroare confirmata in productie (build GHCR, test real end-to-end):
# nvdiffrast/__init__.py face `importlib.metadata.version('nvdiffrast')` la import,
# care cauta metadata pachetului instalat sub numele exact "nvdiffrast" - dar
# `setup.py install` nu mai inregistreaza mereu aceasta metadata (comportament
# fragil, dependent de versiunea pip/setuptools din imaginea de baza la momentul
# build-ului - a functionat in sesiunea SSH de test, dar a esuat identic in
# productie cu eroarea exacta importlib.metadata.PackageNotFoundError: No package
# metadata was found for nvdiffrast). Fix robust, independent de comportamentul
# setup.py: suprascriem fisierul cu o versiune hardcodata, eliminand complet
# dependenta de importlib.metadata la import. Continutul original al fisierului
# (verificat direct din sursa oficiala NVlabs/nvdiffrast) e doar acest lookup de
# versiune, nimic altceva - sigur de inlocuit integral.
RUN printf '__version__ = "0.3.3"\n' > /app/nvdiffrast_src/nvdiffrast/__init__.py

ENV PYTHONPATH="/app/nvdiffrast_src:${PYTHONPATH}"

# Restul dependințelor pipeline-ului (neschimbate)
RUN pip3 install --no-cache-dir runpod requests Pillow numpy trimesh ipywidgets

# ── Checkpoint-uri InstantMesh/Zero123++ (~6.7GB) — NU mai pre-descărcate la
# build time. Build-ul RunPod eșuează consecvent ("Creating cache directory"
# apoi nimic, fără nicio linie din execuția Dockerfile-ului) pe mașina lor de
# build, separată de Container Disk-ul endpoint-ului — confirmat de suportul
# RunPod, mărirea Container Disk-ului nu a avut niciun efect. Suspiciune:
# imaginea (CUDA devel + torch + aceste checkpoint-uri) depășește limita
# mașinii lor de build. Le lăsăm să se descarce la runtime, la primul request
# servit de fiecare worker nou — cost: ~2-3 minute în plus doar la cold start,
# o singură dată per worker (huggingface_hub cache-uiește local pe disk după
# prima descărcare, requesturile următoare pe același worker nu re-descarcă).
# Repo ID-urile sunt cele verificate din codul sursă run.py (TencentARC/InstantMesh,
# sudo-ai/zero123plus-v1.2) — descărcarea se întâmplă automat prin run.py însuși,
# nu necesită cod separat aici.

# Model SAM (~2.4GB) — păstrat pre-descărcat la build (funcționa deja înainte de
# migrarea la InstantMesh, nu e suspectat ca fiind cauza eșecului de build).
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
    -O /app/sam_vit_h_4b8939.pth

COPY handler.py .
COPY blender_cleanup.py .
COPY sam_segment.py .

CMD ["python3", "handler.py"]
