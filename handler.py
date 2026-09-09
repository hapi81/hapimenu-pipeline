import sys
import traceback
import runpod
import os
import requests
import tempfile
import subprocess
import torch
import numpy as np
from PIL import Image

def download_image(url, path):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(path, 'wb') as f:
        f.write(response.content)

def upload_to_supabase(file_path, bucket, filename):
    supabase_url = os.environ.get("SUPABASE_URL")
    service_key = os.environ.get("SUPABASE_SERVICE_KEY")
    url = f"{supabase_url}/storage/v1/object/{bucket}/{filename}"
    print(f"[Upload] URL: {url}")
    print(f"[Upload] Key prefix: {service_key[:20] if service_key else 'None'}...")
    with open(file_path, 'rb') as f:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {service_key}",
                "apikey": service_key,
                "Content-Type": "model/gltf-binary",
                "x-upsert": "true"
            },
            data=f
        )
    print(f"[Upload] Status: {response.status_code}, Response: {response.text[:200]}")
    if response.status_code in [200, 201]:
        return f"{supabase_url}/storage/v1/object/public/{bucket}/{filename}"
    else:
        raise Exception(f"Upload Supabase eșuat: {response.text}")

def run_sam_segmentation(image_paths, output_dir):
    segmented = []
    for i, img_path in enumerate(image_paths):
        out_path = os.path.join(output_dir, f"seg_{i}.png")
        result = subprocess.run([
            "python3", "/app/sam_segment.py",
            "--input", img_path,
            "--output", out_path
        ], capture_output=True, text=True, timeout=120)

        # Logging vizibil, altfel stdout-ul SAM e capturat si aruncat silentios.
        print(f"[SAM {i}] stdout: {result.stdout.strip()}")
        if result.stderr:
            print(f"[SAM {i}] stderr: {result.stderr.strip()}")
        print(f"[SAM {i}] returncode: {result.returncode}")

        if result.returncode != 0:
            raise Exception(f"SAM eroare: {result.stderr}")

        # Verificare explicita daca fallback-ul (fara masca gasita) a fost activat.
        if "Segmentare completă" not in result.stdout:
            print(f"[SAM {i}] AVERTISMENT: nu am gasit confirmarea de segmentare in stdout. "
                  f"Posibil fallback pe imaginea originala fara canal alpha transparent.")

        segmented.append(out_path)
    return segmented

def run_shape_e_DEPRECATED(image_path, output_dir):
    """Metoda veche de generare 3D (single-image, fără textură/UV — doar
    culoare per-vertex). Păstrată neschimbată ca fallback rapid dacă
    InstantMesh eșuează la runtime; nu mai e calea principală (vezi
    run_instantmesh mai jos și handler())."""
    from shap_e.diffusion.sample import sample_latents
    from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
    from shap_e.models.download import load_model, load_config
    from shap_e.util.notebooks import decode_latent_mesh

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Shap-E device: {device}")

    xm = load_model('transmitter', device=device)
    model = load_model('image300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    image = Image.open(image_path).convert('RGBA').resize((256, 256))

    latents = sample_latents(
        batch_size=1,
        model=model,
        diffusion=diffusion,
        guidance_scale=3.0,
        model_kwargs=dict(images=[image]),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
    obj_path = os.path.join(output_dir, "mesh.obj")
    with open(obj_path, 'w') as f:
        mesh.write_obj(f)
    return obj_path

def run_instantmesh(image_path, output_dir):
    """Generare 3D cu InstantMesh (Tencent ARC) — single-image, cu textură UV
    reală (Zero123++ pentru sinteza celor 6 vederi + reconstrucție LRM).
    Înlocuiește Shap-E ca metodă principală, confirmat empiric funcțional pe
    Blackwell/sm_120 (test pe RunPod, mesh + textură UV verificate vizual).

    Rulat ca subprocess (nu import direct în proces), la fel ca run_sam_segmentation
    — run.py e un script CLI monolitic (config via omegaconf, setup CUDA propriu),
    nu conceput ca funcție importabilă; refactorizarea lui nu a fost testată și ar
    risca să difere de fluxul dovedit funcțional în test.

    Folosește imaginea ORIGINALĂ (needitată de SAM), nu output-ul SAM: InstantMesh
    face propria eliminare de fundal intern (rembg, implicit dacă --no_rembg nu e
    dat) — exact configurația testată empiric cu succes. Nicio dovadă găsită că
    imaginea SAM-segmentată ar da rezultate mai bune; schimbarea asta ar însemna
    să ne abatem de la singura configurație confirmată funcțională, fără motiv.
    """
    cmd = [
        "python3", "/app/InstantMesh/run.py",
        "/app/InstantMesh/configs/instant-mesh-large.yaml",
        image_path,
        "--export_texmap",
        "--output_path", output_dir,
    ]

    env = os.environ.copy()
    # nvdiffrast nu e instalat ca pachet normal (vezi Dockerfile) — trebuie expus
    # explicit prin PYTHONPATH și în subprocess, altfel run.py nu-l găsește.
    env["PYTHONPATH"] = "/app/nvdiffrast_src:" + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        env=env, cwd="/app/InstantMesh"
    )

    print(f"[InstantMesh] stdout (ultimele 3000 caractere): {result.stdout[-3000:]}")
    if result.stderr:
        print(f"[InstantMesh] stderr (ultimele 3000 caractere): {result.stderr[-3000:]}")
    print(f"[InstantMesh] returncode: {result.returncode}")

    if result.returncode != 0:
        raise Exception(f"InstantMesh eroare: {result.stderr[-1500:]}")

    # run.py salvează în {output_path}/instant-mesh-large/meshes/{basename}.obj
    # (plus .mtl și .png alături, referențiate din .mtl — nu mutate/redenumite aici,
    # run_trimesh_export le citește direct de la calea .obj).
    basename = os.path.splitext(os.path.basename(image_path))[0]
    obj_path = os.path.join(output_dir, "instant-mesh-large", "meshes", f"{basename}.obj")

    if not os.path.exists(obj_path):
        raise Exception(f"InstantMesh nu a produs fișierul așteptat: {obj_path}")

    return obj_path

def run_trimesh_export(obj_path, output_glb):
    """Exportă .obj → .glb folosind trimesh (fără Blender).

    Verificat explicit local (fișierele reale din testul InstantMesh anterior,
    trimesh 5.1.0): trimesh.load(obj_path, force='mesh') citește corect
    materialul din .mtl și textura din .png (mesh.visual.material.image,
    UV coords prezente), iar după export + reîncărcare GLB, textura
    supraviețuiește ca baseColorTexture încorporat — nu doar geometria goală.
    Cu Shap-E (fără material/textură deloc) acest cod funcționa oricum, deci
    n-a fost nevoie de nicio modificare aici pentru InstantMesh."""
    import trimesh
    print(f"[trimesh] Import {obj_path}...")
    mesh = trimesh.load(obj_path, force='mesh')
    print(f"[trimesh] Mesh încărcat: {mesh}")

    # Normalizează la max 1m
    scale = 1.0 / max(mesh.extents) if max(mesh.extents) > 0 else 1.0
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.centroid)

    print(f"[trimesh] Export .glb → {output_glb}")
    mesh.export(output_glb)
    print(f"[trimesh] Export complet, size: {os.path.getsize(output_glb)} bytes")
    return output_glb

def handler(job):
    job_input = job["input"]
    product_id = job_input.get("product_id")
    job_id = job_input.get("job_id")
    image_urls = job_input.get("images", [])

    # Relaxat de la "== 5" strict: Unity-ul curent trimite tot 5 imagini (Față/
    # Spate/Stânga/Dreapta/Sus), păstrăm compatibilitatea cu el neschimbat. Dar
    # InstantMesh (single-image) folosește doar prima — restul sunt acceptate,
    # nu erori, dar ignorate silențios pentru generarea efectivă (vezi mai jos).
    if len(image_urls) < 1:
        return {"error": f"Necesară cel puțin o imagine, primite {len(image_urls)}"}
    if not product_id or not job_id:
        return {"error": "product_id și job_id sunt obligatorii"}

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f"[1/4] Descărcare imagini...")
            image_paths = []
            for i, url in enumerate(image_urls):
                path = os.path.join(tmp_dir, f"image_{i}.jpg")
                download_image(url, path)
                image_paths.append(path)
                print(f"  Imagine {i+1}/{len(image_urls)} descărcată")

            # SAM rulează doar pe prima imagine (singura folosită efectiv, mai jos) —
            # segmentarea celorlalte 4 ar fi timp/cost irosit, InstantMesh nici nu le
            # folosește. Output-ul SAM rămâne disponibil pentru fallback-ul Shap-E
            # (run_shape_e_DEPRECATED), care chiar are nevoie de imagine cu fundal
            # eliminat (RGBA) — InstantMesh nu, vezi run_instantmesh().
            print("[2/4] Segmentare SAM (doar prima imagine)...")
            seg_dir = os.path.join(tmp_dir, "segmented")
            os.makedirs(seg_dir)
            segmented_paths = run_sam_segmentation(image_paths[:1], seg_dir)

            print("[3/4] Generare 3D cu InstantMesh...")
            mesh_dir = os.path.join(tmp_dir, "instantmesh_output")
            os.makedirs(mesh_dir)
            # Imaginea ORIGINALĂ (nu segmented_paths[0]) — vezi motivul detaliat
            # în docstring-ul run_instantmesh().
            obj_path = run_instantmesh(image_paths[0], mesh_dir)

            print("[4/4] Export .glb cu trimesh...")
            glb_path = os.path.join(tmp_dir, f"{product_id}.glb")
            run_trimesh_export(obj_path, glb_path)

            print("Upload .glb în Supabase...")
            filename = f"{product_id}.glb"
            public_url = upload_to_supabase(glb_path, "product-models", filename)

            return {
                "success": True,
                "product_id": product_id,
                "job_id": job_id,
                "model_url": public_url
            }

    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "product_id": product_id,
            "job_id": job_id,
            "error": str(e)
        }

runpod.serverless.start({"handler": handler})
