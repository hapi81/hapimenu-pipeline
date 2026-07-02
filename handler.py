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
        if result.returncode != 0:
            raise Exception(f"SAM eroare: {result.stderr}")
        segmented.append(out_path)
    return segmented

def run_shape_e(image_path, output_dir):
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

def run_trimesh_export(obj_path, output_glb):
    """Exportă .obj → .glb folosind trimesh (fără Blender)."""
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

    if len(image_urls) != 5:
        return {"error": f"Necesare 5 imagini, primite {len(image_urls)}"}
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
                print(f"  Imagine {i+1}/5 descărcată")

            print("[2/4] Segmentare SAM...")
            seg_dir = os.path.join(tmp_dir, "segmented")
            os.makedirs(seg_dir)
            segmented_paths = run_sam_segmentation(image_paths, seg_dir)

            print("[3/4] Generare 3D cu Shap-E...")
            shape_dir = os.path.join(tmp_dir, "shape_output")
            os.makedirs(shape_dir)
            obj_path = run_shape_e(segmented_paths[0], shape_dir)

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
