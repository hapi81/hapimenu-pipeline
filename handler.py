import sys
import traceback

try:
    import runpod
    print("runpod ok")
    import os
    import requests
    print("requests ok")
    import tempfile
    import subprocess
    import torch
    print(f"torch ok: {torch.__version__}")
    import numpy as np
    from PIL import Image
    print("PIL ok")
except Exception as e:
    print(f"IMPORT ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

def download_image(url, path):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(path, 'wb') as f:
        f.write(response.content)

def upload_to_supabase(file_path, bucket, filename):
    supabase_url = os.environ.get("SUPABASE_URL")
    service_key = os.environ.get("SUPABASE_SERVICE_KEY")
    url = f"{supabase_url}/storage/v1/object/{bucket}/{filename}"
    with open(file_path, 'rb') as f:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "model/gltf-binary"
            },
            data=f
        )
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
    try:
        from shap_e.diffusion.sample import sample_latents
        from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
        from shap_e.models.download import load_model, load_config
        from shap_e.util.notebooks import decode_latent_mesh
        print("shap_e imports ok")
    except Exception as e:
        raise Exception(f"Shap-E import error: {e}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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

def run_blender_cleanup(obj_path, output_glb):
    result = subprocess.run([
        "blender", "--background", "--python", "/app/blender_cleanup.py",
        "--", obj_path, output_glb
    ], capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise Exception(f"Blender eroare: {result.stderr}")
    if not os.path.exists(output_glb):
        raise Exception("Blender nu a generat .glb")
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

            print("[2/4] Segmentare SAM...")
            seg_dir = os.path.join(tmp_dir, "segmented")
            os.makedirs(seg_dir)
            segmented_paths = run_sam_segmentation(image_paths, seg_dir)

            print("[3/4] Generare 3D cu Shap-E...")
            shape_dir = os.path.join(tmp_dir, "shape_output")
            os.makedirs(shape_dir)
            obj_path = run_shape_e(segmented_paths[0], shape_dir)

            print("[4/4] Export .glb cu Blender...")
            glb_path = os.path.join(tmp_dir, f"{product_id}.glb")
            run_blender_cleanup(obj_path, glb_path)

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

print("Handler loaded, starting runpod...")
runpod.serverless.start({"handler": handler})
