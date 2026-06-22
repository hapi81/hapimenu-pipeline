import runpod
import os
import requests
import tempfile
import subprocess
import json
from pathlib import Path

def download_image(url, path):
    """Descarcă o imagine de la URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(path, 'wb') as f:
        f.write(response.content)

def upload_to_supabase(file_path, bucket, filename):
    """Urcă fișierul .glb în Supabase Storage."""
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
    """Rulează SAM pe imagini pentru a elimina fundalul."""
    segmented = []
    
    for i, img_path in enumerate(image_paths):
        out_path = os.path.join(output_dir, f"seg_{i}.png")
        
        # SAM segmentation via Python script
        result = subprocess.run([
            "python", "/app/sam_segment.py",
            "--input", img_path,
            "--output", out_path
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise Exception(f"SAM eroare: {result.stderr}")
        
        segmented.append(out_path)
    
    return segmented

def run_triposr(image_paths, output_dir):
    """Generează modelul 3D cu TripoSR din imagini segmentate."""
    main_image = image_paths[0]  # TripoSR folosește imaginea principală (față)
    
    result = subprocess.run([
        "python", "/app/TripoSR/run.py",
        main_image,
        "--output-dir", output_dir,
        "--no-remove-bg"  # SAM a făcut deja segmentarea
    ], capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        raise Exception(f"TripoSR eroare: {result.stderr}")
    
    # TripoSR salvează modelul ca mesh.obj
    obj_path = os.path.join(output_dir, "0", "mesh.obj")
    if not os.path.exists(obj_path):
        raise Exception("TripoSR nu a generat mesh.obj")
    
    return obj_path

def run_blender_cleanup(obj_path, output_glb):
    """Curăță modelul și exportă .glb cu Blender headless."""
    blender_script = "/app/blender_cleanup.py"
    
    result = subprocess.run([
        "blender", "--background", "--python", blender_script,
        "--", obj_path, output_glb
    ], capture_output=True, text=True, timeout=180)
    
    if result.returncode != 0:
        raise Exception(f"Blender eroare: {result.stderr}")
    
    if not os.path.exists(output_glb):
        raise Exception("Blender nu a generat .glb")
    
    return output_glb

def handler(job):
    """
    Handler principal RunPod.
    
    Input JSON:
    {
        "input": {
            "product_id": "uuid",
            "job_id": "uuid", 
            "images": ["url1", "url2", "url3", "url4", "url5"]
        }
    }
    """
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
            # 1. Descarcă imaginile
            print(f"[1/4] Descărcare imagini pentru produsul {product_id}...")
            image_paths = []
            for i, url in enumerate(image_urls):
                path = os.path.join(tmp_dir, f"image_{i}.jpg")
                download_image(url, path)
                image_paths.append(path)
                print(f"  Imagine {i+1}/5 descărcată")
            
            # 2. SAM segmentation
            print("[2/4] Segmentare imagini cu SAM...")
            seg_dir = os.path.join(tmp_dir, "segmented")
            os.makedirs(seg_dir)
            segmented_paths = run_sam_segmentation(image_paths, seg_dir)
            
            # 3. TripoSR generare 3D
            print("[3/4] Generare model 3D cu TripoSR...")
            triposr_dir = os.path.join(tmp_dir, "triposr_output")
            os.makedirs(triposr_dir)
            obj_path = run_triposr(segmented_paths, triposr_dir)
            
            # 4. Blender cleanup + export .glb
            print("[4/4] Curățare și export .glb cu Blender...")
            glb_path = os.path.join(tmp_dir, f"{product_id}.glb")
            run_blender_cleanup(obj_path, glb_path)
            
            # 5. Upload în Supabase
            print("Uploading .glb în Supabase...")
            filename = f"{product_id}.glb"
            public_url = upload_to_supabase(glb_path, "product-models", filename)
            
            print(f"Gata! Model disponibil la: {public_url}")
            
            return {
                "success": True,
                "product_id": product_id,
                "job_id": job_id,
                "model_url": public_url
            }
    
    except Exception as e:
        print(f"EROARE: {str(e)}")
        return {
            "success": False,
            "product_id": product_id,
            "job_id": job_id,
            "error": str(e)
        }

# Pornire RunPod worker
runpod.serverless.start({"handler": handler})
