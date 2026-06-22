"""
sam_segment.py — elimină fundalul din imagini folosind Meta SAM
Folosire: python sam_segment.py --input input.jpg --output output.png
"""
import argparse
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

SAM_CHECKPOINT = "/app/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

def segment_food(input_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM rulează pe: {device}")
    
    # Încarcă modelul SAM
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=500,
    )
    
    # Procesează imaginea
    image = np.array(Image.open(input_path).convert("RGB"))
    masks = mask_generator.generate(image)
    
    if not masks:
        # Dacă SAM nu găsește nimic, returnează imaginea originală
        Image.open(input_path).save(output_path)
        return
    
    # Selectează masca centrală (obiectul principal = mâncarea)
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    best_mask = None
    best_score = -1
    
    for mask in masks:
        bbox = mask['bbox']  # x, y, w, h
        mask_cx = bbox[0] + bbox[2] / 2
        mask_cy = bbox[1] + bbox[3] / 2
        
        dist = ((mask_cx - center_x)**2 + (mask_cy - center_y)**2)**0.5
        area = mask['area']
        score = area / (dist + 1)
        
        if score > best_score:
            best_score = score
            best_mask = mask['segmentation']
    
    # Aplică masca (fundal transparent)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = image
    rgba[:, :, 3] = (best_mask * 255).astype(np.uint8)
    
    Image.fromarray(rgba, 'RGBA').save(output_path)
    print(f"Segmentare completă: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    segment_food(args.input, args.output)
