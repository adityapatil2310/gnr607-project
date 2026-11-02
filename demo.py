import os
import numpy as np
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR / "src"))

from prepare_data import load_and_crop_data, extract_template
from ncc import validate_ncc
from run_experiments import run_all

def main():
    np.random.seed(42)
    print("--- Principles of Satellite Image Processing: NCC Project ---")
    
    scene_id = "c98a544c-4206-4176-b480-35b8bdf5bb14.jpg"
    crop_box = (0, 0, 1500, 1500)
    output_dir = PROJECT_DIR / "data" / "processed"
    
    print("\n[Step 1] Loading Data & Cropping...")
    scene, gt = load_and_crop_data(scene_id, crop_box, output_dir)
    template = extract_template(scene, gt[0], 60, output_dir, "main")
    
    print("\n[Step 2] Validating NCC Integrity...")
    validate_ncc()
    
    print(f"\n[Step 3-5] Running Pipeline")
    print(f"Parameters:\n- Template Size: 60x60 (matches approx aircraft wingspan)\n- Threshold: 0.55 (derived from PR curve knee)\n- NMS Radius: 30 (half template size to prevent double-counting)\n- Match Tolerance: 30 (half template size)")
    run_all(scene, gt, template, PROJECT_DIR / "results")
    
    print("\nPipeline finished successfully. Check the 'results/' folder for outputs.")

if __name__ == "__main__":
    main()