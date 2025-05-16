import os
import sys
import torch
import argparse
import tempfile
import shutil

# MASt3R & Dust3R imports
from mast3r.model import AsymmetricMASt3R
from mast3r.demo import get_reconstructed_scene
# from dust3r.utils.image import load_images
import time


def run_mast3r(image_dir, output_glb_path, weights_path, device='cuda'):
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Validate image list
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if len(image_paths) < 2:
        print("❌ Need at least two images for reconstruction.")
        sys.exit(1)

    print(f"✅ Found {len(image_paths)} images...")

    # Load model
    print("🔄 Loading MASt3R model...")
    print("Loading MASt3R model...")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device).eval()

    # Create temporary working directory
    tmp_dir = tempfile.mkdtemp()
    print(f"📁 Temporary directory: {tmp_dir}")

    start_time = time.time()

    # Run full reconstruction pipeline and save .glb
    print("🚀 Running MASt3R reconstruction...")
    scene_state, glb_path = get_reconstructed_scene(
        outdir=tmp_dir,
        gradio_delete_cache=None,
        model=model,
        retrieval_model=None,
        device=device,
        silent=False,
        image_size=512,
        current_scene_state=None,
        filelist=image_paths,
        optim_level='refine+depth',
        lr1=0.07,
        niter1=300,
        lr2=0.01,
        niter2=300,
        min_conf_thr=1.5,
        matching_conf_thr=0.0,
        as_pointcloud=True,
        mask_sky=False,
        clean_depth=True,
        transparent_cams=False,
        cam_size=0.2,
        scenegraph_type='complete',
        winsize=1,
        win_cyclic=False,
        refid=0,
        TSDF_thresh=0.0,
        shared_intrinsics=False,
    )

    end_time = time.time()
    print(f"✅ Reconstruction completed in {end_time - start_time:.2f} seconds.")

    if os.path.exists(glb_path):
        shutil.move(glb_path, output_glb_path)
        print(f"✅ Exported GLB model to: {output_glb_path}")
    else:
        print("❌ Failed to generate GLB file.")
    # Clean up temporary directory
    shutil.rmtree(tmp_dir)
    print(f"🗑️ Cleaned up temporary directory: {tmp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GLB file using MASt3R")
    parser.add_argument('--image_dir', type=str, required=False, help="Path to directory with input images",default="/home/koustubh/test/")
    parser.add_argument('--output', type=str, default='output_model.glb', help="Path to save the output .glb file")
    parser.add_argument('--weights', type=str, required=False, help="Path to MASt3R weights (.pth)", default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run on: 'cuda' or 'cpu'")

    args = parser.parse_args()
    run_mast3r(args.image_dir, args.output, args.weights, args.device)
