# MASt3R 3D Model Generation

![MASt3R Logo](mast3r/assets/mast3r.jpg)

**MASt3R** (Multi-view Asymmetric Stereo Transformer with Refinement) is a powerful transformer-based framework for generating accurate 3D models from just a few posed RGB images. This repository provides a clean implementation with support for GLB exports, Docker builds, and a test script to run the entire pipeline easily.

---

## 🌟 Features

- 📸 Multi-view stereo reconstruction from as few as **2 images**
- 🤖 Transformer-based feature matching and depth refinement
- 🔧 Automatic camera pose optimization
- 🌍 Outputs textured 3D meshes in **.glb** format
- 🐳 Docker setup for CPU or GPU (CUDA) inference
- 🧪 Ready-to-use test script for one-line model generation

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your_username/mast3r-3d-model-generation.git
cd mast3r-3d-model-generation
```

### 2. Initialize Submodules
```bash
git submodule update --init --recursive
```

### 3. Create a Python Environment and Install Dependencies
```bash
bash build.sh
```

## ⏳ Running the Model
```bash

python3 mast3r_generate_model.py \
    --image_dir mast3r/assets/NLE_tower \
    --output output_model.glb \
    --weights naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
    --device cuda
```

| Flag          | Description                               | Default                                                  |
| ------------- | ----------------------------------------- | -------------------------------------------------------- |
| `--image_dir` | Folder containing at least 2 RGB images   | `/home/koustubh/test/`                                   |
| `--output`    | Path to save `.glb` model                 | `output_model.glb`                                       |
| `--weights`   | Pretrained MASt3R model or path to `.pth` | `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` |
| `--device`    | Choose between `cuda` or `cpu`            | `cuda`                                                   |

## 📦 Pretrained Models

By default, we use:

    naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric

This is auto-downloaded when using the Gradio/Docker scripts. You can also provide a .pth path manually.

## 📄 License

Code is under MIT License.
Model checkpoints and Dust3R fall under their respective licenses (see Dust3R License).

## 🙏 Acknowledgements

[MASt3R](https://github.com/naver/mast3r) by NAVER LABS Europe

Dust3R used for dense matching and camera initialization

Portions of visualization and 3D rendering adapted from their official demos

