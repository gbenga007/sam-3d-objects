# SAM3D CUDA 12.8 Base Image

This image is intentionally independent of any one checkout. It bakes in the slow and fragile CUDA packages for a SAM3D-style environment:

- CUDA 12.8 devel image with cuDNN
- Python 3.11
- PyTorch 2.8.0, torchvision 0.23.0, torchaudio 2.8.0 from the official `cu128` PyTorch index
- Kaolin 0.18.0 from NVIDIA's `torch-2.8.0_cu128` wheel index
- flash-attn 2.8.3
- PyTorch3D built from source
- Build/runtime utilities: `build-essential`, `cmake`, `ninja`, `git`, `curl`, `wget`, `ffmpeg`, `rsync`, `psmisc`, `procps`, `openssh-client`, `tmux`, `vim`, `less`
- Extra Python packages: `opencv-python-headless`, `shortuuid`, `colour`, `kornia`
- SAM3D support packages from the repo requirements, with CUDA/Torch pins updated or excluded so they do not downgrade Torch 2.8/cu128
- `utils3d` from `EasternJournalist/utils3d`
- `MoGe` from the same pinned Microsoft Git commit used by the SAM3D requirements
- `gsplat` from the pinned nerfstudio Git commit used by the SAM3D inference requirements
- Node.js 24.x and `npm@latest`

Notes on intentionally changed requirements:

- `kaolin==0.17.0` is replaced with `kaolin==0.18.0` because NVIDIA publishes a `torch-2.8.0_cu128` wheel for that version.
- `torchaudio==2.5.1+cu121` is replaced with `torchaudio==2.8.0` from the official PyTorch cu128 index.
- `cuda-python==12.1.0` is replaced with `cuda-python>=12.8,<13`.
- `nvidia-cuda-nvcc-cu12==12.1.105` is skipped because the base image already includes CUDA 12.8 `nvcc`.
- `opencv-python==4.9.0.80` is replaced with `opencv-python-headless`.
- `spconv-cu121` is replaced with `spconv-cu126`, the closest available CUDA 12 package line found for current spconv wheels.
- `dataclasses==0.6` is skipped because Python 3.11 already includes `dataclasses`.
- `auto_gptq==0.7.1` is skipped for now because it is not used by SAM3D imports found in this repo and is a common source of stale Torch/CUDA constraints.
- FlexiCubes is not installed as a separate package: SAM3D vendors its own `FlexiCubes` implementation, and Kaolin 0.18 also includes FlexiCubes functionality.

Build:

```bash
docker build -t sam3d-cu128:torch2.8 docker/sam3d-cu128
```

Run with a local SAM3D checkout mounted:

```bash
docker run --rm -it --gpus all --shm-size=16g \
  -v "$PWD":/workspace/sam-3d-objects \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace/sam-3d-objects \
  sam3d-cu128:torch2.8
```

Inside the container, install only the lightweight project layer:

```bash
pip install -e .
```

Avoid installing SAM3D's old CUDA-pinned requirements blindly if they still pin `cu121` or older torch packages. Keep torch, torchvision, torchaudio, kaolin, flash-attn, and pytorch3d owned by the image.
