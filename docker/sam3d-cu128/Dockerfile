# syntax=docker/dockerfile:1.7

ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCHAUDIO_VERSION=2.8.0
ARG KAOLIN_VERSION=0.18.0
ARG FLASH_ATTN_VERSION=2.8.3
ARG XFORMERS_VERSION=0.0.32.post2
ARG PYTORCH3D_REF=main
ARG MAMBA_VERSION=24.11.3-0
ARG NODE_MAJOR=24

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_DIR=/opt/conda \
    ENV_NAME=sam3d \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    MAX_JOBS=4 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

ENV PATH=${CONDA_DIR}/envs/${ENV_NAME}/bin:${CONDA_DIR}/bin:${PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        less \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ninja-build \
        openssh-client \
        pkg-config \
        procps \
        psmisc \
        rsync \
        tmux \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g npm@latest \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/mambaforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Miniforge3-${MAMBA_VERSION}-Linux-x86_64.sh" \
    && bash /tmp/mambaforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/mambaforge.sh \
    && mamba create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" pip \
    && mamba clean -a -y

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja cmake pybind11 \
    && python -m pip install \
        torch=="${TORCH_VERSION}" \
        torchvision=="${TORCHVISION_VERSION}" \
        torchaudio=="${TORCHAUDIO_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install \
        colour \
        kornia \
        opencv-python-headless \
        shortuuid

COPY requirements-sam3d-base.txt /tmp/requirements-sam3d-base.txt

RUN python -m pip install -r /tmp/requirements-sam3d-base.txt \
    && python -m pip install \
        "git+https://github.com/EasternJournalist/utils3d.git" \
        "git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" \
        "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"

RUN python -m pip install \
        kaolin=="${KAOLIN_VERSION}" \
        -f "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${TORCH_VERSION}_cu128.html"

# FlashAttention and PyTorch3D are installed after torch so their CUDA extensions
# compile against the exact torch/CUDA ABI in this image.
RUN python -m pip install flash-attn=="${FLASH_ATTN_VERSION}" --no-build-isolation

RUN python -m pip install \
        xformers=="${XFORMERS_VERSION}" \
        --no-deps

RUN python -m pip install spconv-cu126

RUN python -m pip install iopath fvcore \
    && python -m pip install \
        "git+https://github.com/facebookresearch/pytorch3d.git@${PYTORCH3D_REF}"

RUN python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
import kaolin
print("kaolin", kaolin.__version__)
import flash_attn
print("flash_attn", flash_attn.__version__)
import pytorch3d
print("pytorch3d", pytorch3d.__version__)
import utils3d
print("utils3d", utils3d.__file__)
import gsplat
print("gsplat", gsplat.__file__)
PY

WORKDIR /workspace

CMD ["/bin/bash"]
