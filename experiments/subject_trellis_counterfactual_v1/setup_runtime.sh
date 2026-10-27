#!/usr/bin/env bash
# Isolated, pinned TRELLIS dependencies; never change an existing environment.
set -euo pipefail
test -z "$(git status --porcelain)"
ASSETS=/var/tmp/colorpeel-r12user5/assets/trellis2_v1
BASE=/home/r12user5/miniforge3/envs/univlr/bin/python
test ! -e "$ASSETS"
mkdir -p "$ASSETS/sources" "$ASSETS/build" "$ASSETS/cache"
exec > >(tee "$ASSETS/setup.log") 2>&1
git rev-parse HEAD
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/var/tmp
export CUDA_HOME=/usr/local/cuda-12.4 TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=4
export PIP_CACHE_DIR="$ASSETS/cache/pip" TORCH_EXTENSIONS_DIR="$ASSETS/cache/torch_extensions"
"$BASE" -m venv --system-site-packages "$ASSETS/env"
PY="$ASSETS/env/bin/python"
"$PY" -B -m pip install 'transformers==4.57.3' 'tokenizers==0.22.1' easydict ninja imageio-ffmpeg kornia zstandard trimesh scipy imageio einops plyfile 'opencv-python-headless==4.11.0.86'

clone_pinned() {
    local name=$1 url=$2 revision=$3
    git clone --filter=blob:none --no-checkout "$url" "$ASSETS/sources/$name"
    git -C "$ASSETS/sources/$name" checkout --detach "$revision"
    if test -f "$ASSETS/sources/$name/.gitmodules"; then
        git -C "$ASSETS/sources/$name" submodule update --init --recursive --depth 1
    fi
    git -C "$ASSETS/sources/$name" rev-parse HEAD
}
build_package() {
    local name=$1
    mkdir "$ASSETS/build/$name"
    cp -a "$ASSETS/sources/$name/." "$ASSETS/build/$name/"
    "$PY" -B -m pip install --no-build-isolation --no-deps "$ASSETS/build/$name"
}

git clone --filter=blob:none --sparse --no-checkout https://github.com/microsoft/TRELLIS.2.git "$ASSETS/sources/TRELLIS.2"
git -C "$ASSETS/sources/TRELLIS.2" sparse-checkout set trellis2 o-voxel assets/hdri
git -C "$ASSETS/sources/TRELLIS.2" checkout --detach 75fbf0183001ed9876c8dbb35de6b68552ee08bd
git -C "$ASSETS/sources/TRELLIS.2" submodule update --init --recursive --depth 1

clone_pinned utils3d https://github.com/EasternJournalist/utils3d.git 9a4eb15e4021b67b12c460c7057d642626897ec8
build_package utils3d
clone_pinned nvdiffrast https://github.com/NVlabs/nvdiffrast.git 253ac4fcea7de5f396371124af597e6cc957bfae
build_package nvdiffrast
clone_pinned nvdiffrec https://github.com/JeffreyXiang/nvdiffrec.git b296927cc7fd01c2ac1087c8065c4d7248f72da4
build_package nvdiffrec
clone_pinned CuMesh https://github.com/JeffreyXiang/CuMesh.git 12289e1062f0603f2f0d0771b02e1395d247f26f
build_package CuMesh
clone_pinned FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git 6dd94a859c26ee8246888502eada3dd8ad85532e
# Upstream setup seeds this cache. Preserve any existing cache instead of overwriting it.
test ! -e /home/r12user5/.flex_gemm/autotune_cache.json
build_package FlexGEMM
mkdir "$ASSETS/build/o-voxel"
cp -a "$ASSETS/sources/TRELLIS.2/o-voxel/." "$ASSETS/build/o-voxel/"
"$PY" -B -m pip install --no-build-isolation --no-deps "$ASSETS/build/o-voxel"
"$PY" -B -m pip freeze --local > "$ASSETS/installed-local.txt"
"$PY" -B -c 'import torch, triton, cumesh, flex_gemm, o_voxel, nvdiffrast, nvdiffrec_render; print("dependency_imports_ok", torch.__version__, triton.__version__)'
printf 'Setup finished; complete inference compatibility remains unverified.\n'
