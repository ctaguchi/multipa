#!/bin/bash

set -e  

PYTHON_VERSION="3.9.20"
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
  pyenv install $PYTHON_VERSION
fi
pyenv local $PYTHON_VERSION
eval "$(pyenv init --path)"   

python -m venv venv
source venv/bin/activate

echo "Instalando dependências..."
pip install --upgrade pip
pip install --use-deprecated=legacy-resolver -r requirements.txt

echo "Ambiente ativado"

source .env

export HF_HOME="$HF_HOME"
export TORCH_USE_HIP_DSA=1 
# export AMD_SERIALIZE_KERNEL=3 
# export TORCH_CUDA_ARCH_LIST="gfx1032"
export TORCH_CUDA_ARCH_LIST="gfx1030"   
export ROCM_TENSILE_ARCH=gfx1030
export HSA_OVERRIDE_GFX_VERSION=10.3.2  

# After installing via pip, you need to download the dictionary using the following command:
# python -m unidic download

# pip install fsspec==2023.9.2
#pip install -U datasets
# https://stackoverflow.com/questions/77433096/notimplementederror-loading-a-dataset-cached-in-a-localfilesystem-is-not-suppor

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm4.2

# sudo usermod -aG render lain
# sudo usermod -aG video lain


# python preprocess.py \                
#     -l ja pl mt hu fi el \
#     --num_proc 2 \
#     --cache_dir "$CACHE_DIR" \
#     --clear_cache