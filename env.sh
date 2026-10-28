#!/usr/bin/env bash
# Source this file: source /root/workspace/muxpp_work/env.sh
source /root/venvs/sgl/bin/activate
# Make the editable package win over the repository directory when cwd is /root/workspace.
case ":${PYTHONPATH:-}:" in
  *:/root/workspace/sglang/python:*) ;;
  *) export PYTHONPATH="/root/workspace/sglang/python${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=16
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FLASHINFER_WORKSPACE_BASE=/root/workspace/muxpp_work/cache/flashinfer
export TORCH_EXTENSIONS_DIR=/root/workspace/muxpp_work/cache/torch_extensions
export TRITON_CACHE_DIR=/root/workspace/muxpp_work/cache/triton
