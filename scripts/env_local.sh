#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mpl_cache}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.torch_cache}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/.pip_cache}"
export DYSTREAM_WAV2VEC_PATH="${DYSTREAM_WAV2VEC_PATH:-pretrained_model/wav2vec2-base-960h}"

export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7891}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7891}"
export ALL_PROXY="${ALL_PROXY:-socks5://127.0.0.1:7891}"

mkdir -p "$MPLCONFIGDIR" "$TORCH_HOME" "$HF_HOME" "$PIP_CACHE_DIR"
