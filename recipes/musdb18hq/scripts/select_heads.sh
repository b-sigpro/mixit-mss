#! /bin/bash

# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

model_path=$1

dataset_path="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)/segmented_wav/dev_seg6sec_shift6sec"

if [[ "$model_path" =~ \.(ckpt|pth|pt)$ ]]; then
    wav_output_path="$(dirname "$(dirname "$model_path")")"
else
    wav_output_path=$model_path
fi
wav_output_path="${wav_output_path}/dev_seg6sec_shift6sec"

# Separation
if [ ! -e $wav_output_path ]; then
    mkdir -p $wav_output_path

    python -m mixit_mss.mixit.separate batch \
        $model_path \
        "${dataset_path}" \
        "${wav_output_path}" \
        --ext wav
else
    echo "${wav_output_path} exists."
    echo "Please delete it if you want to run separation again."
fi

# Evaluation
if [ ! -f "${wav_output_path}/inst_head.json" ]; then
    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m recipes.musdb18hq.scripts.evaluate_snr_pit  \
        --est_dir "${wav_output_path}" \
        --ref_dir "${dataset_path}"  \
        --verbose
fi

model_output_path="$(dirname "${model_path}")/model_4heads.pth"
if [ ! -f "${model_output_path}" ]; then
    python -m recipes.musdb18hq.scripts.select_heads \
        --input_path "${model_path}" \
        --output_path "${model_output_path}" \
        --head_json ${wav_output_path}/inst_head.json
fi