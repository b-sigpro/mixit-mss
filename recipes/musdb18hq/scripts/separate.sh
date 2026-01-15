#! /bin/bash

# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

model_path=$1
css_segment_size=${2:-12}
css_shift_size=${3:-6}

split=test
dataset_path="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)/data/${split}"

metrics=("usdr" "csdr")
# metrics=("usdr")

# Define path to save wav files
if [[ "$model_path" =~ \.(ckpt|pth|pt)$ ]]; then
    wav_output_path="$(dirname "$(dirname "$model_path")")"
else
    wav_output_path=$model_path
fi
wav_output_path="${wav_output_path}/wav_css${css_segment_size}-${css_shift_size}/${split}"

# Separation
if [ ! -e $wav_output_path ]; then
    mkdir -p $wav_output_path

   python -m mixit_mss.mixit.separate batch \
        $model_path \
        "${dataset_path}" \
        "${wav_output_path}" \
        --ext wav \
        --css_segment_size ${css_segment_size} --css_shift_size ${css_shift_size}
else
    echo "${wav_output_path} exists."
    echo "Please delete it if you want to run separation again."
fi

# Evaluation
for metric in ${metrics[@]}; do
    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m recipes.musdb18hq.scripts.evaluate_sdr \
        --est_dir "${wav_output_path}" \
        --ref_dir "${dataset_path}"  \
        --metric ${metric} \
        --verbose
done