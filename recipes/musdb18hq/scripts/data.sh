#!/usr/bin/env bash

# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
recipe_dir="$(dirname "$script_dir")"
output_dir=${recipe_dir}/data
mkdir -p $output_dir

dev_names=(
    "Actions - One Minute Smile"
    "Clara Berry And Wooldog - Waltz For My Victims"
    "Johnny Lokke - Promises & Lies"
    "Patrick Talbot - A Reason To Leave"
    "Triviul - Angelsaint"
    "Alexander Ross - Goodbye Bolero"
    "Fergessen - Nos Palpitants"
    "Leaf - Summerghost"
    "Skelpolu - Human Mistakes"
    "Young Griffo - Pennies"
    "ANiMAL - Rockshow"
    "James May - On The Line"
    "Meaxic - Take A Step"
    "Traffic Experiment - Sirens"
)

# download
if [ ! -f "${output_dir}/musdb18hq.zip" ]; then
    wget -c -O "${output_dir}/musdb18hq.zip"  "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
fi

# unzip
if [[ ! -e "${output_dir}/train" || ! -e "${output_dir}/test" ]]; then
    unzip "${output_dir}/musdb18hq.zip" -d "${output_dir}"
fi

# split train and dev
if [ ! -e "${output_dir}/dev" ]; then
    mkdir -p "${output_dir}/dev"
    for name in "${dev_names[@]}"; do
        mv "${output_dir}/train/${name}" "${output_dir}/dev/${name}"
    done
fi

hdf5_output_dir=${output_dir}/../hdf5
mkdir -p ${hdf5_output_dir}
# mkdir -p ${output_dir}/hdf5_striped
# lfs setstripe -S 1m -c -1 -i -1 "${subset_dir}/hdf5_striped"

wav_output_dir=${output_dir}/../segmented_wav
mkdir -p ${wav_output_dir}


# apply source activity detection to each instrument
seg_len=8
for split in train; do
    for inst in drums bass vocals other; do
        basename=${split}_${inst}_seg${seg_len}sec_sad
        hdf5_path=${hdf5_output_dir}/${basename}.hdf5
        if [ ! -f "${hdf5_path}" ]; then
            python ${script_dir}/chunking_sad_inst.py \
                --data_dir "${output_dir}/${split}" \
                --hdf5_output_path "${hdf5_path}" \
                --wav_output_dir "${wav_output_dir}/${basename}" \
                --inst ${inst} \
                --split ${split} \
                --duration ${seg_len}
            # cp "${hdf5_path}" ${output_dir}/hdf5_striped
        fi
    done
done

# segment dev set. This data is used when selecting output channels
seg_len=6
shift=6
for split in dev; do
    hdf5_path=${hdf5_output_dir}/${split}_seg${seg_len}sec_shift${shift}sec.hdf5
    if [ ! -f "${hdf5_path}" ]; then
        python ${script_dir}/chunking.py \
            --data_dir "${output_dir}/${split}" \
            --hdf5_output_path "${hdf5_path}" \
            --wav_output_dir "${wav_output_dir}/${split}_seg${seg_len}sec_shift${shift}sec" \
            --split $split \
            --duration ${seg_len} --shift ${shift}
        # cp "${hdf5_path}" ${output_dir}/hdf5_striped
    fi
done