#!/usr/bin/env bash

# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

size=$1  # small, medium, large, full
np=${2:-1}  # number of parallel processes in mpirun

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
recipe_dir="$(dirname "$script_dir")"
output_dir=${recipe_dir}/data
mkdir -p $output_dir

seg_len=10
shift=5

# metadata download
if [ ! -f "${output_dir}/fma_metadata.zip" ]; then
    wget -c -O "${output_dir}/fma_metadata.zip" "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
    echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  ${output_dir}/fma_metadata.zip" | sha1sum -c -
fi

# unzip
if [ ! -e "${output_dir}/fma_metadata" ]; then
    unzip "${output_dir}/fma_metadata.zip" -d "${output_dir}"
fi

# dataset download
if [ ! -f "${output_dir}/fma_${size}.zip" ]; then
    wget -c -O "${output_dir}/fma_${size}.zip"  "https://os.unil.cloud.switch.ch/fma/fma_${size}.zip"

    # checksum, may take quite some time when checking the large or full subset
    if [ "$size" = "small" ]; then
        echo "ade154f733639d52e35e32f5593efe5be76c6d70  ${output_dir}/fma_${size}.zip" | sha1sum -c -
    elif [ "$size" = "medium" ]; then
        echo "c67b69ea232021025fca9231fc1c7c1a063ab50b  ${output_dir}/fma_${size}.zip" | sha1sum -c -
    elif [ "$size" = "large" ]; then
        echo "497109f4dd721066b5ce5e5f250ec604dc78939e  ${output_dir}/fma_${size}.zip" | sha1sum -c -
    elif [ "$size" = "full" ]; then
        echo "0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab  ${output_dir}/fma_${size}.zip" | sha1sum -c -
    fi
fi

# unzip
if [ ! -e "${output_dir}/fma_${size}" ]; then
    unzip "${output_dir}/fma_${size}.zip" -d "${output_dir}"
fi


# generate a json file including subset of fma data paths
json_output_dir=${output_dir}/../metadata
mkdir -p ${json_output_dir}
if [ ! -e "${json_output_dir}/fma_split_${size}.json" ]; then
    python ${script_dir}/fma_split.py \
        --csv_path ${output_dir}/fma_metadata/tracks.csv \
        --output_dir ${json_output_dir} \
        --size ${size}
fi

hdf5_output_dir=${output_dir}/../hdf5
mkdir -p ${hdf5_output_dir}
for split in dev; do
    hdf5_path="${hdf5_output_dir}/fma_${size}_${split}_seg${seg_len}sec.hdf5"
    if [ ! -f "${hdf5_path}" ]; then
        echo "Save HDF5 to ${hdf5_path}"
        MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 mpirun --mca mpi_warn_on_fork 0 -np ${np} python ${script_dir}/chunking_sad.py \
            --data_dir "${output_dir}/fma_${size}" \
            --split_json "${json_output_dir}/fma_split_${size}.json" \
            --hdf5_output_path "${hdf5_path}" \
            --split $split \
            --duration $seg_len --shift $shift
        # cp "${hdf5_path}" ${output_dir}/hdf5_striped
    else
        echo "${hdf5_path} already exists. Skip making HDF5."
    fi
done