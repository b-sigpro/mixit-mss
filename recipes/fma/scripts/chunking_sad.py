#!/usr/bin/env python3

# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT


from argparse import ArgumentParser
import json
from math import ceil
from pathlib import Path

from progressbar import ProgressBar

import numpy as np

import librosa
import soundfile as sf


class EmptySample(Exception):
    pass


def chunk_sad(segment, num_chunks=20):
    num_zeros = 0
    chunk_size = len(segment) // num_chunks
    for i in range(num_chunks):
        chunk = segment[i * chunk_size : (i + 1) * chunk_size]
        energy = np.sum(chunk**2)  # / len(chunk)
        if energy == 0:
            num_zeros += 1

    return not num_zeros >= num_chunks // 2


def make_dataset(args, unk_args):
    import h5py
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("================================")
        print("Parameters")
        print("--------------------------------")
        for key, val in args.__dict__.items():
            print(f"{key:20s}: {val}")
        print("================================")

    # obtain file list
    if rank == 0:
        # load train/dev/test split metadata
        with open(args.split_json) as f:
            fma_split = json.load(f)
        # all paths
        mix_paths = list(args.data_dir.rglob("*.mp3"))
        # paths to either of train/dev/test
        mix_paths = [p for p in mix_paths if p.name in fma_split[args.split]]
        mix_paths += (ceil(len(mix_paths) / size) * size - len(mix_paths)) * [None]
    else:
        mix_paths = None
    mix_paths = comm.bcast(mix_paths, root=0)

    hdf_name = args.hdf5_output_path
    with h5py.File(hdf_name, "w", driver="mpio", comm=comm) as f:
        pbar = ProgressBar(redirect_stdout=True) if rank == 0 else lambda x: x
        for idx, mix_path in enumerate(pbar(mix_paths[rank::size])):
            # load spectrogram
            try:
                if mix_path is None:
                    print(f"{mix_path} does not exist")
                    raise EmptySample()

                wav, sr = sf.read(mix_path, always_2d=True)

                if sr != 44100:
                    if sr > 44100:
                        wav = librosa.resample(wav, orig_sr=sr, target_sr=44100, axis=0)
                    else:
                        print("Skip a sample with sample rate of: ", sr)
                        raise EmptySample()
                sr = 44100

                n_samples, n_mic = wav.shape
                if n_samples < sr * args.duration:
                    raise EmptySample()

                if n_mic == 1:
                    print("Skip monaural sample: ", mix_path)
                    raise EmptySample()
                assert n_mic == 2, wav.shape

                shift = args.duration if args.shift is None else args.shift

                ts, grp_names = [], []
                for tidx, t in enumerate(range(0, n_samples, sr * shift)):
                    if abs(wav[t : t + sr * args.duration]).sum() == 0.0:
                        raise EmptySample()

                    if not chunk_sad(wav[t : t + sr * args.duration]):
                        print("Skip too silent signal")
                        raise EmptySample()

                    if t + sr * args.duration <= n_samples:
                        ts.append(t)
                    else:
                        ts.append(max(0, n_samples - sr * args.duration))

                    grp_names.append(f"{size*idx + rank:08d}-{tidx:03d}")

            except EmptySample:
                ts, grp_names = [], []
            except Exception as e:
                print(e)
                ts, grp_names = [], []

            # initialize datasets
            all_grp_names = sum(comm.allgather(grp_names), [])
            for grp_name in all_grp_names:
                _ = f.create_group(grp_name).create_dataset("wav", [n_mic, sr * args.duration], "float32")

            for t, grp_name in zip(ts, grp_names):
                f[f"{grp_name}/wav"][...] = wav[t : t + sr * args.duration].T


def main():
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--split_json", type=Path, required=True)
    parser.add_argument("--hdf5_output_path", type=Path, required=True)
    parser.add_argument("--split", type=str, choices=["train", "dev", "test"], required=True)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--shift", type=int, default=None)

    args, unk_args = parser.parse_known_args()
    make_dataset(args, unk_args)


if __name__ == "__main__":
    main()
