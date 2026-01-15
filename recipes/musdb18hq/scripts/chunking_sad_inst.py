#!/usr/bin/env python3

# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from math import ceil
from pathlib import Path

from progressbar import ProgressBar

import numpy as np

import soundfile as sf


class EmptySample(Exception):
    pass


def source_activity_detection(audio, L, num_chunks=10, min_energy=1e-5, min_thres=1e-3):
    """Source Activity Detection (SAD) for removing silent segment.
    This follows BSRNN [1] (see Section IV-A)


    References:
        [1] Y. Luo, et al., "Music Source Separation with Band-split RNN"
            https://arxiv.org/pdf/2209.15174.pdf
    """

    # 1. split the input audio into L-sample segments with 50% overlap
    hop_size = L // 2
    segments = []
    for start in range(0, len(audio) - L + 1, hop_size):
        segments.append(audio[start : start + L])

    # 2. Split each segment into num_chunks chunks and save energies
    energy_list = []
    num_zeros = 0
    for segment in segments:
        chunk_size = len(segment) // num_chunks
        chunk_energies = []
        for i in range(num_chunks):
            chunk = segment[i * chunk_size : (i + 1) * chunk_size]
            energy = np.sum(chunk**2)  # / len(chunk)
            if energy == 0:
                num_zeros += 1
                energy = min_energy
            chunk_energies.append(energy)
        energy_list.append(chunk_energies)

    # 3. 15% quantile energy is used as threshold in 4.
    all_energies = np.array([p for sublist in energy_list for p in sublist])
    threshold = np.percentile(all_energies, 15)
    if threshold < min_thres:
        threshold = min_thres

    # 4. Keep segments which have more than 50% chunks
    # whose energies are more than threshold
    processed_segments = []
    for seg_idx, chunk_energies in enumerate(energy_list):
        if sum(p > threshold for p in chunk_energies) >= (num_chunks // 2):
            processed_segments.append(segments[seg_idx])

    return processed_segments, len(segments)


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
        mix_paths = list(args.data_dir.rglob("mixture.wav"))
        mix_paths += (ceil(len(mix_paths) / size) * size - len(mix_paths)) * [None]

        args.wav_output_dir.mkdir(exist_ok=True, parents=True)
    else:
        mix_paths = None
    mix_paths = comm.bcast(mix_paths, root=0)

    hdf_name = args.hdf5_output_path
    inst = args.inst
    with h5py.File(hdf_name, "w", driver="mpio", comm=comm) as f:
        pbar = ProgressBar(redirect_stdout=True) if rank == 0 else lambda x: x
        for idx, mix_path in enumerate(pbar(mix_paths[rank::size])):
            # load spectrogram
            try:
                if mix_path is None:
                    raise EmptySample()

                ref_path = mix_path.parent / f"{inst}.wav"
                ref_inst, sr = sf.read(ref_path, always_2d=True)  # (n_samples, n_chan)

                orig_duration, n_mic = ref_inst.shape
                if orig_duration < sr * args.duration:
                    raise EmptySample()

                # segments: list of segmented samples after activity detection
                # each with a shape of (args.duration*sr, n_chan)
                segments, _ = source_activity_detection(ref_inst, args.duration * sr)

                grp_names = []
                for tidx in range(len(segments)):
                    grp_names.append(f"{size*idx + rank:08d}-{tidx:03d}")

            except EmptySample:
                segments, grp_names = [], []
            except Exception as e:
                print(e)
                segments, grp_names = [], []

            # store data
            song_name = mix_path.parent.name
            wav_output_dir = args.wav_output_dir / song_name / inst
            wav_output_dir.mkdir(exist_ok=True, parents=True)

            # initialize datasets
            all_grp_names = sum(comm.allgather(grp_names), [])
            for grp_name in all_grp_names:
                g = f.create_group(grp_name)

                if args.mono:
                    g.create_dataset("wav", [1, sr * args.duration], "float32")
                else:
                    g.create_dataset("wav", [n_mic, sr * args.duration], "float32")

            for tidx, (segment, grp_name) in enumerate(zip(segments, grp_names)):
                if args.mono:
                    segment = segment[:, [0]]

                sf.write(wav_output_dir / f"seg{tidx}.wav", segment, sr)
                f[f"{grp_name}/wav"][:] = np.swapaxes(segment, -1, -2)


def main():
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--hdf5_output_path", type=Path, required=True)
    parser.add_argument("--wav_output_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, choices=["train", "dev", "test"], required=True)
    parser.add_argument("--inst", type=str, required=True)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--mono", action="store_true")

    args, unk_args = parser.parse_known_args()
    make_dataset(args, unk_args)


if __name__ == "__main__":
    main()
