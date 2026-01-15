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
    with h5py.File(hdf_name, "w", driver="mpio", comm=comm) as f:
        pbar = ProgressBar(redirect_stdout=True) if rank == 0 else lambda x: x
        for idx, mix_path in enumerate(pbar(mix_paths[rank::size])):
            # load spectrogram
            try:
                if mix_path is None:
                    raise EmptySample()

                wav, sr = sf.read(mix_path, always_2d=True)
                duration, n_mic = wav.shape

                duration = wav.shape[0]
                if duration < sr * args.duration:
                    raise EmptySample()

                shift = args.duration if args.shift is None else args.shift

                ts, grp_names = [], []
                for tidx, t in enumerate(range(0, duration, sr * shift)):
                    if abs(wav[t : t + sr * args.duration, [0]]).sum() == 0.0:
                        raise EmptySample()

                    if t + sr * args.duration <= duration:
                        ts.append(t)
                    else:
                        ts.append(max(0, duration - sr * args.duration))

                    grp_names.append(f"{size*idx + rank:08d}-{tidx:03d}")
            except EmptySample:
                ts, grp_names = [], []
            except Exception as e:
                print(e)
                ts, grp_names = [], []

            # store data
            song_name = mix_path.parent.name
            wav_output_dir = args.wav_output_dir / song_name
            wav_output_dir.mkdir(exist_ok=True, parents=True)

            # initialize datasets
            all_grp_names = sum(comm.allgather(grp_names), [])
            for grp_name in all_grp_names:
                g = f.create_group(grp_name)

                if args.mono:
                    g.create_dataset("wav", [1, sr * args.duration], "float32")
                    g.create_dataset("ref", [4, 1, sr * args.duration], "float32")
                else:
                    g.create_dataset("wav", [n_mic, sr * args.duration], "float32")
                    g.create_dataset("ref", [4, n_mic, sr * args.duration], "float32")

            for tidx, (t, grp_name) in enumerate(zip(ts, grp_names)):
                wavt = wav[t : t + sr * args.duration]
                if args.mono:
                    wavt = wav[:, [0]]
                f[f"{grp_name}/wav"][:] = wavt.T

                wav_output_dir_seg = wav_output_dir / f"seg{tidx}"
                wav_output_dir_seg.mkdir(exist_ok=True, parents=True)
                sf.write(wav_output_dir_seg / "mixture.wav", wavt, sr)

                ref = []
                for inst in ["bass", "drums", "vocals", "other"]:
                    ref_path = mix_path.parent / f"{inst}.wav"
                    ref_inst, sr = sf.read(ref_path, start=t, stop=t + sr * args.duration, always_2d=True)
                    if args.mono:
                        ref_inst = ref_inst[:, [0]]

                    sf.write(wav_output_dir_seg / f"{inst}.wav", ref_inst, sr)

                    # ref.append(ref_inst[:, [0]])
                    ref.append(ref_inst)
                ref = np.stack(ref, axis=0)
                f[f"{grp_name}/ref"][:] = np.swapaxes(ref, -1, -2)

    if args.split != "train":
        hdf_name = args.hdf5_output_path.parent / f"{args.split}_unsegmented.hdf5"
        with h5py.File(hdf_name, "w", driver="mpio", comm=comm) as f:
            pbar = ProgressBar(redirect_stdout=True) if rank == 0 else lambda x: x
            for idx, mix_path in enumerate(pbar(mix_paths[rank::size])):
                # load spectrogram
                if mix_path is None:
                    raise EmptySample()

                wav, sr = sf.read(mix_path, always_2d=True)
                duration, n_mic = wav.shape

                ref = []
                for inst in ["bass", "drums", "vocals", "other"]:
                    ref_path = mix_path.parent / f"{inst}.wav"
                    ref_inst, sr = sf.read(ref_path, always_2d=True)
                    ref.append(ref_inst)
                ref = np.stack(ref, axis=0)  # [4src, 2ch, n_samples]

                grp_names = []
                grp_names.append(f"{size*idx + rank:08d}")

                # initialize datasets
                all_grp_names = sum(comm.allgather(grp_names), [])
                for grp_name in all_grp_names:
                    g = f.create_group(grp_name)
                    g.create_dataset("wav", [n_mic, duration], "float32")
                    g.create_dataset("ref", [4, n_mic, duration], "float32")

                    f[f"{grp_name}/wav"][:] = wav.T
                    f[f"{grp_name}/ref"][:] = np.swapaxes(ref, -1, -2)


def main():
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--hdf5_output_path", type=Path, required=True)
    parser.add_argument("--wav_output_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, choices=["train", "dev", "test"], required=True)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--shift", type=int, default=None)
    parser.add_argument("--mono", action="store_true")

    args, unk_args = parser.parse_known_args()
    make_dataset(args, unk_args)


if __name__ == "__main__":
    main()
