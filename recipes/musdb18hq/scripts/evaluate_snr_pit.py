#! /usr/bin/env python3

# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
import json
from multiprocessing import Pool
from pathlib import Path
import pickle as pic
import warnings

import numpy as np

import torch

import soundfile as sf

from mixit_mss.mixit.loss.snr import snr as calc_snr

STEMS = ["bass", "drums", "vocals", "other"]

parser = ArgumentParser(description="SDR evaluation")
parser.add_argument("--method_name", type=str, required=True)
parser.add_argument("--est_dir", type=Path, required=True)
parser.add_argument("--ref_dir", type=Path, required=True)
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()

warnings.simplefilter(action="ignore", category=FutureWarning)


def evaluate(est_dir):
    # basename = os.path.basename(est_fname)
    est_fnames = sorted(list(est_dir.iterdir()))

    wav_est = []
    for est_fname in est_fnames:
        wav_tmp = sf.read(est_fname, always_2d=True)[0].T  # (n_chan, n_samples)
        wav_est.append(wav_tmp)
    wav_est = np.stack(wav_est, axis=0)  # (n_src, n_chan, n_samples)

    song_name = est_dir.parent.name
    seg_id = est_fnames[0].parent.name
    basename = f"{song_name}_{seg_id}"
    seg_id = seg_id.split(".")[0]

    ref_dir = args.ref_dir / song_name / seg_id
    wav_src = []
    inst_list = []
    for inst_name in STEMS:
        inst = sf.read(ref_dir / f"{inst_name}.wav", always_2d=True)[0].T  # (n_chan, n_samples)
        if abs(inst).sum() > 0.0:
            wav_src.append(inst)
            inst_list.append(inst_name)
    wav_src = np.stack(wav_src, axis=0)  # (n_src, n_chan, n_samples)
    n_ref, n_chan, T = wav_src.shape
    wav_est = np.pad(
        wav_est,
        (
            (0, 0),
            (0, 0),
            (0, T - wav_est.shape[-1]),
        ),
        mode="constant",
    )

    wav_est = np.swapaxes(wav_est, 0, 1)
    wav_src = np.swapaxes(wav_src, 0, 1)
    try:
        sdr, perms = calc_snr(
            torch.from_numpy(wav_est[[0]]),  # use only the 1st channel
            torch.from_numpy(wav_src[[0]]),
            solve_perm=True,
            return_perm=True,
            return_mean=False,
            negative=False,
        )
        sdr = sdr[0].numpy()  # remove channel and perm dim
        perms = perms[0]
        assert perms.shape[-1] == wav_src.shape[1], (wav_src.shape, perms.shape)

        perms_inst = {}
        for p, i in zip(perms, inst_list):
            perms_inst[i] = int(p)
    except ValueError as err:
        print(f"Value Error: {err}")

        nans = np.full(wav_est.shape[0], np.nan)
        perms_inst = {}

        return basename, (nans.copy(), n_ref, perms_inst)
    else:
        return basename, (sdr, n_ref, perms_inst)


est_dir_list = [file for dir_path in Path(args.est_dir).iterdir() if dir_path.is_dir() for file in dir_path.iterdir()]

counter = dict(vocals=[], bass=[], drums=[], other=[])


# using all cpus sometimes causes slow evaluation likely because permutation computation is heavy
# reducing processes, e.g., processes=2 works well
with Pool(processes=2) as p:
    scores = {}
    for idx, (key, values) in enumerate(p.imap_unordered(evaluate, est_dir_list)):
        utt_name = f"{key}"
        if not np.isnan(values[0]).any():
            score, n_ref, perms_inst = values
            if n_ref not in scores:
                scores[n_ref] = {}
            scores[n_ref][key] = score
            for inst, perm in perms_inst.items():
                counter[inst].append(perm)

            if args.verbose:
                print(f"{utt_name:>42s} ({idx:04d}) | " + ", ".join([f"{sdr:+06.2f}" for sdr in score]))
        else:
            pass

inst_head = {}
for inst, lst in counter.items():
    most_common = max(set(lst), key=lst.count)
    inst_head[inst] = most_common

n_data, avg_sdr = 0, 0.0
for n_ref, scores_per_nref in scores.items():
    avg_sdr_nref = np.mean([sisdr for sisdr in scores_per_nref.values()])
    print(f"average sdr {n_ref} sources ({len(scores_per_nref.values())} data): {avg_sdr_nref}")
    avg_sdr += np.sum([np.mean(sisdr) for sisdr in scores_per_nref.values()])
    n_data += len(scores_per_nref.values())
avg_sdr /= n_data
print(f"overall sdr ({n_data} data): {avg_sdr}")

with open(f"{args.est_dir}/sdr.pkl", "wb") as f:
    pic.dump(scores, f)

with open(f"{args.est_dir}/inst_head.json", "w") as f:
    json.dump(inst_head, f, indent=2)
