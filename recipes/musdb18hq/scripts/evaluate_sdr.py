#! /usr/bin/env python3

# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from functools import partial
import json
from multiprocessing import Pool
from pathlib import Path
import pickle as pic
import warnings

import numpy as np

import torch

from museval import evaluate as calc_csdr

import soundfile as sf

from mixit_mss.mixit.loss.snr import snr as calc_usdr

STEMS = ["bass", "drums", "vocals", "other"]

parser = ArgumentParser(description="SDR evaluation")
parser.add_argument("--est_dir", type=Path, required=True)
parser.add_argument("--ref_dir", type=Path, required=True)
parser.add_argument("--metric", type=str, required=True, choices=["usdr", "csdr"])
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()

warnings.simplefilter(action="ignore", category=FutureWarning)


def evaluate(est_dir, metric):
    est_fnames = sorted(list(est_dir.iterdir()))

    wav_est = []
    for est_fname in est_fnames:
        wav_tmp = sf.read(est_fname, always_2d=True)[0].T  # (n_chan, n_samples)
        wav_est.append(wav_tmp)
    wav_est = np.stack(wav_est, axis=0)  # (n_src, n_chan, n_samples)

    basename = est_fnames[0].parent.name

    ref_dir = args.ref_dir / basename
    wav_src = []
    inst_list = []
    for inst_name in STEMS:
        # (n_chan, n_samples)
        inst = sf.read(ref_dir / f"{inst_name}.wav", always_2d=True)[0].T
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

    if metric == "usdr":
        wav_est = np.swapaxes(wav_est, 0, 1)
        wav_src = np.swapaxes(wav_src, 0, 1)

        sdr = calc_usdr(
            torch.from_numpy(wav_est),
            torch.from_numpy(wav_src),
            solve_perm=False,
            return_mean=False,
            negative=False,
        )  # (n_chan, n_src)
        sdr = sdr.mean(dim=0).numpy()  # average over channel
    else:
        wav_est = np.swapaxes(wav_est, 1, 2)  # (n_src, n_samples, n_chan)
        wav_src = np.swapaxes(wav_src, 1, 2)  # (n_src, n_samples, n_chan)

        sdr, *_ = calc_csdr(wav_src, wav_est)  # List, (n_src, n_win)
        sdr = [np.nanmedian(s) for s in sdr]  # List, (n_src, )
    return basename, sdr


est_dir_list = [file for dir_path in Path(args.est_dir).iterdir() if dir_path.is_dir() for file in dir_path.iterdir()]

scores_all = dict(vocals=[], bass=[], drums=[], other=[])
scores_per_song = dict()

print(f"Start {args.metric.upper()} evaluation")
evaluate_func = partial(evaluate, metric=args.metric)
with Pool(processes=4) as p:
    for idx, (key, values) in enumerate(p.imap_unordered(evaluate_func, est_dir_list)):
        utt_name = f"{key}"
        scores_per_song[utt_name] = dict()

        score = values
        for stem_idx, stem_name in enumerate(STEMS):
            scores_all[stem_name].append(score[stem_idx])
            scores_per_song[utt_name][stem_name] = score[stem_idx]

        if args.verbose:
            print(f"{utt_name:>42s} ({idx:04d}) | " + ", ".join([f"{sdr:+06.2f}" for sdr in score]))

avg_scores = dict(vocals=0.0, bass=0.0, drums=0.0, other=0.0, avg=0.0)
for stem_name, stem_scores in scores_all.items():
    stem_score = np.mean(stem_scores) if args.metric == "usdr" else np.median(stem_scores)
    avg_scores[stem_name] = stem_score
    avg_scores["avg"] += stem_score
    print(f"{stem_name}: {round(stem_score, 2)}")

avg_scores["avg"] /= len(scores_all)
print(f"Avg: {round(avg_scores['avg'], 2)}")

with open(f"{args.est_dir}/{args.metric}.pkl", "wb") as f:
    pic.dump(scores_per_song, f)

with open(f"{args.est_dir}/avg_{args.metric}.json", "w") as f:
    json.dump(avg_scores, f, indent=2)
