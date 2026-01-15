#! /usr/bin/env python3

# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
# Copyright (c) 2024 Kohei Saijo
#
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Apache-2.0 license

import argparse
import ast
import json
import os
from pathlib import Path

import pandas as pd

partitions = {"training": "train", "validation": "dev", "test": "test"}


def load(filepath):
    """
    Ported from https://github.com/mdeff/fma/blob/master/utils.py#L183
    """

    filename = os.path.basename(filepath)

    if "features" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "echonest" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "genres" in filename:
        return pd.read_csv(filepath, index_col=0)

    if "tracks" in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [
            ("track", "tags"),
            ("album", "tags"),
            ("artist", "tags"),
            ("track", "genres"),
            ("track", "genres_all"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [
            ("track", "date_created"),
            ("track", "date_recorded"),
            ("album", "date_created"),
            ("album", "date_released"),
            ("artist", "date_created"),
            ("artist", "active_year_begin"),
            ("artist", "active_year_end"),
        ]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ("small", "medium", "large")
        try:
            tracks["set", "subset"] = tracks["set", "subset"].astype("category", categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks["set", "subset"] = tracks["set", "subset"].astype(
                pd.CategoricalDtype(categories=SUBSETS, ordered=True)
            )

        COLUMNS = [
            ("track", "genre_top"),
            ("track", "license"),
            ("album", "type"),
            ("album", "information"),
            ("artist", "bio"),
        ]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype("category")

        return tracks


def collect_metadata(tracks: pd.DataFrame, size: str):
    ret = dict(train=[], validation=[], test=[])

    if args.size == "small":
        tracks = tracks[(tracks[("set", "subset")] == "small")]
    elif args.size == "medium":
        tracks = tracks[(tracks[("set", "subset")] == "medium") | (tracks[("set", "subset")] == "small")]

    for split in partitions:
        tracks_split = tracks[tracks[("set", "split")] == split]

        tracks_split.reset_index(inplace=True)
        track_ids = list(tracks_split["track_id"])
        track_ids = [f"{str(tid).zfill(6)}.mp3" for tid in track_ids]
        ret[partitions[split]] = track_ids

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--size", type=str, choices=["small", "medium", "large", "full"], required=True)

    args = parser.parse_args()

    tracks = load(args.csv_path)
    tracks_split = collect_metadata(tracks, args.size)

    with open(args.output_dir / f"fma_split_{args.size}.json", "w") as f:
        json.dump(tracks_split, f, indent=2)
