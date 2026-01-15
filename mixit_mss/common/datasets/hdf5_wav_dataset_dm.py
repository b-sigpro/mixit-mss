# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import numpy as np

import torch

from aiaccel.torch.datasets import CachedDataset, HDF5Dataset


class HDF5WavDMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: list[Path] | list[str],
        duration: int | None = None,
        sr: int | None = None,
        return_ref: bool = False,
        p_source_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self._dataset = []
        for path in dataset_path:
            self._dataset.append(CachedDataset(HDF5Dataset(path)))
        self.min_num_data = min([len(d) for d in self._dataset])

        self.duration = duration
        self.sr = sr

        self.return_ref = return_ref

        self.p_source_dropout = p_source_dropout
        assert 0.0 <= self.p_source_dropout < 1.0

    def __len__(self) -> int:
        return self.min_num_data

    def __getitem__(self, index: int):
        ref = []
        for dataset in self._dataset:
            idx = np.random.randint(len(dataset))
            ref.append(dataset[idx]["wav"])

        # randomly dropout some sources
        if self.p_source_dropout > 0.0:
            drop_source = np.ones(len(ref), dtype=bool)
            # ensure to keep at least a source
            while np.all(drop_source):
                drop_source = np.random.uniform(0, 1, size=len(ref)) < self.p_source_dropout
            ref = [np.zeros_like(r) if drop else r for r, drop in zip(ref, drop_source)]

        ref = np.stack(ref, axis=0)  # (n_src, n_chan, n_samples)

        if self.duration is not None:
            duration = self.duration * self.sr
            t_start = np.random.randint(0, ref.shape[-1] - duration + 1)
            t_end = t_start + duration
            ref = ref[..., t_start:t_end]

        # RMS normalization
        coef = np.sqrt(ref**2).mean(axis=(-1, -2), keepdims=True)
        coef = coef + (coef == 0.0)  # avoid zero-devision
        ref = ref / coef

        # Randomly adjust gain for each source
        gain_db = np.random.uniform(-10, 10, (ref.shape[0], 1, 1))
        gain = 10 ** (gain_db / 20)
        ref *= gain

        wav = ref.sum(axis=0)
        if self.return_ref:
            return wav, ref
        else:
            return wav
