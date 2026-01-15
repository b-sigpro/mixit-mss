# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

import torch
from torch import nn

from aiaccel.torch.lightning import OptimizerConfig
from mixit_mss.mixit.tasks.sup_task import SupTask


class MixITTask(SupTask):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        n_fft: int,
        hop_length: int,
        optimizer_config: OptimizerConfig,
        # general args
        normalize_mixture: bool = True,
        pretrained_model_path: str | None = None,
        css_validation: bool = False,
        # MixIT args
        mom_snr_range: list[float] | None = None,
    ):
        super().__init__(
            model=model,
            loss=loss,
            n_fft=n_fft,
            hop_length=hop_length,
            optimizer_config=optimizer_config,
            # general args
            normalize_mixture=normalize_mixture,
            pretrained_model_path=pretrained_model_path,
            css_validation=css_validation,
        )

        self.mom_snr_range = mom_snr_range if mom_snr_range is not None else [-5.0, 5.0]
        assert len(self.mom_snr_range) == 2, self.mom_snr_range

    @torch.autocast("cuda", enabled=False)
    def training_step(self, wav: torch.Tensor, batch_idx: int):
        wav, ref = self._make_moms(wav)
        return self._step(wav, ref=ref, log_prefix="training")

    @torch.autocast("cuda", enabled=False)
    def validation_step(self, wav: torch.Tensor | list, batch_idx: int):
        if isinstance(wav, list):
            # sometimes we may use labeled dataset to monitor the separation performance
            wav, ref = wav
        else:
            # sometimes we simply monitor MixIT loss
            wav, ref = self._make_moms(wav)
        return self._step(wav, ref=ref, log_prefix="validation")

    def _make_moms(self, wav: torch.Tensor):
        n_batch = wav.shape[0]
        assert n_batch % 2 == 0, "Batch size must be even before making MoMs"
        ref = torch.stack((wav[: n_batch // 2], wav[n_batch // 2 :]), dim=1)  # (n_batch, 2, n_chan, n_samples)

        power_0 = (ref[:, 0] ** 2).mean(dim=(-2, -1), keepdim=True)  # shape: (n_batch, 1, 1)
        power_1 = (ref[:, 1] ** 2).mean(dim=(-2, -1), keepdim=True)  # shape: (n_batch, 1, 1)
        snr_db = torch.empty((n_batch // 2, 1, 1), device=ref.device).uniform_(*self.mom_snr_range)
        ref[:, 0] = ref[:, 0] * torch.sqrt((power_1 / power_0) * (10 ** (snr_db / 10)))

        wav = ref.sum(dim=1)  # (n_batch, n_chan, n_samples)

        return wav, ref
