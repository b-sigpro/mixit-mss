# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

import torch
from torch import nn

from torchaudio.transforms import Spectrogram

from aiaccel.torch.lightning import OptimizerConfig, OptimizerLightningModule


class SupTask(OptimizerLightningModule):
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
    ):
        super().__init__(optimizer_config)

        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.load_pretrained_weight()

        self.loss = loss

        self.css_validation = css_validation

        self.stft = nn.Sequential(
            Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None),
        )

        self.normalize_mixture = normalize_mixture

    @torch.autocast("cuda", enabled=False)
    def _step(self, wav: torch.Tensor, ref: torch.Tensor | None, log_prefix: str):
        # normalize the mixture

        if self.normalize_mixture:
            wav, ref, _ = self.normalize_wav(wav, ref)

        # separation, est: (n_batch, n_src, n_chan, n_samples)
        est = self.model.css(wav, ref=ref) if log_prefix != "training" and self.css_validation else self.model(wav)

        # loss calculation
        loss = self.loss(est.transpose(1, 2), ref.transpose(1, 2)).mean()

        # logging
        log_dict = {"step": float(self.trainer.current_epoch), f"{log_prefix}/loss": loss}
        self.log_dict(
            log_dict,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            batch_size=wav.shape[0],
            sync_dist=True,
        )

        return loss

    @torch.autocast("cuda", enabled=False)
    def training_step(self, wav: torch.Tensor | list, batch_idx: int):
        if isinstance(wav, list):
            wav, ref = wav
        else:
            ref = None
        return self._step(wav, ref=ref, log_prefix="training")

    @torch.autocast("cuda", enabled=False)
    def validation_step(self, wav: torch.Tensor | list, batch_idx: int):
        if isinstance(wav, list):
            wav, ref = wav
        else:
            ref = None
        return self._step(wav, ref=ref, log_prefix="validation")

    def normalize_wav(self, wav: torch.Tensor, ref: torch.Tensor | None = None):
        wav_stft = self.stft(wav)
        scale = wav_stft.abs().square().clip(1e-15).mean(dim=(-1, -2), keepdims=True).sqrt()
        wav = wav / scale.squeeze(-1)
        if ref is not None:
            ref = ref / scale.unsqueeze(1).squeeze(-1)

        return wav, ref, scale

    def load_pretrained_weight(self):
        """
        Although the weights specified by pretrained_model_path is loaded even in evalation,
        it's overwritten by checkpoint_path specified in inference, so there should be no problem
        """
        if self.pretrained_model_path is not None:
            if torch.cuda.is_available():
                state_dict = torch.load(self.pretrained_model_path, weights_only=False)
            else:
                state_dict = torch.load(
                    self.pretrained_model_path,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
            try:
                state_dict = state_dict["state_dict"]
            except KeyError:
                print("No key named state_dict. Directly loading from model.")

            print(f"Load model from {self.pretrained_model_path}")
            for module in ["model"]:
                sd = {k: v for k, v in state_dict.items() if k.startswith(module)}
                sd = {".".join(k.split(".")[1:]): v for k, v in sd.items()}
                sd = {k: v for k, v in sd.items() if k != "n_averaged"}
                sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
                getattr(self, f"{module}").load_state_dict(sd, strict=True)

            # for b, module in enumerate(
            #     self.model.model.band_split_module.bandwise_decoding_module.modules()
            # ):
            #     if hasattr(module, "reset_parameters"):
            #         module.reset_parameters()

    def load_average_checkpoint(self, checkpoint_dir, **kwargs):
        """
        Load the model from the averaged checkpoint files.
        If the checkpoint_dir is a directory, load the averaged model from the directory.
        If the checkpoint_dir is a .ckpt file, load the model from that file.
        """

        from pathlib import Path

        from mixit_mss.utils.average_model_params import average_model_params

        if checkpoint_dir.suffix in [".ckpt", ".pth", ".pt"]:
            checkpoint_paths = [checkpoint_dir]
        else:
            assert checkpoint_dir.is_dir(), f"{checkpoint_dir} is not a directory."
            checkpoint_paths = [
                path
                for path in Path(checkpoint_dir).iterdir()
                if path.suffix in [".ckpt", ".pth", ".pt"] and path.name != "last.ckpt"
            ]
            assert all([checkpoint_paths[0].suffix == c.suffix for c in checkpoint_paths])
        state_dict = average_model_params(checkpoint_paths)
        self.load_state_dict(state_dict, strict=True)
