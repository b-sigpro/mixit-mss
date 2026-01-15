# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

import numpy as np

import torch
import torch.nn as nn

from einops.layers.torch import Rearrange
from torchaudio.transforms import InverseSpectrogram, Spectrogram


class ModelWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        n_fft: int,
        hop_length: int,
        fs: int,
        scaling: bool = False,
        css_segment_size: int = 6,
        css_shift_size: int = 6,
        freeze_separator: bool = False,
    ):
        super().__init__()

        self.model = model

        self.stft = nn.Sequential(
            Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None),
            Rearrange("b m f t -> b f m t"),
        )
        self.istft = InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)

        self.hop_length = hop_length
        self.fs = fs
        self.scaling = scaling

        self.css_segment_size = css_segment_size
        self.css_shift_size = css_shift_size

        if freeze_separator:
            self._freeze_all_except_encdec()

    def forward(self, wav, **kwargs):
        if self.scaling:
            wav_stft = self.stft(wav)
            scale = wav_stft.abs().square().clip(1e-15).mean(dim=(1, -1), keepdims=True).sqrt()
            wav = wav / scale.squeeze(1)

        x = self.stft(wav)[..., : wav.shape[-1] // self.hop_length]
        s = self.model(x, **kwargs)

        est = self.istft(s, wav.shape[-1])  # [B, N, M, T]
        if self.scaling:
            est = est * scale

        return est

    def css(
        self,
        speech_mix: torch.Tensor,
        ref: torch.Tensor = None,
        **kwargs,
    ):
        """
        CSS-style separation for long recording.

        """
        speech_length = speech_mix.shape[-1]
        if speech_length > self.css_segment_size * self.fs:
            # Segment-wise speech enhancement/separation
            overlap_length = int(np.round(self.fs * (self.css_segment_size - self.css_shift_size)))
            num_segments = int(np.ceil((speech_length - overlap_length) / (self.css_shift_size * self.fs)))
            t = T = int(self.css_segment_size * self.fs)
            pad_shape = speech_mix[..., :T].shape

            enh_waves = []
            ref_segments = []

            for i in range(num_segments):
                st = int(i * self.css_shift_size * self.fs)
                en = st + T
                if en >= speech_length:
                    # en - st < T (last segment)
                    en = speech_length
                    speech_seg = speech_mix.new_zeros(pad_shape)
                    t = en - st
                    speech_seg[..., :t] = speech_mix[..., st:en].clone()

                    if ref is not None:
                        ref_seg = ref.new_zeros(ref[..., :T].shape)
                        ref_seg[..., :t] = ref[..., st:en].clone()
                        ref_segments.append(ref_seg)
                else:
                    t = T
                    speech_seg = speech_mix[..., st:en].clone()  # B x T [x C]

                    if ref is not None:
                        ref_seg = ref[..., st:en].clone()
                        ref_segments.append(ref_seg)

                if abs(speech_seg).sum() == 0.0:
                    assert i > 0, "BUG"
                    enh_waves.append(torch.zeros_like(enh_waves[-1]))
                else:
                    processed_wav = self(speech_seg, **kwargs)

                    # List[torch.Tensor(num_spk, B, T)]
                    processed_wav = processed_wav[..., :T]
                    enh_waves.append(processed_wav)

            assert len(enh_waves) == num_segments, (len(enh_waves), num_segments)

            # c. Stitch the enhanced segments together
            waves = enh_waves[0]

            for i in range(1, num_segments):
                if i == num_segments - 1:
                    enh_waves[i][..., t:] = 0
                    enh_waves_res_i = enh_waves[i][..., overlap_length:t]
                else:
                    enh_waves_res_i = enh_waves[i][..., overlap_length:]

                # overlap-and-add (average over the overlapped part)
                if overlap_length > 0:
                    assert waves[..., -overlap_length:].shape == enh_waves[i][..., :overlap_length].shape

                    waves[..., -overlap_length:] = (
                        waves[..., -overlap_length:] + enh_waves[i][..., :overlap_length]
                    ) / 2
                # concatenate the residual parts of the later segment
                waves = torch.cat([waves, enh_waves_res_i], dim=-1)
            # ensure the stitched length is same as input
            assert waves.size(-1) == speech_mix.size(-1), (
                waves.shape,
                speech_mix.shape,
            )

        else:
            # normal forward enhance for short audio
            waves = self(speech_mix, **kwargs)

        return waves

    def _freeze_all_except_encdec(self):
        """
        Freeze parameters except for thee encoder and decoder.
        """

        # freeze all
        for param in self.model.parameters():
            param.requires_grad = False

        # unfreeze some modules
        for module in self.model.band_split_module.bandwise_decoding_module:
            for param in module.parameters():
                param.requires_grad = True

        for module in self.model.band_split_module.band_split_module:
            for param in module.parameters():
                param.requires_grad = True
