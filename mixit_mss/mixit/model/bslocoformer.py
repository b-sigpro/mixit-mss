# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
# Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
# SPDX-License-Identifier: Apache-2.0 license

from typing import Union

from itertools import accumulate
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from packaging.version import parse as V

from mixit_mss.mixit.model.band_split_config import BAND_SPLIT

is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")


class BSLocoformer(nn.Module):
    """BS-Locoformer presented in [1]. Following [2], we do not use any positional encoding.

    Reference:
    [1] Saijo, Kohei, et al., "Task-Aware Unified Source Separation," ICASSP 20255.
    [2] Saijo, Kohei, et al., "A Comparative Study on Positional Encoding for Time-frequency Domain Dual-path
        Transformer-based Source Separation Models," EUSIPCO, 2025


    Args:
        n_srcs: number of output sources/speakers.
        n_imics: number of microphones channels (only fixed-array geometry supported).
        n_layers: number of Locoformer blocks.
        emb_dim: int
            Number of hidden dimension in the encoding Conv2D.
        norm_type: str
            Normalization layer. Must be either of "layernorm" or "rmsgroupnorm".
        num_groups: int
            Number of groups in RMSGroupNorm layer.
        tf_order: str
            Order of frequency and temporal modeling. Must be either of "ft" or "tf".
        n_heads: int
            Number of heads in multi-head self-attention.
        flash_attention: bool
            Whether to use flash attention. Only compatible with half precision.
        ffn_type: str or list
          elif model_name == "prompt_bslocoformer":
        model = PromptBSLocoformer(**model_conf)  FFN type chosen from "conv1d" or "swiglu_conv1d".
            Giving the list (e.g., ["conv1d", "conv1d"]) makes the model Macaron-style.
        ffn_hidden_dim: int or list
            Number of hidden dimension in FFN.
            Giving the list (e.g., [256, 256]) makes the model Macaron-style.
        conv1d_kernel:
            Kernel size in Conv1d.
        conv1d_shift:
            Shift size of Conv1d kernel.
        dropout: float
            Dropout probability.
        eps: float
            Small constant for nomalization layer.
        estimation: str
            How to estimate sources. Must be either of "mapping", "real_imag_masking".
    """

    def __init__(
        self,
        num_spk: int = 2,
        num_chan: int = 1,
        n_layers: int = 6,
        # general setup
        emb_dim: int = 128,
        norm_type: str = "rmsgrouporm",
        num_groups: int = 4,  # used only in RMSGroupNorm
        tf_order: str = "ft",
        # self-attention related
        n_heads: int = 4,
        flash_attention: bool = False,  # available when using mixed precision
        attention_dim: int = 128,
        # ffn related
        ffn_type: Union[str, list] = "swiglu_conv1d",
        ffn_hidden_dim: Union[int, list] = 384,
        conv1d_kernel: int = 4,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        # others
        eps: float = 1.0e-5,
    ):
        super().__init__()
        assert is_torch_2_0_plus, "Support only pytorch >= 2.0.0"
        self._num_spk = num_spk
        self._num_chan = num_chan
        self.n_layers = n_layers
        assert attention_dim % n_heads == 0, (attention_dim, n_heads)

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                TFLocoformerBlock(
                    # general setup
                    emb_dim=emb_dim,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    tf_order=tf_order,
                    # self-attention related
                    n_heads=n_heads,
                    flash_attention=flash_attention,
                    attention_dim=attention_dim,
                    # ffn related
                    ffn_type=ffn_type,
                    ffn_hidden_dim=ffn_hidden_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                    dropout=dropout,
                    eps=eps,
                )
            )

        self.band_split_module = BandSplitModule(num_spk, emb_dim, num_chan)

    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward.

        Args:
            input (torch.Tensor): multi-channel audio tensor with
                    M audio channels in TF-domain [B, F, M, T]

        Returns:
            batch (torch.Tensor): separated sources with a shape of
                [B, N, M, F, T]
        """
        assert input.ndim == 4, "Input must have 4 dims."

        # input: (B, T, M, F)
        batch0 = input.movedim(-1, 1)  # [B, F, M, T] -> [B, T, F, M]
        batch = torch.cat((batch0.real, batch0.imag), dim=-1)  # [B, T, F, 2*M]

        # normal spectrogram -> band-splitted tensor
        batch = self.band_split_module.band_split(batch)  # [B, -1, T, F]

        # separation
        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        # band-splitted tensor -> normal spectrogram
        batch = self.band_split_module.bandwise_decoding(batch)  # [B, n_srcs*2, T, F]
        batch = batch.to(torch.float32)

        # complex masking
        batch = torch.complex(batch[:, 0], batch[:, 1])
        batch = batch0.movedim(-1, -3).unsqueeze(1) * batch  # [B, N, M, T, F]
        return batch.transpose(-1, -2)

    @property
    def num_spk(self):
        return self._num_spk


class BandSplitModule(nn.Module):
    def __init__(self, num_src: int, emb_dim: int, num_chan: int):
        super().__init__()

        self.num_src = num_src
        self.num_chan = num_chan
        self.bands = BAND_SPLIT

        self.band_split_module = nn.ModuleList([])
        for band in self.bands:
            self.band_split_module.append(
                nn.Sequential(
                    nn.GroupNorm(1, band * 2 * num_chan),
                    nn.Conv1d(band * 2 * num_chan, emb_dim, kernel_size=1),
                )
            )

        self.bandwise_decoding_module = nn.ModuleList([])
        for band in self.bands:
            self.bandwise_decoding_module.append(
                nn.Sequential(
                    nn.GroupNorm(1, emb_dim),
                    nn.Conv1d(emb_dim, emb_dim * 4, kernel_size=1),
                    nn.Conv1d(emb_dim * 4, emb_dim * 4, kernel_size=1),
                    nn.Tanh(),
                    nn.Conv1d(
                        emb_dim * 4,
                        band * num_src * num_chan * 2 * 2,
                        kernel_size=1,
                    ),
                    nn.GLU(dim=1),
                )
            )

        self.band_idx = [0] + self.bands
        self.band_idx = list(accumulate(self.band_idx))

    @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def band_split(self, input):
        """Band split process

        input: torch.Tensor (n_batch, n_frame, n_freq, 2)
        """
        n_batch, n_frame = input.shape[:2]
        input = input.movedim(1, -1)

        output = []
        for b in range(len(self.bands)):
            sub_band = input[:, self.band_idx[b] : self.band_idx[b + 1]]
            output.append(self.band_split_module[b](sub_band.reshape(n_batch, -1, n_frame)))
        output = torch.stack(output, dim=-1)
        return output  # (n_batch, emb_dim, n_frame, n_bands)

    @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def bandwise_decoding(self, input):
        """Band-wise decoding process

        input: torch.Tensor (n_batch, emb_dim, n_frame, n_bands)
        """
        n_batch, n_frame = input.shape[0], input.shape[2]
        output = []
        for b in range(len(self.bands)):
            sub_band = self.bandwise_decoding_module[b](input[..., b])
            output.append(sub_band.reshape(n_batch, 2, self.num_src, self.num_chan, -1, n_frame))
        return torch.cat(output, dim=-2).transpose(-1, -2)  # (n_batch, 2*num_src, num_chan, n_frame, n_freq)


class TFLocoformerBlock(nn.Module):
    def __init__(
        self,
        # general setup
        emb_dim=128,
        norm_type="rmsgrouporm",
        num_groups=4,
        tf_order="ft",
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        ffn_type="swiglu_conv1d",
        ffn_hidden_dim=384,
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        assert tf_order in ["tf", "ft"], tf_order
        self.tf_order = tf_order
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

        self.freq_path = LocoformerBlock(
            # general setup
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            # self-attention related
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            # ffn related
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )
        self.frame_path = LocoformerBlock(
            # general setup
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            # self-attention related
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            # ffn related
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )

    @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def forward(self, input):
        """TF-Locoformer forward.

        input: torch.Tensor
            Input tensor, (n_batch, channel, n_frame, n_freq)
        """

        return self.freq_frame_process(input) if self.tf_order == "ft" else self.frame_freq_process(input)

    def freq_frame_process(self, input):
        output = input.movedim(1, -1)  # (B, T, Q_old, H)
        output = self.freq_path(output)

        output = output.transpose(1, 2)  # (B, F, T, H)
        output = self.frame_path(output)
        return output.transpose(-1, 1)

    def frame_freq_process(self, input):
        # Input tensor, (n_batch, hidden, n_frame, n_freq)
        output = input.transpose(1, -1)  # (B, F, T, H)
        output = self.frame_path(output)

        output = output.transpose(1, 2)  # (B, T, F, H)
        output = self.freq_path(output)
        return output.movedim(-1, 1)


class LocoformerBlock(nn.Module):
    def __init__(
        self,
        # general setup
        emb_dim=128,
        norm_type="rmsgrouporm",
        num_groups=4,
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        ffn_type="swiglu_conv1d",
        ffn_hidden_dim=384,
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        FFN = {
            "conv1d": ConvDeconv1d,
            "swiglu_conv1d": SwiGLUConvDeconv1d,
        }
        Norm = {
            "layernorm": nn.LayerNorm,
            "rmsgroupnorm": RMSGroupNorm,
        }
        assert norm_type in Norm, norm_type

        self.macaron_style = len(ffn_type) == 2
        if self.macaron_style:
            assert len(ffn_hidden_dim) == 2, "Need two FFNs when using macaron style model"

        # initialize FFN
        self.ffn_norm = nn.ModuleList([])
        self.ffn = nn.ModuleList([])
        for f_type, f_dim in zip(ffn_type[::-1], ffn_hidden_dim[::-1]):
            assert f_type in FFN, f_type
            if norm_type == "rmsgroupnorm":
                self.ffn_norm.append(Norm[norm_type](num_groups, emb_dim, eps=eps))
            else:
                self.ffn_norm.append(Norm[norm_type](emb_dim, eps=eps))
            self.ffn.append(
                FFN[f_type](
                    emb_dim,
                    f_dim,
                    conv1d_kernel,
                    conv1d_shift,
                    dropout=dropout,
                )
            )

        # initialize self-attention
        if norm_type == "rmsgroupnorm":
            self.attn_norm = Norm[norm_type](num_groups, emb_dim, eps=eps)
        else:
            self.attn_norm = Norm[norm_type](emb_dim, eps=eps)
        self.attn = MultiHeadSelfAttention(
            emb_dim,
            attention_dim=attention_dim,
            n_heads=n_heads,
            dropout=dropout,
            flash_attention=flash_attention,
        )

    def forward(self, x):
        """Locoformer block Forward.

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either of the number of frames or freqs
        """
        B, T, F, C = x.shape

        if self.macaron_style:
            # FFN before self-attention
            input_ = x
            output = self.ffn_norm[-1](x)  # [B, T, F, C]
            output = self.ffn[-1](output)  # [B, T, F, C]
            output = output + input_
        else:
            output = x

        # Self-attention
        input_ = output
        output = self.attn_norm(output)
        output = output.contiguous().view([B * T, F, C])
        output = self.attn(output)
        output = output.contiguous().view([B, T, F, C]) + input_

        # FFN after self-attention
        input_ = output
        output = self.ffn_norm[0](output)  # [B, T, F, C]
        output = self.ffn[0](output)  # [B, T, F, C]
        output = output + input_

        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        flash_attention=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

        if flash_attention:
            self.flash_attention_config = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            self.flash_attention_config = dict(enable_flash=False, enable_math=True, enable_mem_efficient=True)

    def forward(self, input):
        # get query, key, and value
        query, key, value = self.get_qkv(input)

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale
        with torch.backends.cuda.sdp_kernel(**self.flash_attention_config):
            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (batch, head, seq_len, -1)

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)

    def get_qkv(self, input):
        n_batch, seq_len = input.shape[:2]
        x = self.qkv(input).reshape(n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 1)  # (batch, head, seq_len, 3, -1)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value


class ConvDeconv1d(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs):
        super().__init__()

        self.diff_ks = conv1d_kernel - conv1d_shift

        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, conv1d_kernel, stride=conv1d_shift),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """ConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either of the number of frames or freqs
        """
        b, s1, s2, h = x.shape
        x = x.view(b * s1, s2, h)
        x = x.transpose(-1, -2)
        x = self.net(x).transpose(-1, -2)
        x = x[..., self.diff_ks // 2 : self.diff_ks // 2 + s2, :]
        return x.view(b, s1, s2, h)


class SwiGLUConvDeconv1d(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs):
        super().__init__()

        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)

        self.swish = nn.SiLU()
        self.deconv1d = nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x):
        """SwiGLUConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either of the number of frames or freqs
        """
        b, s1, s2, h = x.shape
        x = x.contiguous().view(b * s1, s2, h)
        x = x.transpose(-1, -2)

        # padding
        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
            + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        # conv-deconv1d
        x = self.conv1d(x)
        gate = self.swish(x[..., self.dim_inner :, :])
        x = x[..., : self.dim_inner, :] * gate
        x = self.dropout(x)
        x = self.deconv1d(x).transpose(-1, -2)

        # cut necessary part
        x = x[..., self.diff_ks : self.diff_ks + s2, :]
        return self.dropout(x).view(b, s1, s2, h)


class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim, eps=1e-8, bias=False):
        """
        Root Mean Square Group Normalization (RMSGroupNorm).
        Unlike Group Normalization in vision filed, RMSGroupNorm
        is applied in each TF bins.

        Args:
            num_groups: int
                Number of groups
            dim: int
                number of dimensions
            eps: float
                Small constant to avoid zero division.
            bias: bool
                Whether to add bias term. RMSNorm does not use bias.

        """
        super().__init__()

        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // self.num_groups

        self.gamma = nn.Parameter(torch.Tensor(dim).to(torch.float32))
        nn.init.ones_(self.gamma)

        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.Tensor(dim).to(torch.float32))
            nn.init.zeros_(self.beta)
        self.eps = eps
        self.num_groups = num_groups

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, input):
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))

        # normalization
        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * self.dim_per_group ** (-1.0 / 2)
        output = input / (rms + self.eps)

        # reshape and affine transformation
        output = output.view(others + (-1,))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta

        return output
