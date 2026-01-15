#!/usr/bin/env python3

# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
import json
from pathlib import Path

from hydra.utils import instantiate

import torch
from torch import nn

from aiaccel.config import load_config


def select_heads(model, head_indices):
    num_src = model.model.model.band_split_module.num_src
    num_chan = model.model.model.band_split_module.num_chan
    bands = model.model.model.band_split_module.bands

    target_module = model.model.model.band_split_module.bandwise_decoding_module

    for b, band_module in enumerate(target_module):
        old_conv = band_module[-2]  # Conv1d

        band = bands[b]

        # caoculate the number of channels
        out_per_src = band * num_chan * 2 * 2  # output dim per source
        new_out_channels = len(head_indices) * out_per_src

        has_bias = old_conv.bias is not None
        new_conv = nn.Conv1d(
            in_channels=old_conv.in_channels,
            out_channels=new_out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=has_bias,
        )

        # 対象インデックスを蓄積
        selected_indices = []
        for gate in range(2):  # gate 0: data, gate 1: gate
            for ri in range(2):  # 0: real, 1: imag
                for src in head_indices:
                    for ch in range(num_chan):
                        for bb in range(band):
                            idx = (
                                gate * 2 * num_src * num_chan * band  # offset for glu
                                + ri * num_src * num_chan * band  # real/imag
                                + src * num_chan * band
                                + ch * band
                                + bb
                            )
                            selected_indices.append(idx)
        assert new_out_channels == len(selected_indices)
        selected_indices = torch.tensor(selected_indices, dtype=torch.long, device="cpu")

        # replace
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight[selected_indices])
            if has_bias:
                new_conv.bias.copy_(old_conv.bias[selected_indices])
        band_module[-2] = new_conv

    target_module.num_src = len(head_indices)

    model.model.model._num_spk = len(head_indices)

    return model


def main(args):
    assert args.input_path.suffix in [".ckpt", ".pth", ".pt"], f"{args.input_path}"

    config_path = args.input_path.parent.parent / "merged_config.yaml"
    config = load_config(config_path)

    model = instantiate(config.task)
    model.load_average_checkpoint(args.input_path)
    model.eval()

    with open(args.head_json) as f:
        head_inst = json.load(f)
    head_indices = []
    for inst in ["bass", "drums", "vocals", "other"]:
        head_indices.append(head_inst[inst])

    model = select_heads(model, head_indices)
    torch.save(model.state_dict(), args.output_path)

    # with open(args.input_path / "config_4heads.pkl", "wb") as f:
    #     pkl.dump(config, f)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--head_json", type=Path, required=True)

    args, unk_args = parser.parse_known_args()

    main(args)
