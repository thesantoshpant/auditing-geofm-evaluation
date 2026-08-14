#!/usr/bin/env python
"""Reproduce the trainable-parameter counts reported by the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "results" / "ftw_param_counts.json"


def prithvi_backbone_parameters() -> int:
    """Count the pinned TerraTorch 1.2.7 ``prithvi_eo_v2_300`` encoder.

    The preset uses a 6-channel 1x16x16 Conv3d patch projection, width 1024,
    24 standard timm transformer blocks, MLP ratio 4, and qkv bias. Position
    embeddings are buffers, so they are not trainable parameters.
    """

    width = 1024
    depth = 24
    input_channels = 6
    patch_volume = 1 * 16 * 16
    patch_projection = width * input_channels * patch_volume + width
    class_token = width
    # Per block: qkv/projection + two 4x MLP linears + their biases and two norms.
    transformer_blocks = depth * (12 * width**2 + 13 * width)
    final_norm = 2 * width
    return patch_projection + class_token + transformer_blocks + final_norm


def conv_block(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(True),
        nn.Conv2d(output_channels, output_channels, 3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(True),
    )


class Decoder(nn.Module):
    """Architecture used by ``ftw_finetune_fm.py``."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        channels = [input_channels, 256, 128, 64, 32]
        layers: list[nn.Module] = []
        for index in range(4):
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(channels[index], channels[index + 1], 3, padding=1),
                    nn.BatchNorm2d(channels[index + 1]),
                    nn.ReLU(True),
                ]
            )
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(32, 1, 1)


class UNet(nn.Module):
    """Architecture used by ``ftw_unet_baseline.py``."""

    def __init__(self) -> None:
        super().__init__()
        self.e1 = conv_block(12, 64)
        self.e2 = conv_block(64, 128)
        self.e3 = conv_block(128, 256)
        self.e4 = conv_block(256, 512)
        self.b = conv_block(512, 1024)
        self.u4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.d4 = conv_block(1024, 512)
        self.u3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.d3 = conv_block(512, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d2 = conv_block(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d1 = conv_block(128, 64)
        self.output = nn.Conv2d(64, 1, 1)


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_report() -> dict:
    prithvi_decoder = trainable_parameters(Decoder(1024))
    terramind_decoder = trainable_parameters(Decoder(768))
    unet = trainable_parameters(UNet())
    prithvi_backbone = prithvi_backbone_parameters()
    prithvi_full = prithvi_backbone + prithvi_decoder
    prithvi_full_rounded = round(prithvi_full, -5)
    return {
        "schema_version": 1,
        "generated_by": "scripts/report_param_counts.py",
        "counts": {
            "prithvi_frozen_decoder": {
                "trainable_parameters": prithvi_decoder,
                "count_kind": "exact_from_reconstructed_training_architecture",
            },
            "terramind_frozen_decoder": {
                "trainable_parameters": terramind_decoder,
                "count_kind": "exact_from_reconstructed_training_architecture",
            },
            "unet": {
                "trainable_parameters": unet,
                "count_kind": "exact_from_reconstructed_training_architecture",
            },
            "prithvi_full_finetune": {
                "trainable_parameters": prithvi_full,
                "trainable_parameters_rounded": prithvi_full_rounded,
                "backbone_parameters": prithvi_backbone,
                "count_kind": "exact_from_pinned_architecture_definition",
                "architecture_source": (
                    "terratorch 1.2.7 prithvi_eo_v2_300: width=1024, depth=24, "
                    "heads=16, mlp_ratio=4, qkv_bias=true, in_chans=6, "
                    "patch_size=[1,16,16]"
                ),
            },
        },
        "prithvi_full_to_decoder_ratio": {
            "value": round(prithvi_full / prithvi_decoder, 1),
            "interpretation": "approximately 110 times",
            "uses_rounded_full_finetune_count": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(build_report(), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    try:
        label = args.out.resolve().relative_to(ROOT)
    except ValueError:
        label = args.out
    print(f"Wrote {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
