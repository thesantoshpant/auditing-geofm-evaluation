"""Regenerate the two aggregate manuscript figures from released results."""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
RESULTS = ROOT / "data" / "results"
REGIONS = ["india", "cambodia", "vietnam", "kenya", "france", "netherlands"]

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def transfer_gap() -> None:
    inreg = json.loads((RESULTS / "ftw_inregion_equivalence.json").read_text())
    xfer = json.loads((RESULTS / "ftw_cross_region_transfer.json").read_text())
    labels = [
        "Prithvi\nfrozen",
        "TerraMind\nfrozen",
        "U-Net",
        "Prithvi\nfull-FT",
        "TerraMind\nfull-FT",
    ]
    in_region = np.array(
        [
            np.mean([inreg["models"]["prithvi"]["regions"][r]["frozen_mean"] for r in REGIONS]),
            np.mean([inreg["models"]["terramind"]["regions"][r]["frozen_mean"] for r in REGIONS]),
            np.mean(
                [
                    np.mean(
                        [
                            json.loads((RESULTS / f"ftw_unet_{r}_seed{s}.json").read_text())[r][
                                "unet_true_auroc"
                            ]
                            for s in range(10)
                        ]
                    )
                    for r in REGIONS
                ]
            ),
            np.mean([inreg["models"]["prithvi"]["regions"][r]["fullft_mean"] for r in REGIONS]),
            np.mean([inreg["models"]["terramind"]["regions"][r]["fullft_mean"] for r in REGIONS]),
        ]
    )
    all30 = xfer["subsets"]["all30"]["means"]
    cross_region = np.array(
        [
            all30["prithvi_frozen"],
            all30["terramind_frozen"],
            all30["unet"],
            all30["prithvi_fullft"],
            all30["terramind_fullft"],
        ]
    )
    colors = ["#2ca98c", "#5a8fcb", "#173850", "#ee6a3c", "#9d4edd"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    for i, color in enumerate(colors):
        ax.plot([x[i], x[i]], [cross_region[i], in_region[i]], color=color, lw=2)
        ax.scatter(
            x[i],
            in_region[i],
            s=115,
            marker="o",
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.scatter(
            x[i],
            cross_region[i],
            s=115,
            marker="s",
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        ax.text(x[i] + 0.07, in_region[i] + 0.006, f"{in_region[i]:.3f}", va="bottom")
        ax.text(x[i] + 0.07, cross_region[i] - 0.006, f"{cross_region[i]:.3f}", va="top")
        ax.text(
            x[i] + 0.13,
            (in_region[i] + cross_region[i]) / 2,
            rf"$\Delta$={in_region[i] - cross_region[i]:.3f}",
            color=color,
            weight="bold",
            va="center",
        )

    ax.set_title(r"Transfer gap: in-region ($\bullet$) vs all-30 cross-region ($\blacksquare$)")
    ax.set_ylabel("true-label AUROC")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.74, 0.98)
    ax.set_xlim(-0.55, len(labels) - 0.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        OUT / "cross_region_transfer_gap.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def cost_performance() -> None:
    inreg = json.loads((RESULTS / "ftw_inregion_equivalence.json").read_text())
    frozen_probe_mean = np.mean(
        [
            json.loads(
                (RESULTS / f"ftw_controlled_label_comparison_{region}.json").read_text()
            )["models"]["prithvi_eo_2_0_300m"]["true"]["auroc"]
            for region in REGIONS
        ]
    )
    unet_region_means = []
    for region in REGIONS:
        vals = [
            json.loads((RESULTS / f"ftw_unet_{region}_seed{seed}.json").read_text())[region][
                "unet_true_auroc"
            ]
            for seed in range(10)
        ]
        unet_region_means.append(np.mean(vals))
    labels = [
        "frozen Prithvi + linear probe",
        "frozen Prithvi + decoder",
        "U-Net (from scratch)",
        "full fine-tune Prithvi",
    ]
    counts = json.loads((RESULTS / "ftw_param_counts.json").read_text())["counts"]
    params = np.array(
        [
            1.0e3,
            counts["prithvi_frozen_decoder"]["trainable_parameters"],
            counts["unet"]["trainable_parameters"],
            counts["prithvi_full_finetune"]["trainable_parameters_rounded"],
        ]
    )
    mean_auroc = np.array(
        [
            frozen_probe_mean,
            np.mean([inreg["models"]["prithvi"]["regions"][r]["frozen_mean"] for r in REGIONS]),
            np.mean(unet_region_means),
            np.mean([inreg["models"]["prithvi"]["regions"][r]["fullft_mean"] for r in REGIONS]),
        ]
    )
    colors = ["#2ca98c", "#31566f", "#173850", "#ee6a3c"]
    markers = ["D", "s", "^", "v"]

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for label, x, y, color, marker in zip(labels, params, mean_auroc, colors, markers):
        ax.scatter(
            x,
            y,
            s=120,
            marker=marker,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            label=label,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlim(2e2, 1e9)
    ax.set_ylim(0.88, 0.97)
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("six-region mean true-label AUROC")
    ax.set_title("Mean AUROC vs. trainable parameters")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(
        OUT / "cost_performance_tradeoff.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


if __name__ == "__main__":
    transfer_gap()
    cost_performance()
