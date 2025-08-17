#!/usr/bin/env python3
"""
Visualize Range–Azimuth (RA) heatmaps side-by-side for FFT vs Ground Truth.

- Loads samples from the dataset using `create_dataloader` (batch_size=1).
- Computes spectrum via steering matrix and forms RA heatmaps by averaging across Doppler.
- Plots two heatmaps per sample with y-axis flipped (origin="lower").

Usage:
    python visualize_ra.py \
        --data_path ./data/BAMA \
        --number_elements 10 \
        --output_size 256 \
        --limit 5 \
        --save_dir outputs/ra_viz

Notes:
- Assumes the following helper functions exist in your project:
    - `create_dataloader(root, batch_size=1, transform=False)` from `scr.helpers`
    - `steering_vector(n_elements, angles_deg)` from `scr.helpers`
- Expects the dataloader to yield (signals, labels, alphas) where
    * signals: real-imag stacked tensor shaped (..., 2)
    * labels: same layout as signals
    * alphas: per-sample scaling factors
- Adjust indexing/reshaping if your layout differs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm

from scr.helpers import create_dataloader, steering_vector


def compute_ra_from_signals(
    signals: torch.Tensor,
    alphas: torch.Tensor,
    n_elements: int,
    out_bins: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RA heatmaps (FFT vs Ground Truth) given one sample.

    Args:
        signals: real-imag stacked complex signal tensor, expected shape (..., 2)
        alphas: scaling factors compatible with signals (broadcastable)
        n_elements: number of array elements for steering vector
        out_bins: number of angle bins

    Returns:
        RA, RA_gt: tensors with shape [200, out_bins]
    """
    # Squeeze potential leading singleton batch dims
    signals = signals.squeeze()
    alphas = alphas.squeeze(0)

    # Convert real/imag pair to complex
    signals_complex = torch.view_as_complex(signals)

    # Build angle grid and steering matrix
    linear_space = torch.linspace(-1, 1, out_bins, device=signals_complex.device)
    angle_axis = torch.asin(linear_space) * (180.0 / torch.pi)
    AH = steering_vector(n_elements, angle_axis).conj().T  # [out_bins, n_elements]^H → [out_bins, n_elements]

    # Spectrum per range–doppler line, then scale by alphas
    # spec: [N_frames, out_bins]
    spec = torch.abs(torch.matmul(AH, signals_complex.T)).T

    outputs = spec * alphas  # broadcast scaling

    # Reshape to [range, doppler, angle]; adjust 200,64 if different in your data
    outputs = outputs.reshape(200, 64, out_bins)

    # Average across Doppler to form RA
    RA = outputs.mean(dim=1)
    RA = RA / (RA.max() + 1e-12)

    # Ground-truth path mirrors above using labels scaling (labels provided separately by caller)
    return RA


def compute_ra_from_labels(labels: torch.Tensor, alphas: torch.Tensor, out_bins: int = 256) -> torch.Tensor:
    """Compute RA heatmap from labels (ground truth)."""
    labels = labels.squeeze(0)
    labels = (labels.abs() * alphas).reshape(200, 64, out_bins)
    RA_gt = labels.mean(dim=1)
    RA_gt = RA_gt / (RA_gt.max() + 1e-12)
    return RA_gt


def plot_side_by_side(
    RA: torch.Tensor,
    RA_gt: torch.Tensor,
    title_left: str = "FFT",
    title_right: str = "Ground Truth",
    share_scale: bool = True,
    cmap: str = "viridis",
    figsize: tuple[int, int] = (12, 5),
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot two heatmaps with flipped y-axis.

    If `share_scale` is True, use common vmin/vmax so colors are directly comparable.
    """
    RA_np = RA.detach().cpu().numpy()
    RAgt_np = RA_gt.detach().cpu().numpy()

    vmin = vmax = None
    if share_scale:
        vmin = min(RA_np.min(), RAgt_np.min())
        vmax = max(RA_np.max(), RAgt_np.max())

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    im0 = axes[0].imshow(RA_np, aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title(title_left)
    axes[0].set_xlabel("Range Bin")
    axes[0].set_ylabel("Azimuth Bin")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(RAgt_np, aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(title_right)
    axes[1].set_xlabel("Range Bin")
    axes[1].set_ylabel("Azimuth Bin")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    # Build data loader
    root = os.path.join(args.data_path, str(args.number_elements), "Sample")
    loader = create_dataloader(root, batch_size=1, transform=False)

    # Angle bins from CLI
    out_bins = args.output_size

    pbar = tqdm(loader, desc="Visualizing", total=(args.limit or None))

    count = 0
    for signals, labels, alphas in pbar:
        signals = signals
        labels = labels
        alphas = alphas

        RA = compute_ra_from_signals(signals, alphas, args.number_elements, out_bins)
        RA_gt = compute_ra_from_labels(labels, alphas, out_bins)

        # Compose output filename if saving
        save_path = None
        if args.save_dir:
            save_dir = Path(args.save_dir)
            save_path = save_dir / f"ra_sample_{count:05d}.png"

        plot_side_by_side(
            RA,
            RA_gt,
            title_left="FFT",
            title_right="Ground Truth",
            share_scale=not args.no_share_scale,
            cmap=args.cmap,
            figsize=(args.fig_w, args.fig_h),
            save_path=save_path,
            show=not args.no_show,
        )

        count += 1
        if args.limit is not None and count >= args.limit:
            break


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize RA heatmaps for FFT vs Ground Truth")
    p.add_argument("--data_path", type=str, default="./data/BAMA", help="Root data directory")
    p.add_argument("--number_elements", type=int, default=10, help="Array element count")
    p.add_argument("--output_size", type=int, default=256, help="Angle bins (columns)")
    p.add_argument("--limit", type=int, default=None, help="Max number of samples to visualize")
    p.add_argument("--save_dir", type=str, default=None, help="If set, save PNGs to this folder")
    p.add_argument("--no-show", action="store_true", help="Do not display windows (useful for servers)")
    p.add_argument("--no-share-scale", action="store_true", help="Use separate color scales for each plot")
    p.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name")
    p.add_argument("--fig-w", type=int, default=12, help="Figure width in inches")
    p.add_argument("--fig-h", type=int, default=5, help="Figure height in inches")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
