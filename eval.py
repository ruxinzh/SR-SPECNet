import os
import argparse
import time
import torch
import torch.nn as nn
import piq
import matplotlib.pyplot as plt
from tqdm import tqdm

from scr.helpers import create_dataloader
from models.deepIAA import SRSpec  # or: from models.deepIAA import deepIAA as SRSpec
from data_view import *

# --------------------------- Utils ---------------------------

def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def nmse(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalized MSE: mean((t - y)^2) / mean(t^2). Returns a scalar tensor."""
    target = target.float()
    output = output.float()
    mse = torch.mean((target - output) ** 2)
    denom = torch.mean(target ** 2).clamp_min(1e-12)
    return mse / denom


def _strip_module_prefix(state_dict: dict) -> dict:
    """Support checkpoints saved under DataParallel (keys like 'module.xxx')."""
    return { (k[7:] if k.startswith("module." ) else k): v for k, v in state_dict.items() }


def load_model(args, device: torch.device) -> nn.Module:
    """Instantiate SRSpec, load weights on CPU, move to device, then optionally wrap in DP."""
    model = SRSpec(args.number_elements, 20, args.output_size)

    # Print parameter count before wrapping
    print("Total trainable parameters:", count_parameters(model))

    # Load on CPU (safe), strip 'module.' if present
    state = torch.load(args.checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=False)

    model = model.to(device)

    # Wrap AFTER moving to device; let PyTorch pick device_ids=[0,1,...]
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    return model


def plot_side_by_side(RA: torch.Tensor, RA_gt: torch.Tensor, ssim_val: float, psnr_val: float, nmse_val: float,
                      save_path: str | None = None, show: bool = False, cmap: str = "viridis") -> None:
    """Plot RA and RA_gt as heatmaps with metrics in the title; optionally save to path."""
    RA_np = RA.detach().cpu().numpy()
    RAgt_np = RA_gt.detach().cpu().numpy()
    vmin = min(RA_np.min(), RAgt_np.min())
    vmax = max(RA_np.max(), RAgt_np.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(RA_np, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("RA (Prediction)")
    axes[0].set_xlabel("Angle Bin")
    axes[0].set_ylabel("Range Bin")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(RAgt_np, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("RA Ground Truth")
    axes[1].set_xlabel("Angle Bin")
    axes[1].set_ylabel("Range Bin")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"SSIM={ssim_val:.4f}  |  PSNR={psnr_val:.2f} dB  |  NMSE={nmse_val:.6f}")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
        
def plot_triplet(RA_fft: torch.Tensor, RA: torch.Tensor, RA_gt: torch.Tensor,
                 ssim_val: float, psnr_val: float, nmse_val: float,
                 save_path: str | None = None, show: bool = False, cmap: str = "viridis") -> None:
    """Plot RA_FFT (baseline), RA (prediction), and RA_gt (ground truth) side-by-side with shared color scale."""
    A = RA_fft.detach().cpu().numpy()
    B = RA.detach().cpu().numpy()
    C = RA_gt.detach().cpu().numpy()
    vmin = min(A.min(), B.min(), C.min())
    vmax = max(A.max(), B.max(), C.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    im0 = axes[0].imshow(A, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("FFT")
    axes[0].set_xlabel("Angle / Bin")
    axes[0].set_ylabel("Range / Bin")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(B, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("Angle / Bin")
    axes[1].set_ylabel("Range / Bin")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(C, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title("Ground Truth")
    axes[2].set_xlabel("Angle / Bin")
    axes[2].set_ylabel("Range / Bin")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"SSIM={ssim_val:.4f}  |  PSNR={psnr_val:.2f} dB  |  NMSE={nmse_val:.6f}")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

# --------------------------- Validation ---------------------------

@torch.no_grad()
def validate_model(model: nn.Module, dataloader, device: torch.device,
                   number_elements: int, save_dir: str | None = None, show: bool = False) -> tuple[float, float, float]:
    """Perform validation and return average (SSIM, NMSE, PSNR).

    If `save_dir` is provided, saves a side-by-side plot per batch (or per item if batch>1).
    """
    model.eval()
    total_ssim, total_nmse, total_psnr = 0.0, 0.0, 0.0
    total_time = 0.0
    n_batches = 0

    for idx, (signals, labels, alphas) in enumerate(tqdm(dataloader)):
        # If signals have a trailing singleton dim (e.g., [..., 1]), remove it
        if signals.dim() >= 5 and signals.size(-1) == 1:
            signals = signals.squeeze(-1)
        RA_FFT = compute_ra_from_signals(signals, alphas, args.number_elements, 256)
        
        signals = signals.to(device)
        labels = labels.to(device)
        alphas = alphas.to(device)
        
        B, S, N, C = signals.shape
        signals_flat = signals.view(B * S, N, C)  # [B*S, N, 2]

        start = time.time()
        outputs = model(signals_flat)  # [B*S, 256] (with your config)
        total_time += (time.time() - start)

        # Match original behavior: scale by alphas, reshape to (200,64,256), average over Doppler
        # Handle alphas shape like your earlier code
        if alphas.size(0) == 1:
            alphas_eff = alphas.squeeze(0)
        else:
            alphas_eff = alphas

        outputs = (outputs * alphas_eff).abs()
        outputs = outputs.reshape(200, 64, 256)
        RA = outputs.mean(dim=1)  # [200, 256]
        RA = RA / RA.max().clamp_min(1e-12)

        labels_eff = (labels.squeeze(0) * alphas_eff).abs()
        labels_eff = labels_eff.reshape(200, 64, 256)
        RA_gt = labels_eff.mean(dim=1)
        RA_gt = RA_gt / RA_gt.max().clamp_min(1e-12)

        # Metrics expect [B, C, H, W]
        RA_i = RA.unsqueeze(0).unsqueeze(0)
        RA_gti = RA_gt.unsqueeze(0).unsqueeze(0)

        nmse_val = nmse(RA, RA_gt).item()
        ssim_val = piq.ssim(RA_i, RA_gti).item()
        psnr_val = piq.psnr(RA_i, RA_gti).item()

        total_nmse += nmse_val
        total_ssim += ssim_val
        total_psnr += psnr_val
        n_batches += 1

        # Save plot per batch
        if save_dir is not None:
            fname = os.path.join(save_dir, f"ra_pair_{idx:05d}.png")
            plot_triplet(RA_FFT, RA, RA_gt, ssim_val, psnr_val, nmse_val, save_path=fname, show=show)

    # Averages
    avg_ssim = total_ssim / max(1, n_batches)
    avg_nmse = total_nmse / max(1, n_batches)
    avg_psnr = total_psnr / max(1, n_batches)

    print(f"Inference time (avg per batch): {total_time / max(1, n_batches):.7f} seconds")
    return avg_ssim, avg_nmse, avg_psnr


# --------------------------- Main ---------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    test_root = os.path.join(args.data_path, str(args.number_elements), "Sample")
    test_loader = create_dataloader(test_root, batch_size=args.batch_size, transform=False)

    test_ssim, test_nmse, test_psnr = validate_model(
        model,
        test_loader,
        device,
        args.number_elements,
        save_dir=args.save_dir,
        show=args.show,
    )

    print("\nTest metrics:")
    print(f"  NMSE: {test_nmse:.6f}")
    print(f"  SSIM: {test_ssim:.6f}")
    print(f"  PSNR: {test_psnr:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a DOA estimation model and visualize RA/RA_gt")
    parser.add_argument("--data_path", type=str, default="./data/BAMA/", help="Path to testing data directory")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/SR_SPEC/best_model_checkpoint_snr.pth",
                        help="Path of the model checkpoint (.pth)")
    parser.add_argument("--number_elements", type=int, default=10, help="Number of array elements in the model")
    parser.add_argument("--output_size", type=int, default=256, help="Output size of the model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--save_dir", type=str, default="outputs/ra_pairs", help="Directory to save RA/RA_gt images (if set)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively as well as saving")

    args = parser.parse_args()
    main(args)
