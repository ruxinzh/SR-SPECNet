import os
import argparse
import torch
import torch.nn as nn
import piq
from tqdm import tqdm

from scr.helpers import create_dataloader
from models.deepIAA import SRSpec  


def nmse(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalized MSE: mean((t - y)^2) / mean(t^2)."""
    target = target.float()
    output = output.float()
    mse = torch.mean((target - output) ** 2)
    denom = torch.mean(target ** 2).clamp_min(1e-12)  # avoid /0
    return mse / denom


@torch.no_grad()
def validate_model(model: nn.Module, dataloader, device: torch.device, number_elements: int):
    """Run validation and return average SSIM, NMSE, PSNR."""
    model.eval()
    total_ssim, total_nmse, total_psnr = 0.0, 0.0, 0.0
    n_batches = 0

    for signals, labels, alphas in tqdm(dataloader):
        # shapes: signals [B, S, N, C(=2)] or [B, S, N, C, 1]; we only remove the trailing 1 if it exists
        if signals.dim() >= 5 and signals.size(-1) == 1:
            signals = signals.squeeze(-1)

        signals = signals.to(device)
        labels = labels.to(device)
        alphas = alphas.to(device)

        B, S, N, C = signals.shape
        signals_flat = signals.view(B * S, N, C)  # -> [B*S, N, 2]

        outputs = model(signals_flat)  # -> [B*S, 256] (for your config)
        # Broadcast alphas; many datasets hold alpha per (range,doppler), earlier code used squeeze(0)
        # Keep original behavior but safer:
        if alphas.size(0) == 1:
            alphas = alphas.squeeze(0)
        outputs = (outputs * alphas).abs()

        # Build RA and RA_gt
        if number_elements == 10:
            outputs = outputs.reshape(200, 64, 256)
            RA = outputs.mean(dim=1)  # [200, 256]
        else:
            RA = outputs

        RA = RA / RA.max().clamp_min(1e-12)

        labels = (labels.squeeze(0) * alphas).abs()
        if number_elements == 10:
            labels = labels.reshape(200, 64, 256)
            RA_gt = labels.mean(dim=1)
        else:
            RA_gt = labels

        RA_gt = RA_gt / RA_gt.max().clamp_min(1e-12)

        # Metrics (piq expects [B, C, H, W])
        RA_i   = RA.unsqueeze(0).unsqueeze(0)       # [1,1,200,256]
        RA_gti = RA_gt.unsqueeze(0).unsqueeze(0)    # [1,1,200,256]

        nmse_val = nmse(RA, RA_gt).item()
        ssim_val = piq.ssim(RA_i, RA_gti).item()
        psnr_val = piq.psnr(RA_i, RA_gti).item()

        total_nmse += nmse_val
        total_ssim += ssim_val
        total_psnr += psnr_val
        n_batches += 1

    avg_ssim = total_ssim / max(1, n_batches)
    avg_nmse = total_nmse / max(1, n_batches)
    avg_psnr = total_psnr / max(1, n_batches)
    return avg_ssim, avg_nmse, avg_psnr


def _strip_module_prefix(state_dict: dict) -> dict:
    """Support checkpoints saved under DataParallel (keys like 'module.xxx')."""
    return { (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items() }


def load_model(args, device: torch.device) -> nn.Module:
    """Instantiate SRSpec, load weights on CPU, move to device, then optionally wrap in DP."""
    model = SRSpec(args.number_elements, 20, args.output_size)

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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    test_root = os.path.join(args.data_path, str(args.number_elements), "test")
    test_loader = create_dataloader(test_root, batch_size=args.batch_size, transform=False)

    ssim_val, nmse_val, psnr_val = validate_model(model, test_loader, device, args.number_elements)
    print("\nResults on test set")
    print(f"  SSIM: {ssim_val:.6f}")
    print(f"  NMSE: {nmse_val:.6f}")
    print(f"  PSNR: {psnr_val:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a DOA estimation model")
    parser.add_argument("--data_path", type=str, default="./data/BAMA/", help="Path to testing data directory")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/SR_SPEC/best_model_checkpoint_snr.pth",
                        help="Path of the model checkpoint (.pth)")
    parser.add_argument("--number_elements", type=int, default=10, help="Number of array elements in the model")
    parser.add_argument("--output_size", type=int, default=256, help="Output size of the model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
    args = parser.parse_args()
    main(args)