import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from sklearn.model_selection import KFold

from scr.loss import SNRLoss
from scr.helpers import SignalDataset, custom_transform
from models.deepIAA import SRSpec


def _maybe_squeeze_last_dim(x: torch.Tensor) -> torch.Tensor:
    """If input has a trailing singleton dim (e.g., [..., 1]), remove it."""
    if x.dim() >= 5 and x.size(-1) == 1:
        return x.squeeze(-1)
    return x


@torch.no_grad()
def validate_model(model, dataloader, criterion, device, use_snr: bool) -> float:
    """Validation loop returning average loss."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    for signals, labels, alphas in dataloader:
        signals = _maybe_squeeze_last_dim(signals.to(device))
        labels = labels.to(device)
        alphas = alphas.to(device)

        B, S, N, C = signals.shape
        signals = signals.view(B * S, N, C)
        outputs = model(signals).view(B, S, -1)

        loss = criterion(outputs, labels, alphas) if use_snr else criterion(outputs, labels)
        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    split_dir = os.path.join(args.data_path, str(args.number_elements), "train")
    files = sorted(
        os.path.join(split_dir, f)
        for f in os.listdir(split_dir)
        if os.path.isfile(os.path.join(split_dir, f))
    )
    dataset = SignalDataset(files, transform=custom_transform if args.augmentation else None)

    # K-Fold setup
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    results = {}

    # Ensure checkpoint root exists once
    os.makedirs(args.checkpoint_path, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"========== Fold {fold + 1}/{args.k_folds} ==========")

        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx)
        )
        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(val_idx)
        )

        # Model
        model = SRSpec(args.number_elements, 20, args.output_size).to(device)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)

        # Loss & Optim
        use_snr = bool(args.loss)
        criterion = SNRLoss() if use_snr else nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        # Checkpoint path (keep your original structure compactly)
        ckpt_dir_parts = [args.checkpoint_path, str(args.number_elements)]
        if args.augmentation:
            ckpt_dir_parts.append("aug")
        if use_snr:
            ckpt_dir_parts.append("snr")
        save_dir = os.path.join(*ckpt_dir_parts)
        os.makedirs(save_dir, exist_ok=True)
        best_ckpt = os.path.join(save_dir, f"fold_{fold + 1}_best_model_checkpoint.pth")
        final_ckpt = os.path.join(save_dir, f"fold_{fold + 1}_final_model.pth")

        # Train
        best_val = float("inf")
        for epoch in range(args.epochs):
            model.train()
            running, n_batches = 0.0, 0

            for signals, labels, alphas in tqdm(
                train_loader, desc=f"Fold {fold + 1}, Epoch {epoch + 1} [Training]"
            ):
                signals = _maybe_squeeze_last_dim(signals.to(device))
                labels = labels.to(device)
                alphas = alphas.to(device)

                B, S, N, C = signals.shape
                signals = signals.view(B * S, N, C)
                outputs = model(signals).view(B, S, -1)

                loss = criterion(outputs, labels, alphas) if use_snr else criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running += float(loss.item())
                n_batches += 1

            avg_train = running / max(1, n_batches)
            print(f"[Fold {fold + 1}] Epoch {epoch + 1}: Train Loss = {avg_train:.4f}")

            if (epoch + 1) % args.val_interval == 0:
                val_loss = validate_model(model, val_loader, criterion, device, use_snr)
                print(f"[Fold {fold + 1}] Epoch {epoch + 1}: Val Loss = {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    # Save underlying module state if wrapped by DP
                    to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    torch.save(to_save, best_ckpt)
                    print(f"  -> Saved new best (val {best_val:.4f}) to {best_ckpt}")

        # Save final model
        to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(to_save, final_ckpt)
        print(f"[Fold {fold + 1}] Final model saved at {final_ckpt}")

        results[fold] = best_val

    # Summary
    print("========== Cross-validation Results ==========")
    avg = sum(results.values()) / args.k_folds
    for f, loss in results.items():
        print(f"Fold {f + 1}: Best Val Loss = {loss:.4f}")
    print(f"Average Val Loss: {avg:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DOA estimation model with k-fold cross-validation")
    parser.add_argument("--data_path", type=str, default="./data/BAMA/", help="Path to the data directory")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/", help="Path to save model checkpoints")
    parser.add_argument("--number_elements", type=int, default=10, help="Number of array elements in the model")
    parser.add_argument("--output_size", type=int, default=256, help="Output size of the model")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and validation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for k-fold cross-validation")
    parser.add_argument("--augmentation", type=bool, default=True, help="Use data augmentation")
    parser.add_argument("--loss", type=bool, default=True, help="Use SNR-guided Loss")
    parser.add_argument("--val_interval", type=int, default=5, help="Validate every N epochs")
    args = parser.parse_args()
    main(args)
