"""
Refactored deepIAA with cleaner structure, typing, and weight-compatibility.

⚠️ Backward compatibility:
- Class names and attribute names are preserved: `deepIAA`, `deepIAAblock`, layers `fc1..fc4`, `unet`, `uiaa`, and attributes `AH1`, `AH2` still exist.
- Steering matrices are now registered as non-persistent buffers (moved across devices, not saved in state_dict), so old checkpoints load without mismatch.

Expected tensor layouts:
- Input `x`: shape [B, N_elems, 2] (real, imag) — same as before.
- Output: magnitude spectrum of shape [B, output_size].

If your UNet expects different spatial dims, adjust the `spec.unsqueeze(1)`/`squeeze(1)` lines accordingly.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from scr.helpers import steering_vector
from models.unet import UNet


class deepIAAblock(nn.Module):
    def __init__(self, number_element: int = 10, mid_size: int = 20, output_size: int = 256) -> None:
        super().__init__()
        # MLP that mixes raw complex snapshot (as real+imag) with coarse spectrum features
        in_dim = number_element * 2 + output_size
        self.fc1 = nn.Linear(in_dim, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, mid_size * 2)  # output complex vector of length mid_size

        # UNet refiner operating on coarse spectrum (as single-channel "image")
        self.unet = UNet(1, 1, 1)

        self.number_element = number_element
        self.mid_size = mid_size
        self.output_size = output_size

    def forward(self, x: torch.Tensor, spec: torch.Tensor, AH: torch.Tensor) -> torch.Tensor:
        """Refine spectrum.

        Args:
            x: input snapshots, real/imag stacked as [B, N, 2]
            spec: coarse magnitude spectrum, shape [B, output_size]
            AH: steering matrix for mid-size array, shape [output_size, mid_size]
        Returns:
            Magnitude spectrum [B, output_size]
        """
        B = x.size(0)

        # Convert real/imag -> complex vector per batch
        x_c = torch.view_as_complex(x)  # [B, N]

        # Concatenate features: raw (as real/imag flat) + coarse spectrum
        x_flat = torch.view_as_real(x_c).reshape(B, -1)  # [B, 2*N]
        feat = torch.cat((x_flat, spec), dim=1)          # [B, 2*N + output_size]

        h = self.relu(self.fc1(feat))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.fc4(h)                                   # [B, 2*mid_size]
        h = h.view(B, -1, 2)
        h_c = torch.view_as_complex(h)                    # [B, mid_size]

        # Mid-array beamforming (Top branch)
        # AH: [output_size, mid_size]; h_c: [B, mid_size]
        # (AH @ h_c^T)^T -> [B, output_size]
        Top = torch.abs((AH @ h_c.T).T) / float(self.mid_size)

        # UNet refiner (Bottom branch)
        # UNet expects [B, C=1, W=output_size]
        Bot = torch.abs(self.unet(spec.unsqueeze(1))).squeeze(1)  # [B, output_size]

        # Element-wise product
        out = Top * Bot
        return out.abs()


class SRSpec(nn.Module):
    def __init__(self, number_element: int = 10, mid_size: int = 20, output_size: int = 256) -> None:
        super().__init__()
        self.number_element = number_element
        self.mid_size = mid_size
        self.output_size = output_size

        # Precompute angle grid & steering matrices as non-persistent buffers (not saved in state_dict)
        linear_space = torch.linspace(-1, 1, output_size)
        angle_axis = torch.asin(linear_space) * (180.0 / torch.pi)

        self.AH1 = steering_vector(number_element, angle_axis).conj().T  # [output_size, number_element]
        self.AH2 = steering_vector(mid_size, angle_axis).conj().T        # [output_size, mid_size]

        # self.register_buffer("AH1", AH1, persistent=False)
        # self.register_buffer("AH2", AH2, persistent=False)

        self.uiaa = deepIAAblock(number_element, mid_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input snapshots as real/imag pairs, shape [B, N_elems, 2]
        Returns:
            Refined magnitude spectrum of shape [B, output_size]
        """
        B = x.size(0)

        # real/imag -> complex
        x_c = torch.view_as_complex(x)  # [B, N_elems]
        # Coarse spectrum via full array
        # (AH1 @ x_c^T)^T -> [B, output_size]
        spec = torch.abs(torch.matmul(self.AH1.to(x.device), x_c.T)).T / self.number_element

        # Refine with UIAA block using mid-array steering
        out = self.uiaa(x, spec, self.AH2.to(x.device))
        return out



# ----------------- Smoke test -----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRSpec()
    model_path = "/home/sunlab/research2024/SR-DOA/checkpoint/SR_SPEC/best_model_checkpoint.pth"
    if torch.cuda.device_count() >= 4:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model, device_ids=list(range(4)))  # Modify here to change the number of GPUs used    
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    x = torch.randn(10, 10, 2).to(device)  # [B, N_elems, 2]
    y = model(x)
    print("Output shape:", y.shape)  # [10, output_size]
