import math

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.config import DataConfig
from src.models.utils.pca_basis import patchify, unpatchify


class AsymFlowModel(nn.Module):
    """Asymmetric Flow Matching wrapper (Chen et al., 2026).

    Convention:
        - t in (0, 1]: t = 0 is clean data x_0, t = 1 is pure noise eps.
        - Interpolation: x_t = (1 - t) * x_0 + t * eps.
        - Velocity: u = eps - x_0.
        - Asymmetric target: u_A = P @ eps - x_0, where P = A @ A.T is a shared
          patch-wise rank-r orthogonal projector.
        - Full-rank velocity recovery:
            u_hat = P @ u_A_hat + (I - P) @ ((x_t + u_A_hat) / max(t, t_min)).
        - Sampling integrates t: 1 -> 0.
    """

    def __init__(self, backbone: nn.Module, config, basis_A: torch.Tensor):
        super().__init__()
        self.config = config
        self.backbone = backbone

        asym_cfg = config.model.asymflow
        self.patch_size = int(asym_cfg.patch_size)
        self.rank = int(asym_cfg.rank)
        self.in_channels = int(getattr(backbone, "in_channels", None) or getattr(backbone, "img_channels"))

        if hasattr(backbone, "patch_size") and backbone.patch_size != self.patch_size:
            raise ValueError(
                f"asymflow.patch_size ({self.patch_size}) must match "
                f"backbone.patch_size ({backbone.patch_size})."
            )

        self.patch_dim = self.in_channels * self.patch_size * self.patch_size
        if tuple(basis_A.shape) != (self.patch_dim, self.rank):
            raise ValueError(
                f"basis_A shape mismatch: expected {(self.patch_dim, self.rank)}, "
                f"got {tuple(basis_A.shape)}."
            )

        # Stored as buffer so .to(device), EMA deepcopy, and checkpoints stay self-contained.
        self.register_buffer("basis_A", basis_A.detach().clone().float())

        # Log-Normal timestep sampling (t=0 data, t=1 noise -> mean=+0.8 matches JiT's -0.8)
        ts_cfg = config.model.get("time_step_log_normal", {})
        self.p_mean = float(ts_cfg.get("mean", 0.8))
        self.p_std = float(ts_cfg.get("std", 0.8))

        self.time_scale = float(asym_cfg.get("time_scale", 1.0))
        self.t_eps = float(asym_cfg.get("t_eps", 1e-4))
        self.train_t_min = float(asym_cfg.get("train_t_min", 0.05))
        self.sample_t_min = float(asym_cfg.get("sample_t_min", 0.04))

        if asym_cfg.get("init_jit_bottleneck", True) and self.rank > 0:
            self._init_jit_bottleneck()

    def _init_jit_bottleneck(self):
        """Initialize the first `rank` channels of JiT's bottleneck conv with basis_A^T.

        Follows LakonLab's AsymJiT._init_projected_bottleneck so the input bottleneck
        has immediate access to the low-rank coordinates A^T x_t while staying trainable.
        """
        x_embedder = getattr(self.backbone, "x_embedder", None)
        proj1 = getattr(x_embedder, "proj1", None)
        if proj1 is None or not isinstance(proj1, nn.Conv2d):
            return

        if proj1.out_channels < self.rank:
            raise ValueError(
                f"JiT bottleneck_dim ({proj1.out_channels}) must be >= asymflow.rank ({self.rank})."
            )

        p = self.patch_size
        # Patch order is (c, p, p), matching Conv2d weight layout (out_c, in_c, p, p) directly.
        proj1_basis = self.basis_A.T.reshape(self.rank, self.in_channels, p, p)
        init_gain = math.sqrt(2.0 * self.patch_dim / (self.patch_dim + proj1.out_channels))
        proj1_basis = proj1_basis.mul(init_gain).to(dtype=proj1.weight.dtype, device=proj1.weight.device)
        proj1.weight.data[: self.rank].copy_(proj1_basis)

    def _project_low_rank(self, x: torch.Tensor) -> torch.Tensor:
        """Apply patch-wise orthogonal projector P = A @ A.T in float32."""
        if self.rank == 0:
            return torch.zeros_like(x, dtype=torch.float32)
        if self.rank == self.patch_dim:
            return x.float()

        _, c, h, w = x.shape
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            a_f = self.basis_A.float()
            patches = patchify(x_f, self.patch_size)
            proj_patches = (patches @ a_f) @ a_f.T
            return unpatchify(proj_patches, self.patch_size, c, h, w)

    def _orthogonal_split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose x into (P x, (I - P) x) in float32."""
        x_sub = self._project_low_rank(x)
        x_orth = x.float() - x_sub
        return x_sub, x_orth

    def _recover_velocity(
        self,
        u_a: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_min: float,
    ) -> torch.Tensor:
        """Recover full-rank velocity u_hat from asymmetric prediction u_A_hat (Eq. 5)."""
        with torch.amp.autocast(device_type=x_t.device.type, enabled=False):
            u_a_sub, u_a_orth = self._orthogonal_split(u_a)
            _, x_t_orth = self._orthogonal_split(x_t)
            t_clamped = t.float().view(-1, 1, 1, 1).clamp(min=t_min)
            u_full = u_a_sub + (x_t_orth + u_a_orth) / t_clamped
            return u_full.to(u_a.dtype)

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample continuous noise level t in [t_eps, 1.0] via Log-Normal sigmoid."""
        z = torch.randn(batch_size, device=device) * self.p_std + self.p_mean
        return torch.sigmoid(z).clamp(min=self.t_eps, max=1.0)

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        device = x.device

        t = self.sample_t(b, device=device)
        t_view = t.view(b, 1, 1, 1)
        eps = torch.randn_like(x)

        x_t = (1.0 - t_view) * x + t_view * eps
        u_true = eps - x

        u_a_pred = self.backbone(x_t, t * self.time_scale)

        # Unbiased orthogonal loss decomposition in float32:
        #   ||P(u_A_hat - u)||^2 + ||(I - P)(u_A_hat + x_0) / max(t, train_t_min)||^2
        with torch.amp.autocast(device_type=device.type, enabled=False):
            u_a_sub, u_a_orth = self._orthogonal_split(u_a_pred)
            u_sub, _ = self._orthogonal_split(u_true)
            _, x_orth = self._orthogonal_split(x)

            t_clamped = t_view.float().clamp(min=self.train_t_min)
            loss_low_rank = (u_a_sub - u_sub).square().mean()
            loss_orth = ((u_a_orth + x_orth) / t_clamped).square().mean()
            loss = loss_low_rank + loss_orth

        return {
            "loss": loss,
            "loss_low_rank": loss_low_rank.detach(),
            "loss_orth": loss_orth.detach(),
        }

    @torch.no_grad()
    def _get_velocity(self, x_t: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        u_a_pred = self.backbone(x_t, t_tensor * self.time_scale)
        return self._recover_velocity(u_a_pred, x_t, t_tensor, t_min=self.sample_t_min)

    @torch.no_grad()
    def sample(self, num_images: int, image_size: int, get_stats, device="cuda", **kwargs) -> torch.Tensor:
        num_steps = int(self.config.sampling.solver_num_steps)
        method = str(self.config.sampling.get("method", "heun")).lower()
        if method not in ("heun", "euler"):
            raise ValueError(f"Unsupported sampling method '{method}'. Expected 'heun' or 'euler'.")

        x_t = torch.randn(num_images, self.in_channels, image_size, image_size, device=device)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in tqdm(range(num_steps), desc=f"AsymFlow {method.capitalize()} ({num_steps} steps)"):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr  # Negative: integrating 1 -> 0

            t_tensor = torch.full((num_images,), t_curr, device=device)

            # Use Euler on the final step so we never evaluate at t = 0.
            if method == "heun" and i < num_steps - 1:
                u_curr = self._get_velocity(x_t, t_tensor)
                x_euler = x_t + dt * u_curr
                t_next_tensor = torch.full((num_images,), t_next, device=device)
                u_next = self._get_velocity(x_euler, t_next_tensor)
                u_pred = 0.5 * (u_curr + u_next)
                x_t = x_t + dt * u_pred
            else:
                u_pred = self._get_velocity(x_t, t_tensor)
                x_t = x_t + dt * u_pred

        mean, std = get_stats(DataConfig(**self.config.dataset))
        mean = torch.tensor(mean, device=x_t.device, dtype=x_t.dtype).view(1, -1, 1, 1)
        std = torch.tensor(std, device=x_t.device, dtype=x_t.dtype).view(1, -1, 1, 1)
        x = (x_t * std + mean).clamp(0, 1)
        return x
