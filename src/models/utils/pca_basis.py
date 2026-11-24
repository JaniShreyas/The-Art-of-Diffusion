import argparse
from pathlib import Path
from typing import Optional

from einops import rearrange
import torch

from src.data import get_dataloaders
from src.data.config import DataConfig


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split (B, C, H, W) image tensor into (B, num_patches, C * p * p)."""
    _, _, h, w = x.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            f"Image size {(h, w)} is not divisible by patch_size={patch_size}."
        )
    return rearrange(
        x,
        "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
        p1=patch_size,
        p2=patch_size,
    )


def unpatchify(
    patches: torch.Tensor,
    patch_size: int,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Restore (B, num_patches, C * p * p) patches back to (B, C, H, W)."""
    return rearrange(
        patches,
        "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
        h=height // patch_size,
        w=width // patch_size,
        c=channels,
        p1=patch_size,
        p2=patch_size,
    )


@torch.no_grad()
def compute_pca_basis(
    dataloader: torch.utils.data.DataLoader,
    patch_size: int,
    max_images: Optional[int] = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute uncentered patch PCA basis via float64 Gram matrix eigendecomposition.

    Follows AsymFlow Eq. (8): takes top left singular vectors of normalized patches
    without subtracting the mean, so DC/average patch intensity stays in Im(P).

    Returns:
        U: (D, D) orthonormal basis matrix sorted by descending eigenvalue (float32, CPU).
        eigenvalues: (D,) descending eigenvalues (float32, CPU).
    """
    gram: Optional[torch.Tensor] = None
    num_seen = 0

    for batch in dataloader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        if max_images is not None:
            remaining = max_images - num_seen
            if remaining <= 0:
                break
            images = images[:remaining]

        images = images.to(device=device, dtype=torch.float32)
        patches = patchify(images, patch_size).reshape(-1, images.shape[1] * patch_size * patch_size).double()

        if gram is None:
            dim = patches.shape[1]
            gram = torch.zeros((dim, dim), dtype=torch.float64, device=device)

        gram += patches.T @ patches
        num_seen += images.shape[0]

    if gram is None:
        raise RuntimeError("Dataloader yielded no images; cannot compute PCA basis.")

    evals, evecs = torch.linalg.eigh(gram)
    evals = torch.flip(evals, dims=[0]).clamp_min(0.0).float().cpu()
    basis_full = torch.flip(evecs, dims=[1]).contiguous().float().cpu()
    return basis_full, evals


def _plain_image_size(image_size):
    """Convert OmegaConf ListConfig / sequence to plain Python int or tuple for weights_only=True."""
    if image_size is None:
        return None
    if isinstance(image_size, int):
        return int(image_size)
    return tuple(int(v) for v in image_size)


def get_basis_cache_path(
    cache_dir: str | Path,
    dataset_name: str,
    image_size: int | tuple[int, int],
    patch_size: int,
) -> Path:
    plain_size = _plain_image_size(image_size)
    size_str = f"{plain_size[0]}x{plain_size[1]}" if isinstance(plain_size, tuple) else str(plain_size)
    return Path(cache_dir) / f"{dataset_name}_{size_str}_p{patch_size}.pt"


def get_or_compute_pca_basis(
    data_config: DataConfig,
    in_channels: int,
    patch_size: int,
    rank: int,
    basis_type: str = "pca",
    cache_dir: str | Path = "data/pca_bases",
    max_images: Optional[int] = None,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load, auto-compute + cache, or randomly sample a (D, r) orthonormal patch basis."""
    patch_dim = in_channels * patch_size * patch_size
    if not (0 <= rank <= patch_dim):
        raise ValueError(f"rank must be in [0, {patch_dim}], got {rank}.")

    if rank == 0:
        return torch.empty((patch_dim, 0), dtype=torch.float32)

    basis_type = basis_type.lower()
    if basis_type == "random":
        gen = torch.Generator().manual_seed(seed)
        rand_mat = torch.randn((patch_dim, rank), generator=gen, dtype=torch.float64)
        q, _ = torch.linalg.qr(rand_mat, mode="reduced")
        return q.contiguous().float()

    if basis_type != "pca":
        raise ValueError(f"Unsupported basis_type '{basis_type}'. Expected 'pca' or 'random'.")

    cache_path = get_basis_cache_path(cache_dir, data_config.name, data_config.image_size, patch_size)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        basis_full = payload["U"]
        if basis_full.shape != (patch_dim, patch_dim):
            raise ValueError(
                f"Cached PCA basis at {cache_path} has shape {tuple(basis_full.shape)}, "
                f"expected {(patch_dim, patch_dim)}."
            )
    else:
        print(f"Computing patch PCA basis ({data_config.name}, patch_size={patch_size}) -> {cache_path}")
        train_loader, _ = get_dataloaders(data_config)
        basis_full, evals = compute_pca_basis(
            train_loader,
            patch_size=patch_size,
            max_images=max_images,
            device=device,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "U": basis_full,
                "eigenvalues": evals,
                "dataset": str(data_config.name),
                "image_size": _plain_image_size(data_config.image_size),
                "patch_size": patch_size,
                "in_channels": in_channels,
            },
            cache_path,
        )

    return basis_full[:, :rank].contiguous().float()


def main():
    parser = argparse.ArgumentParser(description="Precompute and cache a patch-wise PCA basis for AsymFlow.")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset key in DATASET_REGISTRY.")
    parser.add_argument("--image-size", type=int, default=32, help="Square image size.")
    parser.add_argument("--patch-size", type=int, default=4, help="Patch size p.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for Gram accumulation.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on training images.")
    parser.add_argument("--cache-dir", type=str, default="data/pca_bases", help="Output cache directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_cfg = DataConfig(
        name=args.dataset,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle_train=True,
    )
    train_loader, _ = get_dataloaders(data_cfg)
    sample_batch = next(iter(train_loader))
    sample_imgs = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
    in_channels = sample_imgs.shape[1]

    basis_full, evals = compute_pca_basis(
        train_loader,
        patch_size=args.patch_size,
        max_images=args.max_images,
        device=args.device,
    )
    cache_path = get_basis_cache_path(args.cache_dir, args.dataset, args.image_size, args.patch_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "U": basis_full,
            "eigenvalues": evals,
            "dataset": args.dataset,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "in_channels": in_channels,
        },
        cache_path,
    )

    cum_var = torch.cumsum(evals, dim=0) / evals.sum().clamp_min(1e-12)
    print(f"Saved full ({basis_full.shape[0]}x{basis_full.shape[1]}) PCA basis to {cache_path}")
    for r in [1, 2, 4, 8, 16]:
        if r <= len(cum_var):
            print(f"  rank {r:2d}/{len(cum_var)}: cumulative energy = {cum_var[r - 1].item():.4f}")


if __name__ == "__main__":
    main()
