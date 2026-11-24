from src.models.backbones.unet import UNet
from src.models.backbones.unet_attention import UNetWithAttention
from src.models.backbones.jit import JiT
from src.models.backbones.dit import DiT
from src.models.backbones.edm_precond import EDMPrecond

from src.models.ddpm import DiffusionModel
from src.models.autoencoder_kl import AutoEncoderKL
from src.models.edm import EDMModel
from src.models.rectified_flow_jit import JiTFlowModel
from src.models.asymflow import AsymFlowModel
from src.models.utils.pca_basis import get_or_compute_pca_basis
from src.data.config import DataConfig

import torch.nn as nn


# Backbone builders & registry
def _create_jit_backbone(config):
    return JiT(**config.model.backbone)


def _create_dit_backbone(config):
    return DiT(**config.model.backbone)


def _create_unet_backbone(config):
    return UNet(**config.model.backbone)


def _create_unet_attention_backbone(config):
    return UNetWithAttention(**config.model.backbone, image_size=config.dataset.image_size)


BACKBONE_REGISTRY = {
    "jit": _create_jit_backbone,
    "dit": _create_dit_backbone,
    "unet": _create_unet_backbone,
    "unet_attention": _create_unet_attention_backbone,
}


def get_backbone(config) -> nn.Module:
    backbone_type = config.model.get("backbone_type")
    if backbone_type not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Backbone '{backbone_type}' not found. Available backbones: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[backbone_type](config)


# Model builder functions

def create_ddpm_unet_base(config):
    backbone = UNet(**config.model.backbone)
    model = DiffusionModel(backbone=backbone, config=config)
    return model

def create_ddpm_unet_attention(config):
    backbone = UNetWithAttention(**config.model.backbone, image_size=config.dataset.image_size)
    model = DiffusionModel(backbone=backbone, config=config)
    return model

def create_autoencoder_kl(config):
    model = AutoEncoderKL(config=config)
    return model

def create_ddpm_jit(config):
    backbone = JiT(**config.model.backbone)
    model = DiffusionModel(backbone=backbone, config=config)
    return model

def create_edm_song_unet(config):
    # The EDMPrecond wrapper will handle the preconditioning and just call the SongUNet inside it
    backbone = EDMPrecond(**config.model.backbone)
    model = EDMModel(backbone=backbone, config=config)
    return model

def create_flow_jit(config):
    backbone = JiT(**config.model.backbone)
    model = JiTFlowModel(backbone=backbone, config=config)
    return model


def create_asymflow(config):
    backbone = get_backbone(config)
    asym_cfg = config.model.asymflow
    data_config = DataConfig(**config.dataset)
    in_channels = int(getattr(backbone, "in_channels", None) or getattr(backbone, "img_channels"))
    basis_A = get_or_compute_pca_basis(
        data_config=data_config,
        in_channels=in_channels,
        patch_size=int(asym_cfg.patch_size),
        rank=int(asym_cfg.rank),
        basis_type=str(asym_cfg.get("basis_type", "pca")),
        cache_dir=str(asym_cfg.get("cache_dir", "data/pca_bases")),
        max_images=asym_cfg.get("pca_max_images", None),
        seed=int(config.get("seed", 42)),
        device=config.get("device", "cpu"),
    )
    return AsymFlowModel(backbone=backbone, config=config, basis_A=basis_A)


# Model Registry
MODEL_REGISTRY = {
    "ddpm_unet_base": create_ddpm_unet_base,
    "ddpm_unet_attention": create_ddpm_unet_attention,
    "autoencoder_kl": create_autoencoder_kl,
    "ddpm_jit": create_ddpm_jit,
    "edm_song_unet": create_edm_song_unet,
    "flow_jit": create_flow_jit,
    "asymflow": create_asymflow,
}

def get_model(config) -> nn.Module:
    name = config.model.get("name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    builder_fn = MODEL_REGISTRY[name]
    return builder_fn(config)