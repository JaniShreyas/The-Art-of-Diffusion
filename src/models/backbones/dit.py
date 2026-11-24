import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from src.models.utils.jit_utils import (
    RMSNorm,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """Single-stage Conv2d patch embedder with optional fixed 2D sin-cos positional embedding."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
        use_pos_embed: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        grid_size = img_size // patch_size
        self.num_patches = grid_size * grid_size
        self.use_pos_embed = use_pos_embed

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

        if use_pos_embed:
            pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
            self.register_buffer(
                "pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0)
            )
        else:
            self.pos_embed = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(
                f"Input image size {(h, w)} doesn't match model {self.img_size}."
            )
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.pos_embed is not None:
            x = (x + self.pos_embed).to(x.dtype)
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            self.mlp[0].weight.dtype
        )
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds class labels (+1 slot reserved for unconditional / CFG null class)."""

    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.ffn_dropout(F.silu(x1) * x2))


class GeluMLP(nn.Module):
    """Standard DiT GELU(tanh) feed-forward block."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class DiTAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        use_rmsnorm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        fused_attn: bool = True,
        upcast_attn: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.upcast_attn = upcast_attn

        norm_cls = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_cls(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_cls(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        drop_p = self.attn_drop.p if self.training else 0.0
        if self.upcast_attn:
            with torch.amp.autocast(device_type=q.device.type, enabled=False):
                if self.fused_attn:
                    x = F.scaled_dot_product_attention(
                        q.float(), k.float(), v.float(), dropout_p=drop_p
                    ).to(v.dtype)
                else:
                    attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
                    attn = self.attn_drop(attn.softmax(dim=-1))
                    x = (attn @ v.float()).to(v.dtype)
        else:
            if self.fused_attn:
                x = F.scaled_dot_product_attention(
                    q.to(v.dtype), k.to(v.dtype), v, dropout_p=drop_p
                )
            else:
                q_cast, k_cast = q.to(v.dtype), k.to(v.dtype)
                attn = (q_cast @ k_cast.transpose(-2, -1)) * self.scale
                attn = self.attn_drop(attn.softmax(dim=-1))
                x = attn @ v

        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class DiTBlock(nn.Module):
    """Unified DiT / LightningDiT block with adaLN-Zero conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        fused_attn: bool = True,
        upcast_attn: bool = False,
    ):
        super().__init__()
        if use_rmsnorm:
            self.norm1 = RMSNorm(hidden_size, eps=1e-6)
            self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = DiTAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            fused_attn=fused_attn,
            upcast_attn=upcast_attn,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        else:
            self.mlp = GeluMLP(hidden_size, mlp_hidden_dim, drop=proj_drop)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, feat_rope=None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        if use_rmsnorm:
            self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        else:
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    """Pure-PyTorch Diffusion Transformer matching LakonLab's LightningDiT & Classic DiT.

    Modes:
        - LightningDiT (default): use_rmsnorm=True, use_swiglu=True, use_rope=True.
        - Classic DiT (Peebles & Xie / asymdit.py): set all four flags to False.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 1,
        use_rmsnorm: bool = True,
        use_swiglu: bool = True,
        use_qknorm: bool = False,
        use_rope: bool = True,
        use_pos_embed: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        fused_attention: bool = True,
        upcast_attention: bool = False,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.gradient_checkpointing = checkpointing

        self.x_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            use_pos_embed=use_pos_embed,
            bias=True,
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        if use_rope:
            head_dim = hidden_size // num_heads
            if head_dim % 4 != 0:
                raise ValueError(
                    f"2D RoPE requires head_dim (hidden_size // num_heads = {head_dim}) to be divisible by 4."
                )
            half_head_dim = head_dim // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0
            )
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qknorm=use_qknorm,
                    use_swiglu=use_swiglu,
                    use_rmsnorm=use_rmsnorm,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    fused_attn=fused_attention,
                    upcast_attn=upcast_attention,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm
        )
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            raise ValueError(f"Expected square number of patches, got {x.shape[1]}.")
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        if y is None:
            y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        c = self.t_embedder(t) + self.y_embedder(y)
        x = self.x_embedder(x)

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = checkpoint.checkpoint(
                    block, x, c, self.feat_rope, use_reentrant=False
                )
            else:
                x = block(x, c, self.feat_rope)

        x = self.final_layer(x, c)
        return self.unpatchify(x)
