import inspect

import torch
import torch.nn as nn
from mamba_ssm.modules.mamba2_simple import Mamba2Simple

from evigre.networks.res_encoder_unet import ResEncoderUNet


class MambaLayer2(nn.Module):
    """Mamba2 block used at the bottleneck of a U-Net style encoder-decoder."""

    def __init__(
        self,
        dim: int,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 80,
        use_mem_eff_path: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba2Simple(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            use_mem_eff_path=use_mem_eff_path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels = x.shape[:2]
        if channels != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {channels}")

        spatial_shape = x.shape[2:]
        n_tokens = x.shape[2:].numel()
        x_flat = x.reshape(batch_size, channels, n_tokens).transpose(-1, -2)
        x_flat = self.norm(x_flat)
        x_flat = self.mamba(x_flat)
        return x_flat.transpose(-1, -2).reshape(batch_size, channels, *spatial_shape)


class EvigreUMambaBot2(ResEncoderUNet):
    """EviDent-CBCT-compatible U-Mamba2 segmentation network for dental Stage 1.

    This mirrors U-Mamba2's UMambaBot2 idea: keep the U-Net encoder-decoder
    interface and inject a Mamba2 block into the deepest encoder feature.
    Keeping the local EviDent-CBCT class avoids depending on which nnUNet fork wins
    Python import resolution at runtime.
    """

    def __init__(
        self,
        *args,
        mamba_d_state: int = 32,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 80,
        mamba_use_mem_eff_path: bool = True,
        **kwargs,
    ):
        res_encoder_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in inspect.signature(ResEncoderUNet.__init__).parameters
        }
        super().__init__(*args, **res_encoder_kwargs)
        self.mamba_layer = MambaLayer2(
            dim=self.encoder.output_channels[-1],
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
            use_mem_eff_path=mamba_use_mem_eff_path,
        )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1])
        return self.decoder(skips)

EvigreUMambaBot2.allowed_init_kwargs = (
    set(inspect.signature(ResEncoderUNet.__init__).parameters.keys())
    - {"self"}
    | {
        "mamba_d_state",
        "mamba_d_conv",
        "mamba_expand",
        "mamba_headdim",
        "mamba_use_mem_eff_path",
    }
)

