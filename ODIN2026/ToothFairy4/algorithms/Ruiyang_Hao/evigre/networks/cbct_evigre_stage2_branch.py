"""Released EviDent-CBCT Stage 2 branch networks.

The internal class names are retained so released checkpoints can reconstruct
their architecture from the embedded nnU-Net plan.
"""

from __future__ import annotations

import inspect
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

from evigre.networks.res_encoder_unet import ResidualEncoder
from evigre.networks.stage2_components import GatedFusion, LabelSpecificHead, MLPHead, SmallCropEncoder
from evigre.networks.umamba2_unet import MambaLayer2


def _require(value: str, expected: str, field: str) -> None:
    if str(value).lower() != expected:
        raise ValueError(f"The released model requires {field}={expected!r}, got {value!r}")


class VisualTokenProjector(nn.Module):
    """Project Stage 1 pooled visual tokens into Stage 2 feature space."""

    def __init__(self, in_dim: int = 1408, out_dim: int = 192, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(out_dim)),
            nn.LayerNorm(int(out_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(dtype=self.net[0].weight.dtype))


class CBCTEVIGREStage2ToothInstanceNet(nn.Module):
    """Final tooth branch: gated crop/token fusion and conditioned missing-tooth head."""

    def __init__(
        self,
        crop_in_channels: int = 6,
        crop_base_channels: int = 12,
        raw_feature_dim: int = 192,
        visual_token_input_dim: int = 1408,
        visual_feature_dim: int = 192,
        fused_feature_dim: int = 256,
        visual_fusion_mode: str = "hybrid_gated",
        num_tooth_labels: int = 11,
        tooth_region_scheme: str = "ten_region",
        num_tooth_regions: int = 10,
        tooth_region_embed_dim: int = 16,
        head_hidden_dim: int = 384,
        dropout: float = 0.1,
        tooth_head_type: str = "label_specific",
        tooth_projection_dim: int = 256,
        tooth_label_projection_dim: int = 128,
        tooth_missing_mode: str = "conditioned",
        missing_context_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        del tooth_region_scheme, tooth_projection_dim, tooth_label_projection_dim, kwargs
        _require(visual_fusion_mode, "hybrid_gated", "visual_fusion_mode")
        _require(tooth_head_type, "label_specific", "tooth_head_type")
        _require(tooth_missing_mode, "conditioned", "tooth_missing_mode")

        self.num_tooth_labels = int(num_tooth_labels)
        self.num_tooth_regions = int(num_tooth_regions)
        self.visual_fusion_mode = "hybrid_gated"
        self.missing_mode = "conditioned"
        self.missing_context_dim = int(missing_context_dim)

        self.encoder = SmallCropEncoder(crop_in_channels, crop_base_channels, raw_feature_dim)
        self.visual_projector = VisualTokenProjector(
            visual_token_input_dim, visual_feature_dim, dropout
        )
        self.region_embedding = nn.Embedding(self.num_tooth_regions, tooth_region_embed_dim)
        self.fusion = GatedFusion(
            raw_feature_dim,
            visual_feature_dim,
            fused_feature_dim,
            extra_dim=tooth_region_embed_dim,
            dropout=dropout,
        )
        in_dim = int(fused_feature_dim) + int(tooth_region_embed_dim)
        self.missing_head = MLPHead(in_dim, 1, head_hidden_dim, dropout)
        self.missing_context = nn.Sequential(
            nn.Linear(in_dim, self.missing_context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attr_head = LabelSpecificHead(
            in_dim + self.missing_context_dim,
            self.num_tooth_labels - 1,
            head_hidden_dim,
            dropout,
        )

    def forward(
        self,
        tooth_crop: torch.Tensor,
        tooth_region: torch.Tensor,
        tooth_visual_token: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> dict:
        if tooth_visual_token is None:
            raise ValueError("tooth_visual_token is required by the released tooth branch")
        region = self.region_embedding(
            tooth_region.long().clamp(0, self.num_tooth_regions - 1)
        )
        raw = self.encoder(tooth_crop)
        visual = self.visual_projector(tooth_visual_token)
        fused = self.fusion(raw, visual, extra=region)
        tooth_feature = torch.cat([fused, region], dim=-1)
        missing_logit = self.missing_head(tooth_feature)
        gate = torch.sigmoid(missing_logit).detach()
        context = self.missing_context(tooth_feature) * gate
        attr_logits = self.attr_head(torch.cat([tooth_feature, context], dim=-1))
        output = {"tooth_logits": torch.cat([missing_logit, attr_logits], dim=-1)}
        if return_features:
            output.update(
                tooth_feature=tooth_feature,
                tooth_fused_feature=fused,
                tooth_region_feature=region,
            )
        return output


class CBCTEVIGREStage2RelationNet(nn.Module):
    """Final relation branch with gated crop/token fusion."""

    def __init__(
        self,
        crop_in_channels: int = 6,
        crop_base_channels: int = 12,
        raw_feature_dim: int = 192,
        visual_token_input_dim: int = 1408,
        visual_feature_dim: int = 192,
        fused_feature_dim: int = 384,
        visual_fusion_mode: str = "hybrid_gated",
        num_relation_labels: int = 8,
        num_relation_crops: int = 2,
        relation_id_embed_dim: int = 16,
        head_hidden_dim: int = 384,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        _require(visual_fusion_mode, "hybrid_gated", "visual_fusion_mode")
        self.num_relation_labels = int(num_relation_labels)
        self.num_relation_crops = int(num_relation_crops)
        self.visual_fusion_mode = "hybrid_gated"
        self.encoder = SmallCropEncoder(crop_in_channels, crop_base_channels, raw_feature_dim)
        self.visual_projector = VisualTokenProjector(
            visual_token_input_dim, visual_feature_dim, dropout
        )
        self.relation_embedding = nn.Embedding(self.num_relation_crops, relation_id_embed_dim)
        raw_flat_dim = int(raw_feature_dim) * self.num_relation_crops
        visual_flat_dim = int(visual_feature_dim) * self.num_relation_crops
        embedding_dim = int(relation_id_embed_dim) * self.num_relation_crops
        self.fusion = GatedFusion(
            raw_flat_dim,
            visual_flat_dim,
            fused_feature_dim,
            extra_dim=embedding_dim,
            dropout=dropout,
        )
        self.head = MLPHead(
            int(fused_feature_dim) + embedding_dim,
            self.num_relation_labels,
            head_hidden_dim,
            dropout,
        )

    def forward(
        self,
        relation_crops: torch.Tensor,
        relation_visual_tokens: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> dict:
        if relation_visual_tokens is None:
            raise ValueError("relation_visual_tokens are required by the released relation branch")
        batch_size, num_crops = relation_crops.shape[:2]
        if num_crops != self.num_relation_crops:
            raise ValueError(f"Expected {self.num_relation_crops} relation crops, got {num_crops}")
        identifiers = torch.arange(num_crops, device=relation_crops.device)
        identifiers = identifiers.unsqueeze(0).expand(batch_size, num_crops)
        embedding = self.relation_embedding(identifiers).reshape(batch_size, -1)
        flattened = relation_crops.reshape(batch_size * num_crops, *relation_crops.shape[2:])
        raw = self.encoder(flattened).reshape(batch_size, -1)
        visual = self.visual_projector(relation_visual_tokens).reshape(batch_size, -1)
        fused = self.fusion(raw, visual, extra=embedding)
        relation_feature = torch.cat([fused, embedding], dim=-1)
        output = {"relation_logits": self.head(relation_feature)}
        if return_features:
            output.update(relation_feature=relation_feature, relation_fused_feature=fused)
        return output


class CBCTEVIGREStage2GlobalNet(nn.Module):
    """Final global branch using the Stage 2 CBCT encoder."""

    def __init__(
        self,
        input_channels: int = 6,
        n_stages: int = 7,
        features_per_stage: Union[int, List[int], Tuple[int, ...]] = (
            32,
            64,
            128,
            256,
            320,
            320,
            320,
        ),
        conv_op: Type[nn.Module] = nn.Conv3d,
        kernel_sizes: Union[int, List[int], Tuple[int, ...]] = 3,
        strides: Union[int, List[int], Tuple[int, ...]] = (
            (1, 1, 1),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (1, 1, 1),
        ),
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]] = (1, 3, 4, 6, 6, 6, 6),
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm3d,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        mamba_d_state: int = 32,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 80,
        mamba_use_mem_eff_path: bool = True,
        visual_token_input_dim: int = 1408,
        visual_feature_dim: int = 320,
        fused_feature_dim: int = 320,
        visual_fusion_mode: str = "scratch_only",
        num_global_labels: int = 12,
        head_hidden_dim: int = 512,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        del visual_token_input_dim, visual_feature_dim, fused_feature_dim, kwargs
        _require(visual_fusion_mode, "scratch_only", "visual_fusion_mode")
        norm_op_kwargs = {"eps": 1e-5, "affine": True} if norm_op_kwargs is None else norm_op_kwargs
        nonlin_kwargs = {"inplace": True} if nonlin_kwargs is None else nonlin_kwargs
        self.num_global_labels = int(num_global_labels)
        self.visual_fusion_mode = "scratch_only"
        self.encoder = ResidualEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            block=BasicBlockD,
            return_skips=True,
            stem_channels=(
                features_per_stage[0]
                if isinstance(features_per_stage, (list, tuple))
                else features_per_stage
            ),
        )
        final_dim = int(self.encoder.output_channels[-1])
        self.mamba_layer = MambaLayer2(
            final_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            headdim=mamba_headdim,
            use_mem_eff_path=mamba_use_mem_eff_path,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.norm = nn.LayerNorm(final_dim)
        self.head = MLPHead(final_dim, self.num_global_labels, head_hidden_dim, dropout)

    def forward(
        self,
        global_image: torch.Tensor,
        global_visual_tokens: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> dict:
        del global_visual_tokens
        skips = list(self.encoder(global_image))
        encoded = self.mamba_layer(skips[-1])
        feature = self.norm(self.pool(encoded).flatten(1))
        output = {"global_logits": self.head(feature)}
        if return_features:
            output["global_feature"] = feature
        return output


for _class in (
    CBCTEVIGREStage2ToothInstanceNet,
    CBCTEVIGREStage2RelationNet,
    CBCTEVIGREStage2GlobalNet,
):
    _class.allowed_init_kwargs = set(inspect.signature(_class.__init__).parameters) - {"self"}
