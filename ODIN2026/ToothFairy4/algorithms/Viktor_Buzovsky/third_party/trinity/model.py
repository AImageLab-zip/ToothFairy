import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.get_num_of_classes import get_num_of_classes

class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True, eps=1e-5)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size, stride=1, padding=padding, bias=True)
        self.norm2 = nn.InstanceNorm3d(channels, affine=True, eps=1e-5)
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm2(self.conv2(out))
        out += residual
        return self.act2(out)


class ToothFairyUNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # Features per stage: [32, 64, 128, 256, 320, 320, 320]
        self.filters = [32, 64, 128, 256, 320, 320, 320]

        # Strides: Note the anisotropic strides in later stages [1, 2, 2]
        self.strides = [
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [1, 2, 2],
            [1, 2, 2]
        ]

        self.n_blocks_encoder = [1, 3, 4, 6, 6, 6, 6]

        # --- ENCODER ---
        self.encoder_stages = nn.ModuleList()
        current_channels = input_channels

        for stage_idx, out_channels in enumerate(self.filters):
            stage_blocks = []
            stride = self.strides[stage_idx]

            # Downsampling / Initial Conv
            stage_blocks.append(
                ConvNormAct(current_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            )

            # Residual Blocks
            for _ in range(self.n_blocks_encoder[stage_idx]):
                stage_blocks.append(ResidualBlock(out_channels))

            self.encoder_stages.append(nn.Sequential(*stage_blocks))
            current_channels = out_channels

        # --- DECODER ---
        self.decoder_stages = nn.ModuleList()
        self.upsample_ops = nn.ModuleList()

        # We track exactly what channels the decoder outputs at each step to build heads correctly
        self.decoder_output_channels = []

        # We go backward from the last stage up to the first
        for i in range(len(self.filters) - 2, -1, -1):
            input_ch = self.filters[i + 1]
            skip_ch = self.filters[i]
            output_ch = self.filters[i]


            stride = self.strides[i + 1]
            self.upsample_ops.append(
                nn.ConvTranspose3d(input_ch, input_ch, kernel_size=stride, stride=stride)
            )

            self.decoder_stages.append(
                ConvNormAct(input_ch + skip_ch, output_ch, kernel_size=3, stride=1, padding=1)
            )
            self.decoder_output_channels.append(output_ch)


        # --- SEGMENTATION HEADS ---
        # We create a head for EVERY decoder stage.
        # decoder_output_channels is ordered [Deepest -> Shallowest]
        self.seg_heads = nn.ModuleList()
        for ch in self.decoder_output_channels:
            self.seg_heads.append(
                nn.Conv3d(ch, num_classes, kernel_size=1, stride=1, bias=True)
            )

    def forward(self, x):
        skips = []
        out = x

        # --- ENCODER ---
        for stage in self.encoder_stages:
            out = stage(out)
            skips.append(out)

        bottleneck = skips.pop()

        # --- DECODER ---
        # Decoder outputs will hold feature maps from Deep -> Shallow
        decoder_feature_maps = []
        x = bottleneck

        for upsample, block, skip in zip(self.upsample_ops, self.decoder_stages, reversed(skips)):
            x = upsample(x)

            # Interpolate if shapes don't match (e.g. odd padding)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = block(x)
            decoder_feature_maps.append(x)

        # --- HEADS ---
        # Apply every head to its corresponding decoder feature map
        # decoder_feature_maps list is [Deepest, ..., Shallowest]
        # self.seg_heads list is [Head for Deepest, ..., Head for Shallowest]
        raw_outputs = []
        for head, feature_map in zip(self.seg_heads, decoder_feature_maps):
            raw_outputs.append(head(feature_map))

        # nnU-Net convention: Return list ordered [HighRes, LowRes, LowerRes ...]
        # Our raw_outputs are [LowRes, ..., HighRes]. So we reverse them.
        outputs = raw_outputs[::-1]

        if self.deep_supervision:
            return outputs  # Returns list: [FullRes, 1/2 Res, 1/4 Res...]
        else:
            return outputs[0]  # Return only FullRes


if __name__ == "__main__":
    import os
    import torch
    import json

    # --- IMPORTS FROM YOUR PROJECT ---
    # Ensure dataset_1.py is in the same directory or python path
    try:
        from dataset import get_toothfairy_loaders
    except ImportError:
        print("❌ Error: Could not import 'get_toothfairy_loaders' from 'dataset_1.py'.")
        print("Make sure both files are in the same directory.")
        exit(1)

    print("--- 🦷 ToothFairy Model Integration Test ---")

    # 1. SETUP & CONFIGURATION
    # We load the config to get the correct paths and patch sizes
    if not os.path.exists("config.json"):
        print("❌ Error: 'config.json' not found. Please create it or provide paths manually.")
        exit(1)

    with open("config.json") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. INSTANTIATE MODEL
    # Note: We use the patch_size from config to verify deep supervision later
    model = ToothFairyUNet(
        input_channels=1,
        num_classes=2, # 2 for IAN only (Background, IAN)
        deep_supervision=False
    )
    model.to(device)

    # pass random tensor to go through the forward pass
    random_data = torch.randn(1, 1, *config["patch_size"]).to(device)
    try:
        with torch.no_grad():
            _ = model(random_data)
    except Exception as e:
        print(f"❌ Model forward pass failed with random data: {e}")


    exit(0)

    print("✅ Model instantiated and moved to device.")

    # 3. INITIALIZE DATALOADERS
    # We only need the train loader for this shape test
    print("\n--- Initializing Data Loaders ---")
    try:
        train_loader, _ = get_toothfairy_loaders(
            root_dir=config["root_dir"],
            split_ratio=config.get("train_split_ratio", 0.8),
            train_bs=config["batch_size"],
            val_bs=config["val_batch_size"],
            workers=config.get("num_workers", 0),
            patch_target_classes=config["patch_target_classes"],
            patch_size=config["patch_size"]
        )
        print(f"✅ Train Loader ready. Batch size: {config['batch_size']}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        exit(1)

    # 4. FETCH ONE REAL BATCH
    print("\n--- Fetching One Batch ---")
    batch = next(iter(train_loader))

    images = batch['image'].to(device)
    labels = batch['label'].to(device)
    case_ids = batch['id']

    # CRITICAL CHECK: Ensure Channel Dimension Exists
    # Model expects (Batch, Channel, D, H, W)
    # If loader returns (Batch, D, H, W), we must unsqueeze.
    if images.ndim == 4:
        print("⚠️  Warning: Input missing channel dimension. Unsqueezing...")
        images = images.unsqueeze(1)

    # Labels usually come as (Batch, D, H, W) for CrossEntropy, or (B, C, D, H, W) for Dice
    # We don't change labels here, just checking input to model.

    print(f"Input Shape: {images.shape}")
    print(f"Label Shape: {labels.shape}")

    # 5. FORWARD PASS
    print("\n--- Running Forward Pass ---")
    try:
        outputs = model(images)

        # 6. VERIFICATION
        if isinstance(outputs, list):
            print(f"✅ Deep Supervision Enabled. Received {len(outputs)} outputs.")

            # Detailed check of the Full Resolution output
            full_res = outputs[0]
            print(f"Full Res Output Shape:   {full_res.shape}")

            # Check 1: Batch size
            assert full_res.shape[0] == images.shape[0], "❌ Batch size mismatch"

            # Check 2: Spatial dimensions (D, H, W)
            # full_res shape is (B, NumClasses, D, H, W)
            assert full_res.shape[2:] == images.shape[
                                         2:], f"❌ Shape mismatch: In {images.shape[2:]} vs Out {full_res.shape[2:]}"

            # Check 3: Deep Supervision Shapes
            # Usually deep supervision outputs are 1/2, 1/4, 1/8 size.
            # We check the second output (should be half size if strides are 2)
            if len(outputs) > 1:
                low_res = outputs[1]
                print(f"1/2 Res Output Shape:    {low_res.shape}")
                # Note: Depending on your strides, this might be 1/1 (if first stride is 1) or 1/2.
                # Just printing it allows you to verify your 'strides' list logic.

            print("\n🎉 SUCCESS: The model accepted the real data loader batch and produced valid outputs.")
        else:
            print(f"Output Shape: {outputs.shape}")
            print("\n🎉 SUCCESS: Forward pass complete (Deep Supervision OFF).")

    except RuntimeError as e:
        print(f"\n❌ FAILED: Runtime Error during forward pass.")
        print(e)
        # Common tip for OOM
        if "out of memory" in str(e).lower():
            print("\n💡 Tip: Try reducing 'batch_size' in config.json or 'filters' in model.py")