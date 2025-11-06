import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_bottleneck: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
       
        if use_bottleneck:
            mid_channels = out_channels // 4
            self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(mid_channels)
            self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(mid_channels)
            self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.conv3 = None
            self.bn3 = None
       
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
       
        self.dropout = nn.Dropout2d(dropout)
        self.use_bottleneck = use_bottleneck
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
       
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
       
        if self.use_bottleneck and self.conv3 is not None:
            out = self.bn3(self.conv3(out))
       
        out = self.dropout(out)
        out += identity
        out = F.relu(out)
       
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
       
        pool = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.sigmoid(self.conv(pool))
       
        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
       
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
       
        avg_pool = self.avg_pool(x).view(b, c)
        max_pool = self.max_pool(x).view(b, c)
       
        avg_attention = self.fc(avg_pool)
        max_attention = self.fc(max_pool)
       
        attention = self.sigmoid(avg_attention + max_attention).view(b, c, 1, 1)
       
        return x * attention

class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class DetectorCNN(nn.Module):
    def __init__(
        self,
        num_input_channels: int = 1,
        num_classes: int = 5,
        base_channels: int = 64,
        num_blocks: List[int] = [3, 4, 6, 3],
        use_attention: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
       
        self.num_classes = num_classes
        self.use_attention = use_attention
       
        self.conv1 = nn.Conv2d(
            num_input_channels, base_channels,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
       
        self.layer1 = self._make_layer(base_channels, base_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, num_blocks[3], stride=2)
       
        if use_attention:
            self.attention1 = CBAM(base_channels)
            self.attention2 = CBAM(base_channels * 2)
            self.attention3 = CBAM(base_channels * 4)
            self.attention4 = CBAM(base_channels * 8)
       
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool_global = nn.AdaptiveMaxPool2d((1, 1))
       
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 8 * 2, base_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, num_classes)
        )
       
        self.energy_head = nn.Sequential(
            nn.Linear(base_channels * 8 * 2, base_channels),
            nn.ReLU(),
            nn.Linear(base_channels, 1),
            nn.Softplus()
        )
       
        self._initialize_weights()
   
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int
    ) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_bottleneck=True))
       
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, use_bottleneck=True))
       
        return nn.Sequential(*layers)
   
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
   
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
       
        x = self.layer1(x)
        if self.use_attention:
            x = self.attention1(x)
       
        x = self.layer2(x)
        if self.use_attention:
            x = self.attention2(x)
       
        x = self.layer3(x)
        if self.use_attention:
            x = self.attention3(x)
       
        x = self.layer4(x)
        if self.use_attention:
            x = self.attention4(x)
       
        return x
   
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
       
        avg_pool = self.avgpool(features).view(features.size(0), -1)
        max_pool = self.maxpool_global(features).view(features.size(0), -1)
        global_features = torch.cat([avg_pool, max_pool], dim=1)
       
        class_logits = self.fc(global_features)
        energy = self.energy_head(global_features)
       
        return class_logits, energy

class MultiScaleDetectorCNN(nn.Module):
    def __init__(
        self,
        num_input_channels: int = 1,
        num_classes: int = 5,
        scales: List[int] = [1, 2, 4]
    ):
        super().__init__()
       
        self.scales = scales
       
        self.branches = nn.ModuleList([
            DetectorCNN(
                num_input_channels=num_input_channels,
                num_classes=num_classes,
                base_channels=64 // scale,
                num_blocks=[2, 2, 2, 2]
            )
            for scale in scales
        ])
       
        total_features = sum([64 * 8 * 2 // scale for scale in scales])
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
       
        for scale, branch in zip(self.scales, self.branches):
            if scale > 1:
                x_scaled = F.interpolate(
                    x,
                    scale_factor=1.0/scale,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                x_scaled = x
           
            feat = branch.extract_features(x_scaled)
            feat_pooled = F.adaptive_avg_pool2d(feat, (1, 1)).view(x.size(0), -1)
            features.append(feat_pooled)
       
        combined = torch.cat(features, dim=1)
        output = self.fusion(combined)
       
        return output

class FPGAOptimizedCNN(nn.Module):
    def __init__(
        self,
        num_input_channels: int = 1,
        num_classes: int = 5,
        bit_width: int = 8
    ):
        super().__init__()
       
        self.bit_width = bit_width
       
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
       
        self.features = nn.Sequential(
            nn.Conv2d(num_input_channels, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
           
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
           
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
           
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
       
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
       
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x
   
    def fuse_model(self):
        for m in self.features:
            if type(m) == nn.Sequential:
                torch.quantization.fuse_modules(m, [['0', '1', '2']], inplace=True)

def export_to_onnx(model: nn.Module, filepath: str, input_shape: Tuple[int, ...]):
    model.eval()
    dummy_input = torch.randn(*input_shape)
   
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['detector_image'],
        output_names=['class_logits', 'energy'],
        dynamic_axes={
            'detector_image': {0: 'batch_size'},
            'class_logits': {0: 'batch_size'},
            'energy': {0: 'batch_size'}
        }
    )
    logger.info(f"Model exported to {filepath}")

if __name__ == "__main__":
    logger.info("Initializing Advanced Detector CNN for SLAC research...")
   
    model = DetectorCNN(
        num_input_channels=1,
        num_classes=5,
        base_channels=64,
        use_attention=True
    )
   
    fpga_model = FPGAOptimizedCNN(
        num_input_channels=1,
        num_classes=5,
        bit_width=8
    )
   
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
   
    logger.info(f"Standard model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"FPGA model parameters: {sum(p.numel() for p in fpga_model.parameters()):,}")
    logger.info(f"Device: {device}")
   
    batch_size = 8
    dummy_input = torch.randn(batch_size, 1, 224, 224).to(device)
   
    model = model.to(device)
    class_logits, energy = model(dummy_input)
   
    logger.info(f"Output shapes - Class: {class_logits.shape}, Energy: {energy.shape}")
    logger.info("Model ready for SLAC detector data analysis!")