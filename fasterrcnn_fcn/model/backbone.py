"""
Faster R-CNN + FCN: Backbone Network
====================================
ResNet50 + FPN backbone with pretrained ImageNet weights.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from collections import OrderedDict
from typing import Dict, List, Tuple


class BackboneWithFPN(nn.Module):
    """
    ResNet50 backbone with Feature Pyramid Network.
    
    Features:
        - Pretrained ImageNet weights
        - Multi-scale feature maps via FPN
        - Shared backbone for detection and segmentation
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        trainable_layers: int = 3,
        out_channels: int = 256,
        returned_layers: List[int] = None
    ):
        """
        Initialize the backbone.
        
        Args:
            pretrained: Use ImageNet pretrained weights
            trainable_layers: Number of trainable layers (from the end)
            out_channels: Number of output channels for FPN
            returned_layers: Which ResNet layers to return (1-4)
        """
        super().__init__()
        
        if returned_layers is None:
            returned_layers = [1, 2, 3, 4]
        
        self.returned_layers = returned_layers
        self.out_channels = out_channels
        
        # Load pretrained ResNet50
        if pretrained:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            backbone = resnet50(weights=None)
        
        # Freeze layers based on trainable_layers
        # ResNet50 has: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        layers_to_freeze = 5 - trainable_layers
        
        freezable_layers = [
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        ]
        
        for idx, layer in enumerate(freezable_layers):
            if idx < layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Extract backbone components
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1  # C2: 256 channels, stride 4
        self.layer2 = backbone.layer2  # C3: 512 channels, stride 8
        self.layer3 = backbone.layer3  # C4: 1024 channels, stride 16
        self.layer4 = backbone.layer4  # C5: 2048 channels, stride 32
        
        # Channel dimensions for each layer
        self.in_channels_list = {
            1: 256,   # layer1 output
            2: 512,   # layer2 output
            3: 1024,  # layer3 output
            4: 2048   # layer4 output
        }
        
        # Build FPN
        in_channels_for_fpn = [self.in_channels_list[i] for i in returned_layers]
        
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_for_fpn,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool()
        )
        
        # Feature names for FPN output
        self.feature_names = [str(i) for i in returned_layers]
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through backbone + FPN.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Dictionary of feature maps at different scales
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        c2 = self.layer1(x)   # stride 4
        c3 = self.layer2(c2)  # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32
        
        # Collect features based on returned_layers
        layer_outputs = {1: c2, 2: c3, 3: c4, 4: c5}
        
        features = OrderedDict()
        for i in self.returned_layers:
            features[str(i)] = layer_outputs[i]
        
        # Apply FPN
        fpn_features = self.fpn(features)
        
        return fpn_features
    
    def get_out_channels(self) -> int:
        """Get the number of output channels."""
        return self.out_channels
    
    def get_feature_strides(self) -> Dict[str, int]:
        """Get the stride of each feature level."""
        strides = {
            '1': 4,
            '2': 8,
            '3': 16,
            '4': 32,
            'pool': 64  # From LastLevelMaxPool
        }
        return {k: strides[k] for k in self.fpn.get_result_keys() if k in strides}


def create_backbone(
    pretrained: bool = True,
    trainable_layers: int = 3,
    out_channels: int = 256
) -> BackboneWithFPN:
    """
    Create a ResNet50-FPN backbone.
    
    Args:
        pretrained: Use ImageNet pretrained weights
        trainable_layers: Number of trainable layers
        out_channels: FPN output channels
        
    Returns:
        BackboneWithFPN instance
    """
    return BackboneWithFPN(
        pretrained=pretrained,
        trainable_layers=trainable_layers,
        out_channels=out_channels
    )


if __name__ == "__main__":
    # Test the backbone
    backbone = create_backbone(pretrained=True, trainable_layers=3)
    
    # Create dummy input
    x = torch.randn(2, 3, 512, 512)
    
    # Forward pass
    features = backbone(x)
    
    print("Backbone output features:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    print(f"\nOutput channels: {backbone.get_out_channels()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
