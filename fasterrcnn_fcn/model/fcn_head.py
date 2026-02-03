"""
Faster R-CNN + FCN: FCN Segmentation Head
==========================================
Fully Convolutional Network for pixel-wise segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class FCNHead(nn.Module):
    """
    Fully Convolutional Network segmentation head.
    
    Architecture:
        - Multi-scale feature fusion from FPN
        - Conv → ReLU → Conv layers
        - Bilinear upsampling to output resolution
        
    Supports both binary and multi-class segmentation.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 2,
        hidden_channels: int = 256,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the FCN head.
        
        Args:
            in_channels: Number of input channels from FPN
            num_classes: Number of segmentation classes (including background)
            hidden_channels: Number of hidden layer channels
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Feature fusion layers for multi-scale features
        self.lateral_convs = nn.ModuleDict({
            '1': nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            '2': nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            '3': nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            '4': nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
        })
        
        # Main segmentation head
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # Final classification layer
        self.classifier = nn.Conv2d(hidden_channels // 2, num_classes, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        features: Dict[str, torch.Tensor],
        target_size: tuple
    ) -> torch.Tensor:
        """
        Forward pass through FCN head.
        
        Args:
            features: Dictionary of FPN feature maps
            target_size: Target output size (H, W)
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Get the highest resolution feature map (P2)
        # and fuse with other scales
        
        # Start with highest resolution available
        if '1' in features:
            x = self.lateral_convs['1'](features['1'])
            target_feat_size = features['1'].shape[-2:]
        elif '2' in features:
            x = self.lateral_convs['2'](features['2'])
            target_feat_size = features['2'].shape[-2:]
        else:
            # Use whatever is available
            key = list(features.keys())[0]
            x = features[key]
            target_feat_size = x.shape[-2:]
        
        # Fuse multi-scale features by upsampling and adding
        for key in ['2', '3', '4']:
            if key in features and key in self.lateral_convs:
                feat = self.lateral_convs[key](features[key])
                feat = F.interpolate(
                    feat, 
                    size=target_feat_size, 
                    mode='bilinear', 
                    align_corners=False
                )
                x = x + feat
        
        # Pass through convolution layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        
        # Classification
        x = self.classifier(x)
        
        # Upsample to target size
        x = F.interpolate(
            x, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        return x


class FCNHeadSimple(nn.Module):
    """
    Simplified FCN head using only the highest resolution feature.
    
    Lighter weight alternative for faster inference.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 2,
        hidden_channels: int = 128
    ):
        """
        Initialize the simple FCN head.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of segmentation classes
            hidden_channels: Number of hidden channels
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        features: Dict[str, torch.Tensor],
        target_size: tuple
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Dictionary of FPN features
            target_size: Target output size (H, W)
            
        Returns:
            Segmentation logits
        """
        # Use highest resolution feature
        if '1' in features:
            x = features['1']
        elif '2' in features:
            x = features['2']
        else:
            x = list(features.values())[0]
        
        x = self.layers(x)
        
        # Upsample to target size
        x = F.interpolate(
            x,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        
        return x


def create_fcn_head(
    in_channels: int = 256,
    num_classes: int = 2,
    hidden_channels: int = 256,
    simple: bool = False
) -> nn.Module:
    """
    Create an FCN segmentation head.
    
    Args:
        in_channels: Number of input channels from backbone
        num_classes: Number of segmentation classes
        hidden_channels: Number of hidden channels
        simple: Use simplified FCN head
        
    Returns:
        FCN head module
    """
    if simple:
        return FCNHeadSimple(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_channels=hidden_channels
        )
    else:
        return FCNHead(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_channels=hidden_channels
        )


if __name__ == "__main__":
    # Test the FCN head
    from collections import OrderedDict
    
    # Create dummy FPN features
    features = OrderedDict({
        '1': torch.randn(2, 256, 128, 128),
        '2': torch.randn(2, 256, 64, 64),
        '3': torch.randn(2, 256, 32, 32),
        '4': torch.randn(2, 256, 16, 16),
    })
    
    # Test full FCN head
    fcn_head = create_fcn_head(
        in_channels=256,
        num_classes=5,
        hidden_channels=256,
        simple=False
    )
    
    output = fcn_head(features, target_size=(512, 512))
    print(f"FCN Head output shape: {output.shape}")
    
    # Test simple FCN head
    fcn_simple = create_fcn_head(
        in_channels=256,
        num_classes=5,
        simple=True
    )
    
    output_simple = fcn_simple(features, target_size=(512, 512))
    print(f"Simple FCN Head output shape: {output_simple.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in fcn_head.parameters())
    print(f"FCN Head parameters: {total_params:,}")
