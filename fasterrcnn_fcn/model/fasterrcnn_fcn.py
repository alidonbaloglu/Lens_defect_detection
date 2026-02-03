"""
Faster R-CNN + FCN: Combined Detection + Segmentation Model
===========================================================
End-to-end model with shared backbone for both tasks.
"""

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from typing import Dict, List, Tuple, Optional, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbone import create_backbone, BackboneWithFPN
from model.fcn_head import create_fcn_head, FCNHead


class FasterRCNNFCN(nn.Module):
    """
    Combined Faster R-CNN + FCN model for detection and segmentation.
    
    Features:
        - Shared ResNet50-FPN backbone
        - Faster R-CNN detection head (RPN + ROI heads)
        - FCN segmentation head
        - Joint forward pass returning both outputs
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        num_seg_classes: int = 2,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        image_mean: List[float] = None,
        image_std: List[float] = None,
        min_size: int = 512,
        max_size: int = 1024,
        # RPN parameters
        rpn_anchor_sizes: Tuple = ((32,), (64,), (128,), (256,), (512,)),
        rpn_aspect_ratios: Tuple = ((0.5, 1.0, 2.0),) * 5,
        rpn_fg_iou_thresh: float = 0.7,
        rpn_bg_iou_thresh: float = 0.3,
        rpn_batch_size_per_image: int = 256,
        rpn_positive_fraction: float = 0.5,
        rpn_pre_nms_top_n_train: int = 2000,
        rpn_pre_nms_top_n_test: int = 1000,
        rpn_post_nms_top_n_train: int = 2000,
        rpn_post_nms_top_n_test: int = 1000,
        rpn_nms_thresh: float = 0.7,
        rpn_score_thresh: float = 0.0,
        # ROI parameters
        box_roi_pool_featmap_names: List[str] = None,
        box_roi_pool_output_size: int = 7,
        box_roi_pool_sampling_ratio: int = 2,
        box_head_hidden_channels: int = 256,
        box_head_fc_hidden_channels: int = 1024,
        box_fg_iou_thresh: float = 0.5,
        box_bg_iou_thresh: float = 0.5,
        box_batch_size_per_image: int = 512,
        box_positive_fraction: float = 0.25,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        # FCN parameters
        fcn_hidden_channels: int = 256,
        fcn_simple: bool = False
    ):
        """
        Initialize the combined model.
        
        Args:
            num_classes: Number of detection classes (including background)
            num_seg_classes: Number of segmentation classes
            pretrained_backbone: Use pretrained backbone weights
            trainable_backbone_layers: Number of trainable backbone layers
            image_mean: Image normalization mean
            image_std: Image normalization std
            min_size: Minimum image size
            max_size: Maximum image size
            ... (RPN and ROI parameters)
            fcn_hidden_channels: FCN hidden layer channels
            fcn_simple: Use simplified FCN head
        """
        super().__init__()
        
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        
        if box_roi_pool_featmap_names is None:
            box_roi_pool_featmap_names = ['1', '2', '3', '4']
        
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        
        # Create shared backbone
        self.backbone = create_backbone(
            pretrained=pretrained_backbone,
            trainable_layers=trainable_backbone_layers
        )
        out_channels = self.backbone.get_out_channels()
        
        # Image transform
        self.transform = GeneralizedRCNNTransform(
            min_size=min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std
        )
        
        # RPN anchor generator
        anchor_generator = AnchorGenerator(
            sizes=rpn_anchor_sizes,
            aspect_ratios=rpn_aspect_ratios
        )
        
        # RPN head
        rpn_head = RPNHead(
            out_channels,
            anchor_generator.num_anchors_per_location()[0]
        )
        
        # RPN
        from torchvision.models.detection.rpn import RegionProposalNetwork
        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=rpn_fg_iou_thresh,
            bg_iou_thresh=rpn_bg_iou_thresh,
            batch_size_per_image=rpn_batch_size_per_image,
            positive_fraction=rpn_positive_fraction,
            pre_nms_top_n={
                'training': rpn_pre_nms_top_n_train,
                'testing': rpn_pre_nms_top_n_test
            },
            post_nms_top_n={
                'training': rpn_post_nms_top_n_train,
                'testing': rpn_post_nms_top_n_test
            },
            nms_thresh=rpn_nms_thresh,
            score_thresh=rpn_score_thresh
        )
        
        # ROI pooling
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=box_roi_pool_featmap_names,
            output_size=box_roi_pool_output_size,
            sampling_ratio=box_roi_pool_sampling_ratio
        )
        
        # Box head
        resolution = box_roi_pool_output_size
        representation_size = box_head_fc_hidden_channels
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size
        )
        
        # Box predictor
        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes
        )
        
        # ROI heads
        self.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=box_fg_iou_thresh,
            bg_iou_thresh=box_bg_iou_thresh,
            batch_size_per_image=box_batch_size_per_image,
            positive_fraction=box_positive_fraction,
            bbox_reg_weights=None,
            score_thresh=box_score_thresh,
            nms_thresh=box_nms_thresh,
            detections_per_img=box_detections_per_img
        )
        
        # FCN segmentation head
        self.fcn_head = create_fcn_head(
            in_channels=out_channels,
            num_classes=num_seg_classes,
            hidden_channels=fcn_hidden_channels,
            simple=fcn_simple
        )
        
        # Store original image sizes for segmentation output
        self._original_image_sizes = None
    
    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass returning both detection and segmentation outputs.
        
        Args:
            images: List of image tensors (C, H, W)
            targets: Optional list of target dicts for training
            
        Returns:
            Tuple of:
                - detections: Detection losses (training) or predictions (inference)
                - segmentation: Segmentation logits (B, num_seg_classes, H, W)
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be provided")
        
        # Store original image sizes for segmentation
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))
        self._original_image_sizes = original_image_sizes
        
        # Transform images
        images, targets = self.transform(images, targets)
        
        # Get feature maps from backbone
        features = self.backbone(images.tensors)
        
        # RPN
        proposals, rpn_losses = self.rpn(images, features, targets)
        
        # ROI heads
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )
        
        # Segmentation
        # Use the first original image size as target (batch must have same size)
        target_size = original_image_sizes[0]
        seg_output = self.fcn_head(features, target_size)
        
        if self.training:
            losses = {}
            losses.update(rpn_losses)
            losses.update(detector_losses)
            return losses, seg_output
        else:
            # Post-process detections
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            return detections, seg_output
    
    def get_detection_losses(
        self,
        loss_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute total detection loss.
        
        Args:
            loss_dict: Dictionary of detection losses
            
        Returns:
            Total detection loss
        """
        return sum(loss for loss in loss_dict.values())


class RPNHead(nn.Module):
    """RPN classification and regression head."""
    
    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
        
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits = []
        bbox_reg = []
        for feature in x:
            t = torch.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class TwoMLPHead(nn.Module):
    """Two-layer MLP head for ROI features."""
    
    def __init__(self, in_channels: int, representation_size: int):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    """Fast R-CNN predictor for classification and box regression."""
    
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


def create_model(
    num_classes: int = 2,
    num_seg_classes: int = 2,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    **kwargs
) -> FasterRCNNFCN:
    """
    Create a Faster R-CNN + FCN model.
    
    Args:
        num_classes: Number of detection classes
        num_seg_classes: Number of segmentation classes
        pretrained: Use pretrained backbone
        trainable_backbone_layers: Number of trainable layers
        **kwargs: Additional model arguments
        
    Returns:
        FasterRCNNFCN model instance
    """
    model = FasterRCNNFCN(
        num_classes=num_classes,
        num_seg_classes=num_seg_classes,
        pretrained_backbone=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs
    )
    return model


if __name__ == "__main__":
    # Test the combined model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(
        num_classes=5,
        num_seg_classes=5,
        pretrained=True
    ).to(device)
    
    # Create dummy input
    images = [torch.randn(3, 512, 512).to(device) for _ in range(2)]
    
    # Create dummy targets
    targets = [
        {
            'boxes': torch.tensor([[100, 100, 200, 200], [50, 50, 150, 150]], dtype=torch.float32).to(device),
            'labels': torch.tensor([1, 2], dtype=torch.int64).to(device)
        },
        {
            'boxes': torch.tensor([[80, 80, 180, 180]], dtype=torch.float32).to(device),
            'labels': torch.tensor([3], dtype=torch.int64).to(device)
        }
    ]
    
    # Training mode
    model.train()
    det_losses, seg_output = model(images, targets)
    
    print("\nTraining mode:")
    print("Detection losses:")
    for k, v in det_losses.items():
        print(f"  {k}: {v.item():.4f}")
    print(f"Segmentation output shape: {seg_output.shape}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        detections, seg_output = model(images)
    
    print("\nInference mode:")
    for i, det in enumerate(detections):
        print(f"Image {i}:")
        print(f"  Boxes shape: {det['boxes'].shape}")
        print(f"  Labels shape: {det['labels'].shape}")
        print(f"  Scores shape: {det['scores'].shape}")
    print(f"Segmentation output shape: {seg_output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
