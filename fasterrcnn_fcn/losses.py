"""
Faster R-CNN + FCN: Loss Functions
==================================
Combined detection and segmentation losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SegmentationLoss(nn.Module):
    """
    Segmentation loss module supporting both binary and multi-class segmentation.
    
    Uses CrossEntropyLoss for multi-class or BCEWithLogitsLoss for binary.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        ignore_index: int = -100,
        class_weights: Optional[torch.Tensor] = None,
        use_dice: bool = True,
        dice_weight: float = 0.5
    ):
        """
        Initialize segmentation loss.
        
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Index to ignore in loss computation
            class_weights: Optional class weights for imbalanced data
            use_dice: Whether to include Dice loss
            dice_weight: Weight for Dice loss component
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        
        if num_classes == 2:
            # Binary segmentation
            self.ce_loss = nn.BCEWithLogitsLoss(
                pos_weight=class_weights[1] if class_weights is not None else None
            )
            self.binary = True
        else:
            # Multi-class segmentation
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=ignore_index
            )
            self.binary = False
    
    def dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1.0
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predicted logits
            target: Ground truth mask
            smooth: Smoothing factor
            
        Returns:
            Dice loss value
        """
        if self.binary:
            pred = torch.sigmoid(pred)
            pred = pred.view(-1)
            target = target.float().view(-1)
            
            intersection = (pred * target).sum()
            dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        else:
            pred = F.softmax(pred, dim=1)
            
            # One-hot encode target
            target_onehot = F.one_hot(target.long(), self.num_classes).permute(0, 3, 1, 2).float()
            
            # Compute Dice per class and average
            dice_per_class = []
            for c in range(self.num_classes):
                pred_c = pred[:, c].contiguous().view(-1)
                target_c = target_onehot[:, c].contiguous().view(-1)
                
                intersection = (pred_c * target_c).sum()
                dice_c = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
                dice_per_class.append(dice_c)
            
            dice = torch.stack(dice_per_class).mean()
        
        return 1. - dice
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute segmentation loss.
        
        Args:
            pred: Predicted logits (B, C, H, W) or (B, 1, H, W) for binary
            target: Ground truth mask (B, H, W)
            
        Returns:
            Total segmentation loss
        """
        if self.binary:
            # Binary: squeeze channel dimension
            pred = pred.squeeze(1)
            target = target.float()
            ce = self.ce_loss(pred, target)
        else:
            ce = self.ce_loss(pred, target.long())
        
        if self.use_dice:
            dice = self.dice_loss(pred if not self.binary else pred.unsqueeze(1), target)
            return (1 - self.dice_weight) * ce + self.dice_weight * dice
        
        return ce


class CombinedLoss(nn.Module):
    """
    Combined loss for detection + segmentation.
    
    Total loss = detection_loss + lambda_seg * segmentation_loss
    """
    
    def __init__(
        self,
        num_seg_classes: int = 2,
        lambda_seg: float = 1.0,
        use_dice: bool = True,
        dice_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize combined loss.
        
        Args:
            num_seg_classes: Number of segmentation classes
            lambda_seg: Weight for segmentation loss
            use_dice: Whether to use Dice loss for segmentation
            dice_weight: Weight for Dice loss component
            class_weights: Optional class weights for segmentation
        """
        super().__init__()
        
        self.lambda_seg = lambda_seg
        
        self.seg_loss = SegmentationLoss(
            num_classes=num_seg_classes,
            class_weights=class_weights,
            use_dice=use_dice,
            dice_weight=dice_weight
        )
    
    def forward(
        self,
        det_losses: Dict[str, torch.Tensor],
        seg_pred: torch.Tensor,
        seg_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            det_losses: Dictionary of detection losses from Faster R-CNN
            seg_pred: Segmentation predictions (B, C, H, W)
            seg_targets: Ground truth segmentation masks (B, H, W)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Sum detection losses
        det_loss = sum(loss for loss in det_losses.values())
        
        # Compute segmentation loss
        seg_loss = self.seg_loss(seg_pred, seg_targets)
        
        # Total loss
        total_loss = det_loss + self.lambda_seg * seg_loss
        
        # Build loss dictionary
        loss_dict = {
            'loss_det': det_loss,
            'loss_seg': seg_loss,
            'loss_total': total_loss
        }
        
        # Add individual detection losses
        for k, v in det_losses.items():
            loss_dict[k] = v
        
        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL = -alpha * (1 - p)^gamma * log(p)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Balancing factor
            gamma: Focusing parameter
            reduction: Reduction mode ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def compute_detection_loss(
    det_losses: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Compute total detection loss from Faster R-CNN loss dictionary.
    
    Args:
        det_losses: Dictionary containing:
            - loss_objectness: RPN objectness loss
            - loss_rpn_box_reg: RPN box regression loss
            - loss_classifier: ROI classification loss
            - loss_box_reg: ROI box regression loss
            
    Returns:
        Total detection loss
    """
    return sum(loss for loss in det_losses.values())


def compute_segmentation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 2,
    use_dice: bool = True,
    dice_weight: float = 0.5
) -> torch.Tensor:
    """
    Compute segmentation loss.
    
    Args:
        pred: Predicted logits (B, C, H, W)
        target: Ground truth mask (B, H, W)
        num_classes: Number of classes
        use_dice: Whether to include Dice loss
        dice_weight: Weight for Dice loss
        
    Returns:
        Segmentation loss
    """
    loss_fn = SegmentationLoss(
        num_classes=num_classes,
        use_dice=use_dice,
        dice_weight=dice_weight
    )
    return loss_fn(pred, target)


def compute_total_loss(
    det_losses: Dict[str, torch.Tensor],
    seg_pred: torch.Tensor,
    seg_target: torch.Tensor,
    num_seg_classes: int = 2,
    lambda_seg: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute combined detection + segmentation loss.
    
    Args:
        det_losses: Detection loss dictionary
        seg_pred: Segmentation predictions
        seg_target: Segmentation ground truth
        num_seg_classes: Number of segmentation classes
        lambda_seg: Segmentation loss weight
        
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    combined_loss = CombinedLoss(
        num_seg_classes=num_seg_classes,
        lambda_seg=lambda_seg
    )
    return combined_loss(det_losses, seg_pred, seg_target)


if __name__ == "__main__":
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test segmentation loss (multi-class)
    seg_loss = SegmentationLoss(num_classes=5, use_dice=True)
    pred = torch.randn(4, 5, 128, 128).to(device)
    target = torch.randint(0, 5, (4, 128, 128)).to(device)
    loss = seg_loss(pred, target)
    print(f"Segmentation loss (multi-class): {loss.item():.4f}")
    
    # Test segmentation loss (binary)
    seg_loss_binary = SegmentationLoss(num_classes=2, use_dice=True)
    pred_binary = torch.randn(4, 1, 128, 128).to(device)
    target_binary = torch.randint(0, 2, (4, 128, 128)).to(device)
    loss_binary = seg_loss_binary(pred_binary, target_binary)
    print(f"Segmentation loss (binary): {loss_binary.item():.4f}")
    
    # Test combined loss
    det_losses = {
        'loss_objectness': torch.tensor(0.5).to(device),
        'loss_rpn_box_reg': torch.tensor(0.3).to(device),
        'loss_classifier': torch.tensor(0.4).to(device),
        'loss_box_reg': torch.tensor(0.2).to(device)
    }
    
    combined = CombinedLoss(num_seg_classes=5, lambda_seg=1.0)
    total, loss_dict = combined(det_losses, pred, target)
    
    print("\nCombined loss:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Test focal loss
    focal = FocalLoss()
    focal_loss = focal(pred, target)
    print(f"\nFocal loss: {focal_loss.item():.4f}")
