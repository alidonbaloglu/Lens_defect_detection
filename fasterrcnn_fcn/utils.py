"""
Faster R-CNN + FCN: Utility Functions
=====================================
Visualization helpers, metric functions, seed fixing, and model weight management.
"""

import os
import random
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Any


def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    """
    Get the best available device (CUDA or CPU).
    
    Returns:
        torch.device: The device to use for computation
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state to save
        epoch: Current epoch number
        loss: Current loss value
        filepath: Path to save the checkpoint
        scheduler: Optional learning rate scheduler
        scaler: Optional gradient scaler for AMP
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        model: The model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        scaler: Optional scaler to load state into
        device: Device to load the checkpoint to
        
    Returns:
        Dictionary containing checkpoint information
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    print(f"Checkpoint loaded: {filepath} (Epoch {checkpoint.get('epoch', 'N/A')})")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf'))
    }


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Intersection over Union for segmentation masks.
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        
    Returns:
        IoU score
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def calculate_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Dice coefficient for segmentation masks.
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth segmentation mask
        
    Returns:
        Dice score
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2 * intersection / total


def calculate_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two bounding boxes.
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
        
    Returns:
        IoU score
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_ap(
    pred_boxes: List[np.ndarray],
    pred_scores: List[float],
    gt_boxes: List[np.ndarray],
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate Average Precision for a single class.
    
    Args:
        pred_boxes: List of predicted bounding boxes
        pred_scores: List of confidence scores
        gt_boxes: List of ground truth boxes
        iou_threshold: IoU threshold for matching
        
    Returns:
        Average Precision score
    """
    if len(pred_boxes) == 0:
        return 0.0 if len(gt_boxes) > 0 else 1.0
    
    if len(gt_boxes) == 0:
        return 0.0
    
    # Sort predictions by score
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_indices]
    pred_scores = [pred_scores[i] for i in sorted_indices]
    
    gt_matched = [False] * len(gt_boxes)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            
            iou = calculate_box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1
    
    # Calculate precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recall = tp_cumsum / len(gt_boxes)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Add start and end points
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[1], precision, [0]])
    
    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # Calculate AP as area under the curve
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    
    return ap


def calculate_map(
    all_pred_boxes: Dict[int, List[np.ndarray]],
    all_pred_scores: Dict[int, List[float]],
    all_gt_boxes: Dict[int, List[np.ndarray]],
    iou_threshold: float = 0.5
) -> Tuple[float, Dict[int, float]]:
    """
    Calculate mean Average Precision across all classes.
    
    Args:
        all_pred_boxes: Dictionary of predicted boxes per class
        all_pred_scores: Dictionary of prediction scores per class
        all_gt_boxes: Dictionary of ground truth boxes per class
        iou_threshold: IoU threshold for matching
        
    Returns:
        Tuple of (mAP, dictionary of AP per class)
    """
    all_classes = set(all_pred_boxes.keys()) | set(all_gt_boxes.keys())
    
    ap_per_class = {}
    for cls in all_classes:
        pred_boxes = all_pred_boxes.get(cls, [])
        pred_scores = all_pred_scores.get(cls, [])
        gt_boxes = all_gt_boxes.get(cls, [])
        
        ap = calculate_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold)
        ap_per_class[cls] = ap
    
    if len(ap_per_class) == 0:
        return 0.0, {}
    
    mAP = np.mean(list(ap_per_class.values()))
    return mAP, ap_per_class


def draw_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    labels: List[int],
    scores: Optional[List[float]] = None,
    class_names: Optional[Dict[int, str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on an image.
    
    Args:
        image: Input image (BGR format)
        boxes: List of bounding boxes [x1, y1, x2, y2]
        labels: List of class labels
        scores: Optional list of confidence scores
        class_names: Optional dictionary mapping class IDs to names
        color: Box color in BGR format
        thickness: Line thickness
        
    Returns:
        Image with drawn boxes
    """
    image = image.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Create label text
        label = labels[i]
        if class_names is not None and label in class_names:
            label_text = class_names[label]
        else:
            label_text = f"Class {label}"
        
        if scores is not None:
            label_text += f" {scores[i]:.2f}"
        
        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image, 
            (x1, y1 - text_height - 10), 
            (x1 + text_width, y1),
            color, 
            -1
        )
        cv2.putText(
            image, 
            label_text, 
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
    
    return image


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay a segmentation mask on an image.
    
    Args:
        image: Input image (BGR format)
        mask: Segmentation mask (H, W) or (H, W, C)
        color: Mask color in BGR format
        alpha: Transparency of the overlay
        
    Returns:
        Image with overlaid mask
    """
    image = image.copy()
    
    if len(mask.shape) == 3:
        mask = mask.argmax(axis=0)
    
    # Create colored overlay
    overlay = np.zeros_like(image)
    unique_classes = np.unique(mask)
    
    # Generate colors for each class
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Olive
    ]
    
    for i, cls in enumerate(unique_classes):
        if cls == 0:  # Skip background
            continue
        class_mask = (mask == cls)
        overlay_color = colors[i % len(colors)] if color is None else color
        overlay[class_mask] = overlay_color
    
    # Blend overlay with image
    mask_binary = mask > 0
    image[mask_binary] = cv2.addWeighted(
        image[mask_binary], 
        1 - alpha,
        overlay[mask_binary], 
        alpha, 
        0
    )
    
    return image


def visualize_predictions(
    image: np.ndarray,
    boxes: List[List[float]],
    labels: List[int],
    scores: List[float],
    mask: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
    score_threshold: float = 0.5
) -> np.ndarray:
    """
    Visualize detection and segmentation predictions.
    
    Args:
        image: Input image (BGR format)
        boxes: List of bounding boxes
        labels: List of class labels
        scores: List of confidence scores
        mask: Optional segmentation mask
        class_names: Optional dictionary of class names
        score_threshold: Minimum score to display
        
    Returns:
        Visualization image
    """
    # Filter predictions by score
    valid_indices = [i for i, s in enumerate(scores) if s >= score_threshold]
    filtered_boxes = [boxes[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    filtered_scores = [scores[i] for i in valid_indices]
    
    # Draw segmentation mask if provided
    if mask is not None:
        image = overlay_mask(image, mask)
    
    # Draw bounding boxes
    image = draw_boxes(
        image, 
        filtered_boxes, 
        filtered_labels,
        filtered_scores, 
        class_names
    )
    
    return image


def denormalize_image(
    image: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Denormalize a normalized image tensor.
    
    Args:
        image: Normalized image tensor (C, H, W)
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized image as numpy array (H, W, C) in BGR format
    """
    image = image.cpu().numpy()
    
    # Denormalize
    for i in range(3):
        image[i] = image[i] * std[i] + mean[i]
    
    # Convert to uint8
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    
    # Convert RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def collate_fn(batch: List[Tuple]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Custom collate function for detection + segmentation.
    
    Args:
        batch: List of (image, target) tuples
        
    Returns:
        Tuple of (images list, targets list)
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets
