"""
Faster R-CNN + FCN: Evaluation Script
======================================
Evaluate trained model with mAP, IoU, and Dice metrics.
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import DetectionSegmentationDataset
from model.fasterrcnn_fcn import create_model
from utils import (
    set_seed, get_device, load_checkpoint, collate_fn,
    calculate_iou, calculate_dice, calculate_map, calculate_box_iou
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate Faster R-CNN + FCN model'
    )
    
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-dir', type=str, default='datasets',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--annotations', type=str, default='datasets/annotations.json',
        help='Path to annotations JSON file'
    )
    parser.add_argument(
        '--num-classes', type=int, default=2,
        help='Number of detection classes'
    )
    parser.add_argument(
        '--num-seg-classes', type=int, default=2,
        help='Number of segmentation classes'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--image-size', type=int, default=512,
        help='Input image size'
    )
    parser.add_argument(
        '--num-workers', type=int, default=4,
        help='Number of data loader workers'
    )
    parser.add_argument(
        '--conf-threshold', type=float, default=0.5,
        help='Confidence threshold for detections'
    )
    parser.add_argument(
        '--iou-threshold', type=float, default=0.5,
        help='IoU threshold for mAP calculation'
    )
    parser.add_argument(
        '--output-file', type=str, default='evaluation_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


@torch.no_grad()
def evaluate_detection(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate detection performance with mAP.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader
        device: Device
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for mAP
        
    Returns:
        Dictionary of detection metrics
    """
    model.eval()
    
    # Collect predictions and ground truths per class
    all_pred_boxes = {}
    all_pred_scores = {}
    all_gt_boxes = {}
    
    for images, targets in tqdm(dataloader, desc='Evaluating detection'):
        images = [img.to(device) for img in images]
        
        # Get predictions
        detections, _ = model(images)
        
        # Process each image
        for i, (det, target) in enumerate(zip(detections, targets)):
            pred_boxes = det['boxes'].cpu().numpy()
            pred_labels = det['labels'].cpu().numpy()
            pred_scores = det['scores'].cpu().numpy()
            
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            
            # Filter by confidence
            mask = pred_scores >= conf_threshold
            pred_boxes = pred_boxes[mask]
            pred_labels = pred_labels[mask]
            pred_scores = pred_scores[mask]
            
            # Group by class
            for cls in set(list(pred_labels) + list(gt_labels)):
                if cls not in all_pred_boxes:
                    all_pred_boxes[cls] = []
                    all_pred_scores[cls] = []
                    all_gt_boxes[cls] = []
                
                # Add predictions for this class
                cls_mask = pred_labels == cls
                for box, score in zip(pred_boxes[cls_mask], pred_scores[cls_mask]):
                    all_pred_boxes[cls].append(box)
                    all_pred_scores[cls].append(score)
                
                # Add ground truths for this class
                gt_cls_mask = gt_labels == cls
                for box in gt_boxes[gt_cls_mask]:
                    all_gt_boxes[cls].append(box)
    
    # Calculate mAP
    mAP, ap_per_class = calculate_map(
        all_pred_boxes, all_pred_scores, all_gt_boxes, iou_threshold
    )
    
    results = {
        'mAP': mAP,
        'AP_per_class': {str(k): v for k, v in ap_per_class.items()}
    }
    
    return results


@torch.no_grad()
def evaluate_segmentation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Evaluate segmentation performance with IoU and Dice.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader
        device: Device
        num_classes: Number of segmentation classes
        
    Returns:
        Dictionary of segmentation metrics
    """
    model.eval()
    
    iou_per_class = {i: [] for i in range(num_classes)}
    dice_per_class = {i: [] for i in range(num_classes)}
    
    for images, targets in tqdm(dataloader, desc='Evaluating segmentation'):
        images = [img.to(device) for img in images]
        
        # Get predictions
        _, seg_output = model(images)
        
        # Get predicted masks
        pred_masks = seg_output.argmax(dim=1).cpu().numpy()
        
        # Get ground truth masks
        for i, target in enumerate(targets):
            gt_mask = target['segmentation'].cpu().numpy()
            pred_mask = pred_masks[i]
            
            # Resize prediction if needed
            if pred_mask.shape != gt_mask.shape:
                import cv2
                pred_mask = cv2.resize(
                    pred_mask.astype(np.float32),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int64)
            
            # Calculate per-class metrics
            for cls in range(num_classes):
                pred_cls = (pred_mask == cls)
                gt_cls = (gt_mask == cls)
                
                # Only calculate if class exists in ground truth or prediction
                if gt_cls.sum() > 0 or pred_cls.sum() > 0:
                    iou = calculate_iou(pred_cls, gt_cls)
                    dice = calculate_dice(pred_cls, gt_cls)
                    
                    iou_per_class[cls].append(iou)
                    dice_per_class[cls].append(dice)
    
    # Calculate mean metrics
    mean_iou_per_class = {}
    mean_dice_per_class = {}
    
    for cls in range(num_classes):
        if len(iou_per_class[cls]) > 0:
            mean_iou_per_class[cls] = float(np.mean(iou_per_class[cls]))
            mean_dice_per_class[cls] = float(np.mean(dice_per_class[cls]))
        else:
            mean_iou_per_class[cls] = 0.0
            mean_dice_per_class[cls] = 0.0
    
    # Overall mean (excluding background class 0)
    fg_ious = [mean_iou_per_class[c] for c in range(1, num_classes)]
    fg_dices = [mean_dice_per_class[c] for c in range(1, num_classes)]
    
    results = {
        'mIoU': float(np.mean(fg_ious)) if fg_ious else 0.0,
        'mDice': float(np.mean(fg_dices)) if fg_dices else 0.0,
        'IoU_per_class': {str(k): v for k, v in mean_iou_per_class.items()},
        'Dice_per_class': {str(k): v for k, v in mean_dice_per_class.items()}
    }
    
    return results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    print("=" * 60)
    print("Faster R-CNN + FCN Evaluation")
    print("=" * 60)
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = DetectionSegmentationDataset(
        root_dir=args.data_dir,
        annotations_file=args.annotations,
        image_size=(args.image_size, args.image_size),
        train=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Evaluation samples: {len(dataset)}")
    print()
    
    # Create model
    print("Loading model...")
    model = create_model(
        num_classes=args.num_classes,
        num_seg_classes=args.num_seg_classes,
        pretrained=False
    ).to(device)
    
    # Load checkpoint
    load_checkpoint(args.checkpoint, model, device=device)
    print()
    
    # Evaluate detection
    print("Evaluating detection...")
    det_results = evaluate_detection(
        model=model,
        dataloader=dataloader,
        device=device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    print(f"\nDetection Results:")
    print(f"  mAP@{args.iou_threshold}: {det_results['mAP']:.4f}")
    print(f"  AP per class:")
    for cls, ap in det_results['AP_per_class'].items():
        print(f"    Class {cls}: {ap:.4f}")
    print()
    
    # Evaluate segmentation
    print("Evaluating segmentation...")
    seg_results = evaluate_segmentation(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=args.num_seg_classes
    )
    
    print(f"\nSegmentation Results:")
    print(f"  mIoU: {seg_results['mIoU']:.4f}")
    print(f"  mDice: {seg_results['mDice']:.4f}")
    print(f"  IoU per class:")
    for cls, iou in seg_results['IoU_per_class'].items():
        print(f"    Class {cls}: {iou:.4f}")
    print(f"  Dice per class:")
    for cls, dice in seg_results['Dice_per_class'].items():
        print(f"    Class {cls}: {dice:.4f}")
    print()
    
    # Save results
    results = {
        'detection': det_results,
        'segmentation': seg_results,
        'config': {
            'checkpoint': args.checkpoint,
            'conf_threshold': args.conf_threshold,
            'iou_threshold': args.iou_threshold,
            'image_size': args.image_size
        }
    }
    
    output_path = args.output_file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
