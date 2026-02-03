"""
Faster R-CNN + FCN: Training Script
===================================
End-to-end training with mixed precision support.
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import cv2
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import DetectionSegmentationDataset, create_dataloaders
from model.fasterrcnn_fcn import create_model
from losses import CombinedLoss, compute_total_loss
from utils import (
    set_seed, get_device, save_checkpoint, load_checkpoint,
    AverageMeter, collate_fn
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Faster R-CNN + FCN model'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir', type=str, 
        default=r'C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Lens_Kabin_Aralik_CNN',
        help='Path to dataset directory containing train/valid/test folders'
    )
    parser.add_argument(
        '--output-dir', type=str, default='checkpoints',
        help='Output directory for checkpoints'
    )
    
    # Model arguments
    parser.add_argument(
        '--num-classes', type=int, default=3,
        help='Number of detection classes (including background)'
    )
    parser.add_argument(
        '--num-seg-classes', type=int, default=3,
        help='Number of segmentation classes'
    )
    parser.add_argument(
        '--pretrained', action='store_true', default=True,
        help='Use pretrained backbone'
    )
    parser.add_argument(
        '--trainable-layers', type=int, default=3,
        help='Number of trainable backbone layers'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay', type=float, default=1e-4,
        help='Weight decay'
    )
    parser.add_argument(
        '--lr-step', type=int, default=10,
        help='LR scheduler step size'
    )
    parser.add_argument(
        '--lr-gamma', type=float, default=0.1,
        help='LR scheduler gamma'
    )
    parser.add_argument(
        '--lambda-seg', type=float, default=1.0,
        help='Segmentation loss weight'
    )
    
    # Other arguments
    parser.add_argument(
        '--image-size', type=int, default=512,
        help='Input image size'
    )
    parser.add_argument(
        '--num-workers', type=int, default=4,
        help='Number of data loader workers'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--amp', action='store_true', default=True,
        help='Use automatic mixed precision'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--save-freq', type=int, default=5,
        help='Checkpoint save frequency (epochs)'
    )
    parser.add_argument(
        '--log-freq', type=int, default=10,
        help='Logging frequency (iterations)'
    )
    
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_fn: CombinedLoss,
    scaler: Optional[GradScaler] = None,
    log_freq: int = 10,
    use_amp: bool = True
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        loss_fn: Combined loss function
        scaler: Gradient scaler for AMP
        log_freq: Logging frequency
        use_amp: Whether to use AMP
        
    Returns:
        Dictionary of average losses for the epoch
    """
    model.train()
    
    # Meters for tracking losses
    loss_meter = AverageMeter('Total Loss')
    det_loss_meter = AverageMeter('Detection Loss')
    seg_loss_meter = AverageMeter('Segmentation Loss')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move data to device
        images = [img.to(device) for img in images]
        
        # Prepare detection targets
        det_targets = []
        seg_targets = []
        
        for target in targets:
            det_target = {
                'boxes': target['boxes'].to(device),
                'labels': target['labels'].to(device)
            }
            det_targets.append(det_target)
            seg_targets.append(target['segmentation'].to(device))
        
        # Stack segmentation targets
        seg_targets = torch.stack(seg_targets, dim=0)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                # Forward pass
                det_losses, seg_output = model(images, det_targets)
                
                # Compute combined loss
                total_loss, loss_dict = loss_fn(det_losses, seg_output, seg_targets)
            
            # Backward pass with scaled gradients
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            det_losses, seg_output = model(images, det_targets)
            
            # Compute combined loss
            total_loss, loss_dict = loss_fn(det_losses, seg_output, seg_targets)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
        
        # Update meters
        loss_meter.update(loss_dict['loss_total'].item())
        det_loss_meter.update(loss_dict['loss_det'].item())
        seg_loss_meter.update(loss_dict['loss_seg'].item())
        
        # Update progress bar
        if (batch_idx + 1) % log_freq == 0:
            pbar.set_postfix({
                'total': f'{loss_meter.avg:.4f}',
                'det': f'{det_loss_meter.avg:.4f}',
                'seg': f'{seg_loss_meter.avg:.4f}'
            })
    
    return {
        'loss_total': loss_meter.avg,
        'loss_det': det_loss_meter.avg,
        'loss_seg': seg_loss_meter.avg
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn: CombinedLoss
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        device: Device
        loss_fn: Combined loss function
        
    Returns:
        Dictionary of average validation losses
    """
    model.train()  # Keep in train mode for loss computation
    
    loss_meter = AverageMeter('Total Loss')
    det_loss_meter = AverageMeter('Detection Loss')
    seg_loss_meter = AverageMeter('Segmentation Loss')
    
    pbar = tqdm(val_loader, desc='Validation')
    
    for images, targets in pbar:
        # Move data to device
        images = [img.to(device) for img in images]
        
        det_targets = []
        seg_targets = []
        
        for target in targets:
            det_target = {
                'boxes': target['boxes'].to(device),
                'labels': target['labels'].to(device)
            }
            det_targets.append(det_target)
            seg_targets.append(target['segmentation'].to(device))
        
        seg_targets = torch.stack(seg_targets, dim=0)
        
        # Forward pass
        det_losses, seg_output = model(images, det_targets)
        
        # Compute loss
        total_loss, loss_dict = loss_fn(det_losses, seg_output, seg_targets)
        
        # Update meters
        loss_meter.update(loss_dict['loss_total'].item())
        det_loss_meter.update(loss_dict['loss_det'].item())
        seg_loss_meter.update(loss_dict['loss_seg'].item())
    
    return {
        'loss_total': loss_meter.avg,
        'loss_det': det_loss_meter.avg,
        'loss_seg': seg_loss_meter.avg
    }


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()
    
    # Create datasets and dataloaders
    print("Loading datasets...")
    
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "valid")
    
    train_dataset = DetectionSegmentationDataset(
        root_dir=train_dir,
        image_size=(args.image_size, args.image_size),
        train=True
    )
    
    val_dataset = DetectionSegmentationDataset(
        root_dir=val_dir,
        image_size=(args.image_size, args.image_size),
        train=False
    )
    
    # Update num_classes based on dataset
    num_classes = train_dataset.get_num_classes()
    num_seg_classes = num_classes
    print(f"Classes found: {train_dataset.get_class_names()}")
    print(f"Number of classes (with background): {num_classes}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    # Create model
    print("Creating model...")
    model = create_model(
        num_classes=num_classes,
        num_seg_classes=num_seg_classes,
        pretrained=args.pretrained,
        trainable_backbone_layers=args.trainable_layers
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step,
        gamma=args.lr_gamma
    )
    
    # Create loss function
    loss_fn = CombinedLoss(
        num_seg_classes=num_seg_classes,
        lambda_seg=args.lambda_seg,
        use_dice=True
    )
    
    # Create gradient scaler for AMP
    scaler = GradScaler() if args.amp else None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint_info = load_checkpoint(
                args.resume, model, optimizer, scheduler, scaler, device
            )
            start_epoch = checkpoint_info['epoch'] + 1
            best_loss = checkpoint_info['loss']
        else:
            print(f"Checkpoint not found: {args.resume}")
    
    # Training loop
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_losses = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            loss_fn=loss_fn,
            scaler=scaler,
            log_freq=args.log_freq,
            use_amp=args.amp
        )
        
        # Validate
        val_losses = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            loss_fn=loss_fn
        )
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Print results
        print(f"\nEpoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Total: {train_losses['loss_total']:.4f}, "
              f"Det: {train_losses['loss_det']:.4f}, "
              f"Seg: {train_losses['loss_seg']:.4f}")
        print(f"  Val   - Total: {val_losses['loss_total']:.4f}, "
              f"Det: {val_losses['loss_det']:.4f}, "
              f"Seg: {val_losses['loss_seg']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_losses['loss_total'] < best_loss:
            best_loss = val_losses['loss_total']
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=best_loss,
                filepath=os.path.join(output_dir, 'best_model.pth'),
                scheduler=scheduler,
                scaler=scaler
            )
            print(f"  -> New best model saved!")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_losses['loss_total'],
                filepath=os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pth'),
                scheduler=scheduler,
                scaler=scaler
            )
        
        print()
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs - 1,
        loss=val_losses['loss_total'],
        filepath=os.path.join(output_dir, 'final_model.pth'),
        scheduler=scheduler,
        scaler=scaler
    )
    
    print("=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    
    # ========================================
    # Final Evaluation with Metrics
    # ========================================
    print("\n" + "=" * 60)
    print("Running Final Evaluation on Validation Set...")
    print("=" * 60)
    
    model.eval()
    
    # Metrics storage
    all_preds = []
    all_targets = []
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []
    
    # Segmentation confusion matrix
    seg_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Final Evaluation'):
            images = [img.to(device) for img in images]
            
            # Get predictions
            detections, seg_output = model(images)
            
            # Process segmentation predictions
            seg_preds = seg_output.argmax(dim=1).cpu().numpy()
            
            for i, target in enumerate(targets):
                gt_mask = target['segmentation'].cpu().numpy()
                pred_mask = seg_preds[i]
                
                # Resize if needed
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = cv2.resize(
                        pred_mask.astype(np.float32),
                        (gt_mask.shape[1], gt_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(np.int64)
                
                # Update confusion matrix
                for true_cls in range(num_classes):
                    for pred_cls in range(num_classes):
                        seg_confusion[true_cls, pred_cls] += np.sum(
                            (gt_mask == true_cls) & (pred_mask == pred_cls)
                        )
                
                # Detection metrics
                det = detections[i]
                pred_boxes = det['boxes'].cpu().numpy()
                pred_labels = det['labels'].cpu().numpy()
                pred_scores = det['scores'].cpu().numpy()
                
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                # Filter by confidence
                conf_mask = pred_scores >= 0.25
                all_pred_boxes.append(pred_boxes[conf_mask])
                all_pred_labels.append(pred_labels[conf_mask])
                all_pred_scores.append(pred_scores[conf_mask])
                all_gt_boxes.append(gt_boxes)
                all_gt_labels.append(gt_labels)
    
    # Calculate Segmentation Metrics per class
    # Only for cizik (1) and siyah_nokta (2) - exclude background (0) and any other classes
    target_classes = [1, 2]  # cizik, siyah_nokta
    
    seg_metrics = {}
    for cls in target_classes:
        if cls >= num_classes:
            continue
        tp = seg_confusion[cls, cls]
        fp = seg_confusion[:, cls].sum() - tp
        fn = seg_confusion[cls, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # IoU (Jaccard)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        seg_metrics[f'class_{cls}'] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'iou': float(iou)
        }
    
    # Mean metrics for target classes only
    mean_precision = np.mean([seg_metrics[f'class_{c}']['precision'] for c in target_classes if f'class_{c}' in seg_metrics])
    mean_recall = np.mean([seg_metrics[f'class_{c}']['recall'] for c in target_classes if f'class_{c}' in seg_metrics])
    mean_f1 = np.mean([seg_metrics[f'class_{c}']['f1'] for c in target_classes if f'class_{c}' in seg_metrics])
    mean_iou = np.mean([seg_metrics[f'class_{c}']['iou'] for c in target_classes if f'class_{c}' in seg_metrics])
    
    # Calculate Detection mAP
    def calculate_ap(pred_boxes_list, pred_scores_list, gt_boxes_list, iou_thresh=0.25):
        """Calculate AP using 11-point interpolation method."""
        total_gt = sum(len(gt) for gt in gt_boxes_list)
        if total_gt == 0:
            return 0.0
        
        # Flatten all predictions with their image indices
        all_preds = []
        for img_idx, (boxes, scores) in enumerate(zip(pred_boxes_list, pred_scores_list)):
            for box_idx, (box, score) in enumerate(zip(boxes, scores)):
                all_preds.append({
                    'img_idx': img_idx,
                    'box': box,
                    'score': score
                })
        
        if len(all_preds) == 0:
            return 0.0
        
        # Sort by score (descending)
        all_preds.sort(key=lambda x: x['score'], reverse=True)
        
        # Track which GT boxes have been matched
        gt_matched = [np.zeros(len(gt), dtype=bool) for gt in gt_boxes_list]
        
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        
        for pred_idx, pred in enumerate(all_preds):
            img_idx = pred['img_idx']
            pred_box = pred['box']
            gt_boxes = gt_boxes_list[img_idx]
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[img_idx][gt_idx]:
                    continue
                
                # Calculate IoU
                x1 = max(pred_box[0], gt_box[0])
                y1 = max(pred_box[1], gt_box[1])
                x2 = min(pred_box[2], gt_box[2])
                y2 = min(pred_box[3], gt_box[3])
                
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                area2 = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                union = area1 + area2 - inter
                
                iou = inter / union if union > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_thresh and best_gt_idx >= 0:
                gt_matched[img_idx][best_gt_idx] = True
                tp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # Calculate precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / total_gt
        
        # 11-point interpolation for AP
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            precisions_at_recall = precision[recall >= t]
            if len(precisions_at_recall) > 0:
                ap += np.max(precisions_at_recall)
        ap /= 11.0
        
        return float(min(ap, 1.0))  # Ensure AP <= 1.0
    
    # Calculate mAP per class - only for target classes (cizik=1, siyah_nokta=2)
    det_metrics = {}
    det_target_classes = [1, 2]  # cizik, siyah_nokta
    
    ap_per_class = {}
    for cls in det_target_classes:
        cls_pred_boxes = []
        cls_pred_scores = []
        cls_gt_boxes = []
        
        for i in range(len(all_pred_boxes)):
            pred_mask = all_pred_labels[i] == cls
            cls_pred_boxes.append(all_pred_boxes[i][pred_mask])
            cls_pred_scores.append(all_pred_scores[i][pred_mask])
            
            gt_mask = all_gt_labels[i] == cls
            cls_gt_boxes.append(all_gt_boxes[i][gt_mask])
        
        ap = calculate_ap(cls_pred_boxes, cls_pred_scores, cls_gt_boxes, iou_thresh=0.25)
        ap_per_class[int(cls)] = ap
    
    mAP = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
    
    # Print Results
    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)
    
    print("\n📊 SEGMENTATION METRICS:")
    print(f"  Mean Precision: {mean_precision:.4f}")
    print(f"  Mean Recall: {mean_recall:.4f}")
    print(f"  Mean F1-Score: {mean_f1:.4f}")
    print(f"  Mean IoU (mIoU): {mean_iou:.4f}")
    
    print("\n  Per-Class Metrics:")
    class_names = train_dataset.get_class_names()
    class_name_map = {1: 'cizik', 2: 'siyah_nokta'}
    for cls in target_classes:
        cls_name = class_name_map.get(cls, f'class_{cls}')
        if f'class_{cls}' in seg_metrics:
            m = seg_metrics[f'class_{cls}']
            print(f"    {cls_name}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}, IoU={m['iou']:.4f}")
    
    print("\n📦 DETECTION METRICS:")
    print(f"  mAP@0.25: {mAP:.4f}")
    
    if ap_per_class:
        print("\n  Per-Class AP:")
        for cls, ap in ap_per_class.items():
            cls_name = class_name_map.get(cls, f'class_{cls}')
            print(f"    {cls_name}: AP={ap:.4f}")
    
    # Save all results to JSON
    results = {
        'training': {
            'epochs': args.epochs,
            'best_val_loss': float(best_loss),
            'final_train_loss': train_losses,
            'final_val_loss': val_losses
        },
        'segmentation': {
            'mean_precision': float(mean_precision),
            'mean_recall': float(mean_recall),
            'mean_f1': float(mean_f1),
            'mean_iou': float(mean_iou),
            'per_class': seg_metrics
        },
        'detection': {
            'mAP@0.25': float(mAP),
            'AP_per_class': {str(k): v for k, v in ap_per_class.items()}
        },
        'config': vars(args),
        'class_names': {str(k): v for k, v in class_names.items()}
    }
    
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {results_path}")


if __name__ == '__main__':
    main()
