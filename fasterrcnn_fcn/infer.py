"""
Faster R-CNN + FCN: Inference Script
====================================
Run inference on single images with visualization.
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import cv2
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.fasterrcnn_fcn import create_model
from utils import (
    get_device, load_checkpoint, denormalize_image,
    draw_boxes, overlay_mask, visualize_predictions
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference with Faster R-CNN + FCN model'
    )
    
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--image', type=str, required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save output image (default: input_name_output.jpg)'
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
        '--image-size', type=int, default=512,
        help='Input image size'
    )
    parser.add_argument(
        '--conf-threshold', type=float, default=0.5,
        help='Confidence threshold for detections'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Display result in a window'
    )
    parser.add_argument(
        '--class-names', type=str, nargs='+', default=None,
        help='Class names (e.g., --class-names background object1 object2)'
    )
    
    return parser.parse_args()


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (512, 512)
) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """
    Preprocess image for inference.
    
    Args:
        image_path: Path to input image
        target_size: Target size (H, W)
        
    Returns:
        Tuple of (preprocessed tensor, original image, original size)
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    original_image = image.copy()
    original_size = image.shape[:2]  # (H, W)
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image_resized = cv2.resize(image_rgb, (target_size[1], target_size[0]))
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_resized / 255.0 - mean) / std
    
    # Convert to tensor (C, H, W)
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
    
    return image_tensor, original_image, original_size


def postprocess_predictions(
    detections: Dict[str, torch.Tensor],
    segmentation: torch.Tensor,
    original_size: Tuple[int, int],
    conf_threshold: float = 0.5
) -> Tuple[List[List[float]], List[int], List[float], np.ndarray]:
    """
    Post-process model predictions.
    
    Args:
        detections: Detection outputs
        segmentation: Segmentation output
        original_size: Original image size (H, W)
        conf_threshold: Confidence threshold
        
    Returns:
        Tuple of (boxes, labels, scores, segmentation_mask)
    """
    # Process detections
    boxes = detections['boxes'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    
    # Filter by confidence
    mask = scores >= conf_threshold
    boxes = boxes[mask].tolist()
    labels = labels[mask].tolist()
    scores = scores[mask].tolist()
    
    # Process segmentation
    seg_mask = segmentation[0].argmax(dim=0).cpu().numpy()
    
    # Resize segmentation to original size
    seg_mask = cv2.resize(
        seg_mask.astype(np.float32),
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)
    
    return boxes, labels, scores, seg_mask


def run_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Run model inference.
    
    Args:
        model: The model
        image_tensor: Input image tensor
        device: Device
        
    Returns:
        Tuple of (detections, segmentation)
    """
    model.eval()
    
    with torch.no_grad():
        images = [image_tensor.to(device)]
        detections, segmentation = model(images)
    
    return detections[0], segmentation


def visualize_result(
    image: np.ndarray,
    boxes: List[List[float]],
    labels: List[int],
    scores: List[float],
    seg_mask: np.ndarray,
    class_names: Optional[Dict[int, str]] = None
) -> np.ndarray:
    """
    Visualize detection and segmentation results.
    
    Args:
        image: Original image (BGR)
        boxes: Detected bounding boxes
        labels: Class labels
        scores: Confidence scores
        seg_mask: Segmentation mask
        class_names: Optional class name mapping
        
    Returns:
        Visualization image
    """
    # Create a copy
    vis_image = image.copy()
    
    # Overlay segmentation mask
    vis_image = overlay_mask(vis_image, seg_mask, alpha=0.4)
    
    # Draw bounding boxes
    vis_image = draw_boxes(
        vis_image,
        boxes,
        labels,
        scores,
        class_names=class_names,
        thickness=2
    )
    
    # Add legend
    if class_names:
        legend_y = 30
        for cls_id, cls_name in class_names.items():
            if cls_id == 0:  # Skip background
                continue
            text = f"Class {cls_id}: {cls_name}"
            cv2.putText(
                vis_image, text, (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            legend_y += 25
    
    return vis_image


def main():
    """Main inference function."""
    args = parse_args()
    
    device = get_device()
    
    print("=" * 60)
    print("Faster R-CNN + FCN Inference")
    print("=" * 60)
    print()
    
    # Build class names dictionary
    class_names = None
    if args.class_names:
        class_names = {i: name for i, name in enumerate(args.class_names)}
        print(f"Class names: {class_names}")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = create_model(
        num_classes=args.num_classes,
        num_seg_classes=args.num_seg_classes,
        pretrained=False
    ).to(device)
    
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    print()
    
    # Preprocess image
    print(f"Processing image: {args.image}")
    image_tensor, original_image, original_size = preprocess_image(
        args.image,
        target_size=(args.image_size, args.image_size)
    )
    print(f"Original size: {original_size}")
    print()
    
    # Run inference
    print("Running inference...")
    detections, segmentation = run_inference(model, image_tensor, device)
    
    # Post-process
    boxes, labels, scores, seg_mask = postprocess_predictions(
        detections,
        segmentation,
        original_size,
        conf_threshold=args.conf_threshold
    )
    
    print(f"Detections found: {len(boxes)}")
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        label_name = class_names.get(label, str(label)) if class_names else str(label)
        print(f"  [{i+1}] {label_name}: {score:.3f} @ {[int(b) for b in box]}")
    print(f"Segmentation classes found: {np.unique(seg_mask).tolist()}")
    print()
    
    # Visualize
    result_image = visualize_result(
        original_image,
        boxes,
        labels,
        scores,
        seg_mask,
        class_names=class_names
    )
    
    # Save output
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_dir = os.path.dirname(args.image) or '.'
        output_path = os.path.join(output_dir, f"{base_name}_output.jpg")
    
    cv2.imwrite(output_path, result_image)
    print(f"Result saved to: {output_path}")
    
    # Show if requested
    if args.show:
        # Resize for display if too large
        max_display_size = 1024
        h, w = result_image.shape[:2]
        if max(h, w) > max_display_size:
            scale = max_display_size / max(h, w)
            display_size = (int(w * scale), int(h * scale))
            display_image = cv2.resize(result_image, display_size)
        else:
            display_image = result_image
        
        cv2.imshow('Faster R-CNN + FCN Result', display_image)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("=" * 60)


def infer_batch(
    model: torch.nn.Module,
    image_dir: str,
    output_dir: str,
    device: torch.device,
    image_size: int = 512,
    conf_threshold: float = 0.5,
    class_names: Optional[Dict[int, str]] = None
) -> None:
    """
    Run inference on a directory of images.
    
    Args:
        model: The model
        image_dir: Input directory
        output_dir: Output directory
        device: Device
        image_size: Input image size
        conf_threshold: Confidence threshold
        class_names: Class name mapping
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(extensions)
    ]
    
    print(f"Found {len(image_files)} images")
    
    model.eval()
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        try:
            # Preprocess
            image_tensor, original_image, original_size = preprocess_image(
                image_path,
                target_size=(image_size, image_size)
            )
            
            # Inference
            detections, segmentation = run_inference(model, image_tensor, device)
            
            # Post-process
            boxes, labels, scores, seg_mask = postprocess_predictions(
                detections,
                segmentation,
                original_size,
                conf_threshold=conf_threshold
            )
            
            # Visualize
            result_image = visualize_result(
                original_image,
                boxes,
                labels,
                scores,
                seg_mask,
                class_names=class_names
            )
            
            # Save
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, result_image)
            
            print(f"Processed: {image_file} ({len(boxes)} detections)")
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
