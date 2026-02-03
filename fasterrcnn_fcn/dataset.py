"""
Faster R-CNN + FCN: Dataset Class
=================================
PyTorch Dataset for combined detection and segmentation.
Adapted for Roboflow COCO Segmentation format.
"""

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DetectionSegmentationDataset(Dataset):
    """
    PyTorch Dataset for detection + segmentation tasks.
    
    Loads:
        - RGB images
        - Bounding boxes (xmin, ymin, xmax, ymax)
        - Class labels
        - Polygon segmentation masks
    
    Output format:
        image, {
            "boxes": Tensor[N, 4],
            "labels": Tensor[N],
            "segmentation": Tensor[H, W]
        }
    """
    
    def __init__(
        self,
        root_dir: str,
        transforms: Optional[Callable] = None,
        image_size: Tuple[int, int] = (512, 512),
        train: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing images and _annotations.coco.json
            transforms: Optional albumentations transforms
            image_size: Target image size (height, width)
            train: Whether this is training set (enables augmentations)
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.train = train
        
        # Load COCO annotations from root directory
        annotations_file = os.path.join(root_dir, "_annotations.coco.json")
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations_data = json.load(f)
        
        # Build image id to annotations mapping
        self.images = {img['id']: img for img in self.annotations_data['images']}
        self.categories = {cat['id']: cat['name'] for cat in self.annotations_data['categories']}
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.annotations_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        # List of valid image IDs
        self.image_ids = list(self.images.keys())
        
        # Set up transforms
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = self._get_default_transforms()
    
    def _get_default_transforms(self) -> A.Compose:
        """Get default albumentations transforms."""
        if self.train:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_visibility=0.3
            ))
        else:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_visibility=0.3
            ))
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def _polygon_to_mask(self, polygon: List[float], height: int, width: int) -> np.ndarray:
        """
        Convert polygon points to binary mask.
        
        Args:
            polygon: List of [x1, y1, x2, y2, ...] coordinates
            height: Image height
            width: Image width
            
        Returns:
            Binary mask of shape (height, width)
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        if len(polygon) >= 6:  # At least 3 points needed
            pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image tensor, target dict)
        """
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image from root directory (images are directly in root_dir)
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Get annotations for this image
        annotations = self.image_annotations.get(img_id, [])
        
        # Extract boxes, labels, and create segmentation mask
        boxes = []
        labels = []
        
        # Initialize empty segmentation mask
        mask = np.zeros((orig_h, orig_w), dtype=np.int64)
        
        for ann in annotations:
            # Bounding box: [x, y, width, height] -> [x1, y1, x2, y2]
            bbox = ann['bbox']
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            
            # Clip to image bounds
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['category_id'])
            
            # Create segmentation mask from polygon
            if 'segmentation' in ann and ann['segmentation']:
                for polygon in ann['segmentation']:
                    poly_mask = self._polygon_to_mask(polygon, orig_h, orig_w)
                    # Use category_id as the mask value
                    mask[poly_mask > 0] = ann['category_id']
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        
        # Apply transforms
        if len(boxes) > 0:
            transformed = self.transforms(
                image=image,
                mask=mask,
                bboxes=boxes.tolist(),
                labels=labels.tolist()
            )
        else:
            # Handle case with no boxes
            transformed = self.transforms(
                image=image,
                mask=mask,
                bboxes=[],
                labels=[]
            )
        
        image = transformed['image']
        mask = transformed['mask']
        transformed_boxes = transformed['bboxes']
        transformed_labels = transformed['labels']
        
        # Convert to tensors
        if len(transformed_boxes) > 0:
            boxes_tensor = torch.tensor(transformed_boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(transformed_labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        
        # Ensure mask is a tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.int64)
        
        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'segmentation': mask,
            'image_id': torch.tensor([img_id])
        }
        
        return image, target
    
    def get_class_names(self) -> Dict[int, str]:
        """Get dictionary mapping class IDs to names."""
        return self.categories.copy()
    
    def get_num_classes(self) -> int:
        """Get total number of classes (including background)."""
        return len(self.categories) + 1  # +1 for background


def create_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 512)
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_root: Root directory containing train/, valid/, test/ folders
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        image_size: Target image size
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from utils import collate_fn
    
    # Create datasets for each split
    train_dataset = DetectionSegmentationDataset(
        root_dir=os.path.join(data_root, "train"),
        image_size=image_size,
        train=True
    )
    
    val_dataset = DetectionSegmentationDataset(
        root_dir=os.path.join(data_root, "valid"),
        image_size=image_size,
        train=False
    )
    
    test_dataset = DetectionSegmentationDataset(
        root_dir=os.path.join(data_root, "test"),
        image_size=image_size,
        train=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset with your data
    import matplotlib.pyplot as plt
    
    # Dataset path
    dataset_path = r"C:\Users\ali.donbaloglu\Desktop\Lens\datasetler\Lens_Kabin_Aralik_CNN"
    
    # Create train dataset
    dataset = DetectionSegmentationDataset(
        root_dir=os.path.join(dataset_path, "train"),
        image_size=(512, 512),
        train=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    print(f"Class names: {dataset.get_class_names()}")
    
    if len(dataset) > 0:
        image, target = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Boxes shape: {target['boxes'].shape}")
        print(f"Labels shape: {target['labels'].shape}")
        print(f"Segmentation shape: {target['segmentation'].shape}")
        print(f"Labels: {target['labels']}")
