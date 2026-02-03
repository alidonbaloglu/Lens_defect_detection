import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
from PIL import Image
import json
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import albumentations as A

# ============================================
# CONFIGURATION
# ============================================
COCO_ROOT = r"C:/Users/ali.donbaloglu/Desktop/Lens/cnn_kabin_dataset_v2"
TRAIN_SPLIT = "train"
VAL_SPLIT = "valid"
OUTPUT_DIR = "results/enhanced_maskrcnn"
MODEL_PATH = "models/enhanced_maskrcnn.pt"
TRAINING_LOG_PATH = "results/training_log.json"
CONFIG_PATH = "results/training_config.json"

NUM_CLASSES = 3  # background + 2 sınıf
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# MULTI-MODAL PREPROCESSOR
# ============================================
class MultiModalPreprocessor:
    """RGB, Edge ve Gradient bilgilerini çıkarır"""
    
    def __call__(self, image):
        """
        Args:
            image: PIL Image veya numpy array
        Returns:
            dict: {'rgb': tensor, 'edge': tensor, 'gradient': tensor}
        """
        # PIL'den numpy'ye çevir
        if isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        # RGB
        rgb = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        
        # Edge detection (Canny)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_tensor = torch.from_numpy(edges).unsqueeze(0).float() / 255.0
        
        # Gradient (Sobel)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        gradient = (gradient / gradient.max() * 255).astype(np.uint8)
        gradient_tensor = torch.from_numpy(gradient).unsqueeze(0).float() / 255.0
        
        return {
            'rgb': rgb,
            'edge': edge_tensor,
            'gradient': gradient_tensor
        }

# ============================================
# MULTI-MODAL BACKBONE
# ============================================
class MultiModalBackbone(torch.nn.Module):
    """RGB + Edge + Gradient girişlerini işleyen backbone"""
    
    def __init__(self, backbone_name='resnet50', trainable_layers=3):
        super().__init__()
        
        # Ana backbone (RGB için)
        weights = torchvision.models.ResNet50_Weights.DEFAULT if backbone_name == 'resnet50' else torchvision.models.ResNet101_Weights.DEFAULT
        backbone = torchvision.models.resnet50(weights=weights) if backbone_name == 'resnet50' else torchvision.models.resnet101(weights=weights)
        
        # RGB için backbone
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Edge ve Gradient için ek conv katmanları
        self.edge_conv = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gradient_conv = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Feature fusion
        self.fusion_conv = torch.nn.Conv2d(64*3, 64, kernel_size=1)
        
        self.out_channels = 2048 if 'resnet50' in backbone_name or 'resnet101' in backbone_name else 512
    
    def forward(self, x):
        # x bir dict: {'rgb': tensor, 'edge': tensor, 'gradient': tensor}
        if isinstance(x, dict):
            rgb = x['rgb']
            edge = x['edge']
            gradient = x['gradient']
            
            # Her modality için feature extraction
            rgb_feat = self.conv1(rgb)
            edge_feat = self.edge_conv(edge)
            gradient_feat = self.gradient_conv(gradient)
            
            # Fusion
            fused = torch.cat([rgb_feat, edge_feat, gradient_feat], dim=1)
            x = self.fusion_conv(fused)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            # Normal RGB input
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

# ============================================
# REFINED MASK PREDICTOR
# ============================================
class RefinedMaskRCNNPredictor(torch.nn.Module):
    """Contour refinement ile geliştirilmiş mask predictor"""
    
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__()
        
        self.conv5_mask = torch.nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        
        # Contour refinement branch
        self.contour_conv1 = torch.nn.Conv2d(dim_reduced, 128, 3, padding=1)
        self.contour_conv2 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.contour_conv3 = torch.nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        x = self.conv5_mask(x)
        x = self.relu(x)
        
        # Main mask prediction
        mask_logits = self.mask_fcn_logits(x)
        
        # Contour refinement
        contour = self.relu(self.contour_conv1(x))
        contour = self.relu(self.contour_conv2(contour))
        contour_logits = self.contour_conv3(contour)
        
        # Combine
        refined_logits = mask_logits + 0.3 * contour_logits
        
        return refined_logits

# ============================================
# BOUNDARY-AWARE LOSS
# ============================================
class BoundaryAwareMaskLoss(torch.nn.Module):
    """Sınır bölgelerine ağırlık veren loss"""
    
    def __init__(self, boundary_weight=2.5):
        super().__init__()
        self.boundary_weight = boundary_weight
    
    def forward(self, pred_masks, target_masks):
        # Standard BCE loss
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_masks, target_masks, reduction='none'
        )
        
        # Boundary detection
        target_np = target_masks.cpu().detach().numpy()
        boundary_weights = torch.ones_like(target_masks)
        
        for i in range(target_np.shape[0]):
            for j in range(target_np.shape[1]):
                mask = target_np[i, j].astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                boundary_mask = np.zeros_like(mask)
                cv2.drawContours(boundary_mask, contours, -1, 1, thickness=3)
                boundary_weights[i, j] = torch.from_numpy(boundary_mask).to(target_masks.device) * self.boundary_weight + 1.0
        
        # Weighted loss
        weighted_loss = bce_loss * boundary_weights
        return weighted_loss.mean()

# ============================================
# DATASET
# ============================================
class MultiModalCOCODataset(Dataset):
    """Multi-modal COCO dataset with augmentation"""
    
    def __init__(self, json_path, img_root, use_augmentation=False, augmentation_params=None):
        self.coco = COCO(json_path)
        self.img_root = img_root
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.preprocessor = MultiModalPreprocessor()
        self.use_augmentation = use_augmentation
        
        # Get number of classes
        cat_ids = self.coco.getCatIds()
        self.num_classes = len(cat_ids) + 1  # +1 for background
        
        # Augmentation
        if use_augmentation and augmentation_params:
            self.transform = A.Compose([
                A.HorizontalFlip(p=augmentation_params.get('reflection_prob', 0.5)),
                A.RandomBrightnessContrast(p=augmentation_params.get('contrast_prob', 0.5)),
                A.GaussianBlur(p=augmentation_params.get('distortion_prob', 0.3)),
            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
        else:
            self.transform = None
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        masks = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])
            
            # Mask
            mask = self.coco.annToMask(ann)
            masks.append(mask)
        
        # Augmentation
        if self.transform and len(boxes) > 0:
            transformed = self.transform(
                image=img_np,
                bboxes=boxes,
                labels=labels,
                masks=masks
            )
            img_np = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
            masks = transformed['masks']
        
        # Convert to PIL for preprocessing
        img = Image.fromarray(img_np)
        
        # Multi-modal preprocessing
        modalities = self.preprocessor(img)
        
        # Prepare target
        target = {}
        if len(boxes) > 0:
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
            target['masks'] = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['masks'] = torch.zeros((0, img_np.shape[0], img_np.shape[1]), dtype=torch.uint8)
        
        target['image_id'] = torch.tensor([img_id])
        target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
        target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        return modalities, target
    
    def __len__(self):
        return len(self.ids)

def collate_fn_multimodal(batch):
    """Custom collate function for multi-modal data"""
    modalities_list = []
    targets_list = []
    
    for modalities, target in batch:
        modalities_list.append(modalities)
        targets_list.append(target)
    
    # Stack RGB
    rgb_batch = torch.stack([m['rgb'] for m in modalities_list])
    edge_batch = torch.stack([m['edge'] for m in modalities_list])
    gradient_batch = torch.stack([m['gradient'] for m in modalities_list])
    
    modalities_batch = {
        'rgb': rgb_batch,
        'edge': edge_batch,
        'gradient': gradient_batch
    }
    
    return modalities_batch, targets_list

# ============================================
# MODEL CONSTRUCTION
# ============================================
def create_enhanced_maskrcnn(num_classes):
    """Tüm iyileştirmeleri içeren Mask R-CNN"""
    
    # Standart Mask R-CNN (basitleştirilmiş)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Box predictor'ı güncelle
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Mask predictor'ı güncelle
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = RefinedMaskRCNNPredictor(
        in_features_mask, 
        dim_reduced=256, 
        num_classes=num_classes
    )
    
    return model

# ============================================
# HELPER FUNCTIONS
# ============================================
def _find_split_paths(split_root):
    """Find images folder and annotations JSON"""
    images_root = os.path.join(split_root, "images")
    json_path = os.path.join(split_root, "_annotations.coco.json")
    
    if not os.path.exists(images_root):
        images_root = split_root
    
    return images_root, json_path

def evaluate_pixel_f1(model, dataloader, device):
    """Basit F1 değerlendirmesi"""
    model.eval()
    total_f1 = 0
    count = 0
    
    with torch.no_grad():
        for modalities, targets in dataloader:
            # RGB'yi kullan (basitleştirilmiş)
            images = modalities['rgb'].to(device)
            
            predictions = model(images)
            
            for pred, target in zip(predictions, targets):
                if len(pred['masks']) > 0 and len(target['masks']) > 0:
                    pred_mask = (pred['masks'][0, 0] > 0.5).cpu().numpy()
                    true_mask = target['masks'][0].cpu().numpy()
                    
                    intersection = np.logical_and(pred_mask, true_mask).sum()
                    union = np.logical_or(pred_mask, true_mask).sum()
                    
                    if union > 0:
                        iou = intersection / union
                        total_f1 += iou
                        count += 1
    
    return total_f1 / count if count > 0 else 0.0

# ============================================
# DATASET PREPARATION
# ============================================
def prepare_datasets():
    """Augmentasyonlu dataset'leri hazırla"""
    
    train_root = os.path.join(COCO_ROOT, TRAIN_SPLIT)
    val_root = os.path.join(COCO_ROOT, VAL_SPLIT)
    
    train_images_root, train_json = _find_split_paths(train_root)
    val_images_root, val_json = _find_split_paths(val_root)
    
    # Augmentation parametreleri
    aug_params = {
        'reflection_prob': 0.5,
        'transparency_prob': 0.5,
        'contrast_prob': 0.5,
        'distortion_prob': 0.3
    }
    
    # Training dataset (augmentation ile)
    train_dataset = MultiModalCOCODataset(
        train_json, train_images_root,
        use_augmentation=True,
        augmentation_params=aug_params
    )
    
    # Validation dataset (augmentation olmadan)
    val_dataset = MultiModalCOCODataset(
        val_json, val_images_root,
        use_augmentation=False
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_multimodal,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_multimodal,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.num_classes

# ============================================
# LOGGING FUNCTIONS
# ============================================
def save_training_config():
    """Eğitim parametrelerini JSON olarak kaydet"""
    config = {
        "model": {
            "name": "Enhanced Mask R-CNN",
            "backbone": "ResNet50-FPN",
            "num_classes": NUM_CLASSES
        },
        "dataset": {
            "coco_root": COCO_ROOT,
            "train_split": TRAIN_SPLIT,
            "val_split": VAL_SPLIT
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer": "AdamW",
            "weight_decay": 5e-5,
            "lr_scheduler": "ReduceLROnPlateau",
            "patience": 10,
            "device": str(DEVICE)
        },
        "augmentation": {
            "reflection_prob": 0.5,
            "transparency_prob": 0.5,
            "contrast_prob": 0.5,
            "distortion_prob": 0.3
        },
        "timestamp": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    }
    
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Training config saved to: {CONFIG_PATH}")
    return config

def save_training_log(training_history):
    """Eğitim geçmişini JSON olarak kaydet"""
    os.makedirs(os.path.dirname(TRAINING_LOG_PATH), exist_ok=True)
    with open(TRAINING_LOG_PATH, 'w') as f:
        json.dump(training_history, f, indent=4)
    
    print(f"Training log saved to: {TRAINING_LOG_PATH}")

# ============================================
# TRAINING
# ============================================
def train_enhanced_model():
    """Geliştirilmiş modeli eğit"""
    
    # Save training config
    print("Saving training configuration...")
    config = save_training_config()
    
    print("Preparing datasets...")
    train_loader, val_loader, num_classes = prepare_datasets()
    
    print("Creating enhanced model...")
    model = create_enhanced_maskrcnn(num_classes)
    model.to(DEVICE)
    
    print(f"Model loaded on {DEVICE}")
    print(f"Number of classes: {num_classes}")
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=5e-5)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Training history
    training_history = {
        "config": config,
        "epochs": [],
        "best_model": {
            "epoch": 0,
            "f1_score": 0.0,
            "model_path": MODEL_PATH
        }
    }
    
    # Training loop
    best_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 80)
        
        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": 0.0,
            "val_f1": 0.0,
            "learning_rate": 0.0,
            "improved": False
        }
        
        # Train
        model.train()
        epoch_loss = 0
        batch_losses = []
        
        for batch_idx, (modalities, targets) in enumerate(train_loader):
            # Sadece RGB'yi kullan (basitleştirilmiş)
            images = modalities['rgb'].to(DEVICE)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            # Forward
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            batch_loss = losses.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {batch_loss:.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        epoch_data["train_loss"] = avg_loss
        epoch_data["batch_losses"] = batch_losses
        print(f"  Average Training Loss: {avg_loss:.4f}")
        
        # Validation
        print("  Evaluating on validation set...")
        val_f1 = evaluate_pixel_f1(model, val_loader, DEVICE)
        epoch_data["val_f1"] = val_f1
        print(f"  Validation F1: {val_f1:.4f}")
        
        # Learning rate scheduling
        lr_scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_data["learning_rate"] = current_lr
        print(f"  Current Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            epoch_data["improved"] = True
            
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'num_classes': num_classes
            }, MODEL_PATH)
            
            # Update best model info
            training_history["best_model"]["epoch"] = epoch + 1
            training_history["best_model"]["f1_score"] = float(best_f1)
            
            print(f"  ✓ Best model saved! (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        # Add epoch data to history
        training_history["epochs"].append(epoch_data)
        
        # Save training log after each epoch
        save_training_log(training_history)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            training_history["early_stopped"] = True
            training_history["early_stop_epoch"] = epoch + 1
            break
    
    # Final summary
    training_history["completed"] = True
    training_history["total_epochs_trained"] = epoch + 1
    save_training_log(training_history)
    
    print("\n" + "=" * 80)
    print(f"Training completed!")
    print(f"Best Validation F1: {best_f1:.4f}")
    print(f"Best Epoch: {training_history['best_model']['epoch']}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training log saved to: {TRAINING_LOG_PATH}")
    print(f"Config saved to: {CONFIG_PATH}")
    
    return model, best_f1

# ============================================
# INFERENCE & VISUALIZATION
# ============================================
@torch.no_grad()
def inference_with_visualization(model, image_path, output_path):
    """Görüntü üzerinde inference ve görselleştirme"""
    model.eval()
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    
    # Preprocess (multi-modal)
    preprocessor = MultiModalPreprocessor()
    modalities = preprocessor(img)
    
    # Sadece RGB kullan (basitleştirilmiş)
    img_tensor = modalities['rgb'].unsqueeze(0).to(DEVICE)
    
    # Inference
    predictions = model(img_tensor)
    pred = predictions[0]
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Masks
    if len(pred['masks']) > 0:
        masks = pred['masks'][:, 0].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        # Combine masks
        combined_mask = np.zeros(masks[0].shape, dtype=np.float32)
        for mask, score in zip(masks, scores):
            if score > 0.5:
                combined_mask = np.maximum(combined_mask, mask)
        
        axes[1].imshow(combined_mask, cmap='jet')
        axes[1].set_title('Predicted Masks')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(img)
        axes[2].imshow(combined_mask, alpha=0.5, cmap='jet')
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    # Training
    model, best_f1 = train_enhanced_model()
    
    # Test inference
    test_image = "path/to/test/image.jpg"
    if os.path.exists(test_image):
        inference_with_visualization(
            model, 
            test_image, 
            "results/prediction_visualization.png"
        )