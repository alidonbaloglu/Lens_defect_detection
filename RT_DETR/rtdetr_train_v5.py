"""
RT-DETR Optimized Training Script
==================================
%90+ mAP hedefi için optimize edilmiş, temiz kod
Kullanım: python rtdetr_optimized_train.py
pip install pycocotools
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import time

try:
    from ultralytics import RTDETR
    import torch
    import yaml
except ImportError:
    print("ERROR: pip install ultralytics torch pyyaml")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
#YOLO_ROOT = r"C:/Users/arge.ortak/Desktop/Lens/Kabin_merged_yolov2"
#DATASET_YAML = r"C:/Users/arge.ortak/Desktop/Lens/RT_DETR/kabin_dataset_yolo.yaml"
#OUTPUT_PROJECT = r"C:/Users/arge.ortak/Desktop/Lens/results/RTDETR"
#MODEL_DIR = r"C:/Users/arge.ortak/Desktop/Lens/Modeller"

YOLO_ROOT = r"C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Lens_Kabin_Aralik_Yolo"
DATASET_YAML = r"C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Lens_Kabin_Aralik_Yolo/data.yaml"
OUTPUT_PROJECT = r"C:/Users/ali.donbaloglu/Desktop/Lens/results/RTDETR"
MODEL_DIR = r"C:/Users/ali.donbaloglu/Desktop/Lens/Modeller"

# Model
MODEL_NAME = "rtdetr-x.pt"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "RTDETR_kabin_BEST.pt")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "RTDETR_kabin_LAST.pt")

# Training Parameters
RANDOM_SEED = 42
NUM_EPOCHS = 150
BATCH_SIZE = 8
IMAGE_SIZE = 1280
PATIENCE = 40

# Learning Rate
LEARNING_RATE = 0.00005
LR_FINAL = 0.005
WEIGHT_DECAY = 5e-5
WARMUP_EPOCHS = 5

# Thresholds
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25

# Stability
GRADIENT_CLIP = 5.0
USE_AMP = False

# Loss Weights
BOX_WEIGHT = 7.5
CLS_WEIGHT = 2.0
DFL_WEIGHT = 1.5

# Augmentation (Minimal - dataset already augmented)
USE_AUGMENTATION = True
AUG_HSVO = 0.01
AUG_HSVS = 0.1
AUG_HSVV = 0.1
AUG_DEGREES = 0.0
AUG_TRANSLATE = 0.05
AUG_SCALE = 0.1
AUG_SHEAR = 0.0
AUG_FLIPUD = 0.0
AUG_FLIPLR = 0.0
AUG_MOSAIC = 0.3
AUG_MIXUP = 0.0
AUG_COPY_PASTE = 0.1

# Global variables
epoch_history = []
epoch_logs_dir = None
epoch_start_time = None

# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ============================================================================
# mAP@0.25 CALCULATION
# ============================================================================

def calculate_map_at_iou_025(model, data_yaml: str) -> float:
    """
    Calculate mAP@0.25 using PyCocoTools
    Returns None if PyCocoTools not installed
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import yaml
    except ImportError:
        return None
    
    # Load dataset info
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    val_path = Path(data['path']) / data['val']
    
    # Get predictions
    results_list = model.predict(source=str(val_path), save=False, verbose=False, stream=True)
    
    # Convert to COCO format
    coco_gt = []
    coco_dt = []
    img_id = 0
    ann_id = 0
    
    for result in results_list:
        # Ground truth
        label_file = str(result.path).replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
        
        if Path(label_file).exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, x_c, y_c, w, h = map(float, parts[:5])
                        
                        # YOLO -> COCO
                        img_w, img_h = result.orig_shape[1], result.orig_shape[0]
                        x = (x_c - w/2) * img_w
                        y = (y_c - h/2) * img_h
                        w = w * img_w
                        h = h * img_h
                        
                        coco_gt.append({
                            'id': ann_id,
                            'image_id': img_id,
                            'category_id': int(cls),
                            'bbox': [x, y, w, h],
                            'area': w * h,
                            'iscrowd': 0
                        })
                        ann_id += 1
        
        # Predictions
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                coco_dt.append({
                    'image_id': img_id,
                    'category_id': int(cls),
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'score': float(score)
                })
        
        img_id += 1
    
    if not coco_gt or not coco_dt:
        return None
    
    # Create COCO objects
    coco_gt_obj = COCO()
    coco_gt_obj.dataset = {
        'images': [{'id': i} for i in range(img_id)],
        'annotations': coco_gt,
        'categories': [{'id': i} for i in range(len(data['names']))]
    }
    coco_gt_obj.createIndex()
    
    coco_dt_obj = coco_gt_obj.loadRes(coco_dt)
    
    # Evaluate at IoU=0.25
    coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, 'bbox')
    coco_eval.params.iouThrs = [0.25]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return float(coco_eval.stats[0])

# ============================================================================
# CALLBACKS
# ============================================================================

def on_train_epoch_start(trainer):
    global epoch_start_time
    epoch_start_time = time.time()

def on_fit_epoch_end(trainer):
    global epoch_history, epoch_logs_dir, epoch_start_time
    
    epoch = trainer.epoch + 1
    epoch_time = time.time() - epoch_start_time if epoch_start_time else 0
    
    metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
    loss_items = trainer.loss_items if hasattr(trainer, 'loss_items') else None
    train_loss = float(loss_items.mean()) if loss_items is not None else 0.0
    lr = trainer.optimizer.param_groups[0]['lr'] if trainer.optimizer else LEARNING_RATE
    
    epoch_info = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_map50': float(metrics.get('metrics/mAP50(B)', 0)),
        'val_map50_95': float(metrics.get('metrics/mAP50-95(B)', 0)),
        'val_precision': float(metrics.get('metrics/precision(B)', 0)),
        'val_recall': float(metrics.get('metrics/recall(B)', 0)),
        'learning_rate': lr,
        'epoch_time': epoch_time,
    }
    
    epoch_history.append(epoch_info)
    
    if epoch_logs_dir:
        epoch_json_path = os.path.join(epoch_logs_dir, f"epoch_{epoch:03d}.json")
        try:
            with open(epoch_json_path, "w", encoding="utf-8") as f:
                json.dump(epoch_info, f, indent=2, ensure_ascii=False)
        except:
            pass

# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_training_results(training_info: dict, output_dir: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    json_path = os.path.join(output_dir, f"training_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False, default=str)
    
    # TXT
    txt_path = os.path.join(output_dir, f"training_summary_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("="*100 + "\n")
        f.write("RT-DETR OPTIMIZED TRAINING RESULTS\n")
        f.write("="*100 + "\n\n")
        
        # Config
        f.write("CONFIGURATION:\n" + "-"*50 + "\n")
        config = training_info.get('config', {})
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Time
        f.write("TRAINING TIME:\n" + "-"*50 + "\n")
        time_info = training_info.get('training_time', {})
        f.write(f"Start: {time_info.get('start_time', '-')}\n")
        f.write(f"End: {time_info.get('end_time', '-')}\n")
        f.write(f"Duration: {time_info.get('total_duration', '-')}\n\n")
        
        # Epoch history table
        f.write("EPOCH HISTORY:\n" + "="*100 + "\n")
        epoch_hist = training_info.get('epoch_history', [])
        if epoch_hist:
            f.write(f"{'Epoch':>5}  {'Loss':>8}  {'mAP50':>8}  {'mAP95':>8}  {'Prec':>8}  {'Rec':>8}  {'LR':>10}  {'Time':>8}\n")
            f.write("-"*100 + "\n")
            for e in epoch_hist:
                f.write(f"{e.get('epoch', 0):>5}  "
                       f"{e.get('train_loss', 0):8.4f}  "
                       f"{e.get('val_map50', 0):8.4f}  "
                       f"{e.get('val_map50_95', 0):8.4f}  "
                       f"{e.get('val_precision', 0):8.4f}  "
                       f"{e.get('val_recall', 0):8.4f}  "
                       f"{e.get('learning_rate', 0):10.2e}  "
                       f"{e.get('epoch_time', 0):8.1f}\n")
        f.write("\n")
        
        # Best results
        f.write("="*100 + "\n")
        f.write("BEST MODEL RESULTS:\n")
        f.write("="*100 + "\n")
        best = training_info.get('best_results', {})
        f.write(f"Best Epoch: {best.get('epoch', '-')}\n")
        f.write(f"mAP@0.50: {best.get('map50', 0):.4f}\n")
        f.write(f"mAP@0.50-0.95: {best.get('map50_95', 0):.4f}\n")
        f.write(f"Precision: {best.get('precision', 0):.4f}\n")
        f.write(f"Recall: {best.get('recall', 0):.4f}\n\n")
        
        # Test results
        best_test = training_info.get('best_model_test', {})
        if best_test:
            f.write("TEST RESULTS (BEST MODEL):\n" + "-"*50 + "\n")
            f.write(f"mAP@0.50: {best_test.get('map50', 0):.4f}\n")
            f.write(f"mAP@0.50-0.95: {best_test.get('map50_95', 0):.4f}\n")
            f.write(f"Precision: {best_test.get('precision', 0):.4f}\n")
            f.write(f"Recall: {best_test.get('recall', 0):.4f}\n")
            if 'map25' in best_test:
                f.write(f"mAP@0.25: {best_test.get('map25', 0):.4f}\n")
            f.write("\n")
        
        f.write("="*100 + "\n")
        f.write(f"Epoch logs: {training_info.get('model_paths', {}).get('epoch_logs', '-')}\n")
        f.write("="*100 + "\n")
    
    return json_path, txt_path

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train():
    global epoch_history, epoch_logs_dir
    
    set_seed(RANDOM_SEED)
    ensure_dir(OUTPUT_PROJECT)
    ensure_dir(MODEL_DIR)
    
    epoch_history = []
    
    print("="*80)
    print("RT-DETR OPTIMIZED TRAINING (Target: 90%+ mAP)")
    print("="*80)
    
    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nDevice: {device} | GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print(f"\nDevice: {device}")
    
    print(f"Model: {MODEL_NAME} | Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | Size: {IMAGE_SIZE}")
    print(f"LR: {LEARNING_RATE} | Patience: {PATIENCE} | Augmentation: Minimal (dataset pre-augmented)")
    print("="*80 + "\n")
    
    # Setup directories
    OUTPUT_NAME = f"kabin_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(OUTPUT_PROJECT, OUTPUT_NAME)
    epoch_logs_dir = os.path.join(run_dir, "epoch_logs")
    ensure_dir(run_dir)
    ensure_dir(epoch_logs_dir)
    
    # Load model
    model = RTDETR(MODEL_NAME)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    # Start training
    print("Training started...")
    start_time = datetime.now()
    start_timestamp = time.time()
    
    # Augmentation params
    aug_params = {}
    if USE_AUGMENTATION:
        aug_params = {
            "hsv_h": AUG_HSVO,
            "hsv_s": AUG_HSVS,
            "hsv_v": AUG_HSVV,
            "degrees": AUG_DEGREES,
            "translate": AUG_TRANSLATE,
            "scale": AUG_SCALE,
            "shear": AUG_SHEAR,
            "flipud": AUG_FLIPUD,
            "fliplr": AUG_FLIPLR,
            "mosaic": AUG_MOSAIC,
            "mixup": AUG_MIXUP,
            "copy_paste": AUG_COPY_PASTE,
        }
    
    # Train
    results = model.train(
        data=DATASET_YAML,
        epochs=NUM_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        
        # Learning rate
        lr0=LEARNING_RATE,
        lrf=LR_FINAL,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=0.8,
        warmup_bias_lr=0.0,
        
        # Loss weights
        box=BOX_WEIGHT,
        cls=CLS_WEIGHT,
        dfl=DFL_WEIGHT,
        
        # Regularization
        label_smoothing=0.1,
        dropout=0.1,
        
        # Other
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        
        # Validation
        val=True,
        iou=IOU_THRESHOLD,
        conf=CONF_THRESHOLD,
        
        # System
        device=0 if device == "cuda" else "cpu",
        project=OUTPUT_PROJECT,
        name=OUTPUT_NAME,
        exist_ok=True,
        seed=RANDOM_SEED,
        verbose=True,
        plots=True,
        save=True,
        amp=USE_AMP,
        max_det=300,
        
        **aug_params
    )
    
    end_time = datetime.now()
    total_duration = (time.time() - start_timestamp) / 60
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Duration: {total_duration:.2f} minutes")
    print(f"Results: {run_dir}\n")
    
    # Copy models
    best_model_src = os.path.join(run_dir, "weights", "best.pt")
    last_model_src = os.path.join(run_dir, "weights", "last.pt")
    
    if os.path.exists(best_model_src):
        shutil.copy2(best_model_src, BEST_MODEL_PATH)
        print(f"Best model: {BEST_MODEL_PATH}")
    
    if os.path.exists(last_model_src):
        shutil.copy2(last_model_src, LAST_MODEL_PATH)
        print(f"Last model: {LAST_MODEL_PATH}")
    
    # Evaluate best model
    best_model_test = {}
    if os.path.exists(BEST_MODEL_PATH):
        print("\nEvaluating best model...")
        best_model = RTDETR(BEST_MODEL_PATH)
        best_val = best_model.val(data=DATASET_YAML, verbose=False)
        best_model_test = {
            'map50': float(best_val.box.map50),
            'map50_95': float(best_val.box.map),
            'precision': float(best_val.box.mp),
            'recall': float(best_val.box.mr),
        }
        print(f"mAP@0.50: {best_model_test['map50']:.4f}")
        print(f"mAP@0.50-0.95: {best_model_test['map50_95']:.4f}")
        print(f"Precision: {best_model_test['precision']:.4f}")
        print(f"Recall: {best_model_test['recall']:.4f}")
        
        # Calculate mAP@0.25 with PyCocoTools
        try:
            map25 = calculate_map_at_iou_025(best_model, DATASET_YAML)
            if map25 is not None:
                best_model_test['map25'] = map25
                print(f"mAP@0.25: {map25:.4f}")
        except Exception as e:
            print(f"WARNING: mAP@0.25 calculation failed (install pycocotools)")
    
    # Find best epoch
    best_epoch = 0
    best_map50 = 0
    for e in epoch_history:
        if e.get('val_map50', 0) > best_map50:
            best_map50 = e.get('val_map50', 0)
            best_epoch = e.get('epoch', 0)
    
    # Compile training info
    training_info = {
        "config": {
            "model": MODEL_NAME,
            "dataset_path": YOLO_ROOT,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "learning_rate": LEARNING_RATE,
            "lr_final": LR_FINAL,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "iou_threshold": IOU_THRESHOLD,
            "conf_threshold": CONF_THRESHOLD,
            "gradient_clip": GRADIENT_CLIP,
            "use_amp": USE_AMP,
            "warmup_epochs": WARMUP_EPOCHS,
            "box_weight": BOX_WEIGHT,
            "cls_weight": CLS_WEIGHT,
            "dfl_weight": DFL_WEIGHT,
            "augmentation": "Minimal (Mosaic+CopyPaste)",
            "device": device,
            "gpu_name": gpu_name,
            "seed": RANDOM_SEED,
        },
        "training_time": {
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": f"{total_duration:.2f} minutes",
        },
        "epoch_history": epoch_history,
        "best_results": {
            "epoch": best_epoch,
            "map50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        },
        "best_model_test": best_model_test,
        "model_paths": {
            "best": BEST_MODEL_PATH,
            "last": LAST_MODEL_PATH,
            "run_dir": run_dir,
            "epoch_logs": epoch_logs_dir,
        }
    }
    
    # Save results
    json_path, txt_path = save_training_results(training_info, run_dir)
    print(f"\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  TXT: {txt_path}")
    print("\n" + "="*80)
    
    return results

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================

def validate(model_path: str = None):
    if model_path is None:
        model_path = BEST_MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return None
    
    print(f"Loading model: {model_path}")
    model = RTDETR(model_path)
    
    print("Running validation...")
    results = model.val(data=DATASET_YAML)
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS:")
    print("="*80)
    print(f"mAP@0.50: {results.box.map50:.4f}")
    print(f"mAP@0.50-0.95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print("="*80)
    
    return results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--val":
        validate()
    else:
        train()