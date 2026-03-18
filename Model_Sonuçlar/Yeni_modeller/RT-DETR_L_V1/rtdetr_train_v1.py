"""
RT-DETR Eğitim Scripti - Improved Version
==========================================
Kabin yüzey hatası tespiti için RT-DETR modeli eğitimi.
Mask R-CNN kodundaki parametreler ve sonuç kaydetme yapısı ile uyumlu.

Kullanım:
    python rtdetr_train.py

Gereksinimler:
    pip install ultralytics>=8.0.0
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import time

# Ultralytics import
try:
    from ultralytics import RTDETR
    import torch
except ImportError:
    print("Ultralytics yüklü değil. Yüklemek için:")
    print("pip install ultralytics")
    exit(1)

# ----------------------------
# Konfigürasyon - Mask R-CNN ile Uyumlu
# ----------------------------
# Dataset yolları
YOLO_ROOT = r"C:/Users/arge.ortak/Desktop/Lens/Kabin_merged_yolo"
DATASET_YAML = r"C:/Users/arge.ortak/Desktop/Lens/RT_DETR/kabin_dataset_yolo.yaml"

# Çıktı klasörleri
OUTPUT_PROJECT = r"C:/Users/arge.ortak/Desktop/Lens/results/RTDETR"
OUTPUT_NAME = f"kabin_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
MODEL_DIR = r"C:/Users/arge.ortak/Desktop/Lens/Modeller"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "RTDETR_kabin_BEST.pt")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "RTDETR_kabin_LAST.pt")

# Model seçimi: rtdetr-l.pt veya rtdetr-x.pt
MODEL_NAME = "rtdetr-l.pt"

# ✅ IMPROVED HYPERPARAMETERS (Mask R-CNN ile uyumlu)
RANDOM_SEED = 42
NUM_EPOCHS = 100  # 75 → 100
BATCH_SIZE = 8  # 2 → 4
IMAGE_SIZE = 1000  # Kullanıcı tercihi
PATIENCE = 50     # 25 → 50 (Early stopping)
LEARNING_RATE = 0.0001  # Mask R-CNN ile aynı
WEIGHT_DECAY = 1e-4     # Mask R-CNN ile aynı
IOU_THRESHOLD = 0.25    # IoU threshold
CONF_THRESHOLD = 0.25   # Confidence threshold

# Augmentation ayarları
USE_AUGMENTATION = True
AUG_HSVO = 0.015  # HSV-Hue augmentation
AUG_HSVS = 0.2    # HSV-Saturation augmentation  
AUG_HSVV = 0.2    # HSV-Value augmentation
AUG_DEGREES = 0.0  # Rotation
AUG_TRANSLATE = 0.1
AUG_SCALE = 0.2
AUG_SHEAR = 0.0
AUG_FLIPUD = 0.0
AUG_FLIPLR = 0.5
AUG_MOSAIC = 0.5
AUG_MIXUP = 0.1


def set_seed(seed: int = 42):
    """Tekrarlanabilirlik için seed ayarla"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass


def ensure_dir(path: str):
    """Klasör yoksa oluştur"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_training_results(training_info: dict, output_dir: str):
    """Eğitim sonuçlarını JSON ve TXT olarak kaydet"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON kaydet
    json_path = os.path.join(output_dir, f"training_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False, default=str)
    
    # TXT kaydet
    txt_path = os.path.join(output_dir, f"training_summary_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RT-DETR EĞİTİM SONUÇLARI\n")
        f.write("=" * 80 + "\n\n")
        
        # Konfigürasyon
        f.write("EĞİTİM KONFİGÜRASYONU:\n")
        f.write("-" * 40 + "\n")
        config = training_info.get('config', {})
        f.write(f"Model: {config.get('model', '-')}\n")
        f.write(f"Veri Seti: {config.get('dataset_path', '-')}\n")
        f.write(f"Epoch Sayısı: {config.get('num_epochs', '-')}\n")
        f.write(f"Batch Size: {config.get('batch_size', '-')}\n")
        f.write(f"Image Size: {config.get('image_size', '-')}\n")
        f.write(f"Learning Rate: {config.get('learning_rate', '-')}\n")
        f.write(f"Weight Decay: {config.get('weight_decay', '-')}\n")
        f.write(f"Patience: {config.get('patience', '-')}\n")
        f.write(f"Sınıf Sayısı: {config.get('num_classes', '-')}\n")
        f.write(f"Augmentation: {'Aktif' if config.get('augmentation', False) else 'Kapalı'}\n")
        if config.get('device'):
            f.write(f"Cihaz: {config.get('device')}\n")
        if config.get('gpu_name'):
            f.write(f"GPU: {config.get('gpu_name')}\n")
        f.write("\n")
        
        # Eğitim süresi
        f.write("EĞİTİM SÜRESİ:\n")
        f.write("-" * 40 + "\n")
        time_info = training_info.get('training_time', {})
        f.write(f"Başlangıç: {time_info.get('start_time', '-')}\n")
        f.write(f"Bitiş: {time_info.get('end_time', '-')}\n")
        f.write(f"Toplam Süre: {time_info.get('total_duration', '-')}\n")
        f.write("\n")
        
        # En iyi model sonuçları
        f.write("EN İYİ MODEL SONUÇLARI:\n")
        f.write("=" * 80 + "\n")
        best = training_info.get('best_results', {})
        f.write(f"En İyi Epoch: {best.get('epoch', '-')}\n")
        f.write(f"mAP50: {best.get('map50', 0):.4f}\n")
        f.write(f"mAP50-95: {best.get('map50_95', 0):.4f}\n")
        f.write(f"Precision: {best.get('precision', 0):.4f}\n")
        f.write(f"Recall: {best.get('recall', 0):.4f}\n")
        f.write("\n")
        
        # Sınıf bazlı sonuçlar
        per_class = best.get('per_class', {})
        if per_class:
            f.write("SINIF BAZLI METRİKLER:\n")
            f.write("-" * 80 + "\n")
            for class_name, metrics in per_class.items():
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
                f.write(f"  Recall: {metrics.get('recall', 0):.4f}\n")
                f.write(f"  mAP50: {metrics.get('map50', 0):.4f}\n")
        
        # Test sonuçları
        test_results = training_info.get('test_results', {})
        if test_results:
            f.write("\n" + "=" * 80 + "\n")
            f.write("TEST SETİ SONUÇLARI:\n")
            f.write("=" * 80 + "\n")
            f.write(f"mAP50: {test_results.get('map50', 0):.4f}\n")
            f.write(f"mAP50-95: {test_results.get('map50_95', 0):.4f}\n")
            f.write(f"Precision: {test_results.get('precision', 0):.4f}\n")
            f.write(f"Recall: {test_results.get('recall', 0):.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("NOT: Bu sonuçlar otomatik olarak oluşturulmuştur.\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Eğitim sonuçları kaydedildi:")
    print(f"   JSON: {json_path}")
    print(f"   TXT:  {txt_path}")
    
    return json_path, txt_path


def train():
    """RT-DETR eğitimini başlat"""
    set_seed(RANDOM_SEED)
    ensure_dir(OUTPUT_PROJECT)
    ensure_dir(MODEL_DIR)
    
    print("=" * 80)
    print("RT-DETR EĞİTİM (IMPROVED VERSION)")
    print("=" * 80)
    
    # Cihaz bilgisi
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🖥️  Cihaz: {device}")
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 GPU Bellek: {gpu_memory:.2f} GB")
    else:
        print(f"\n🖥️  Cihaz: {device}")
    
    print(f"\n📁 Dataset: {YOLO_ROOT}")
    print(f"🔧 Model: {MODEL_NAME}")
    print(f"📊 Epochs: {NUM_EPOCHS}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🖼️  Image Size: {IMAGE_SIZE}")
    print(f"📈 Learning Rate: {LEARNING_RATE}")
    print(f"⏱️  Patience: {PATIENCE}")
    print(f"🎲 Augmentation: {'Aktif' if USE_AUGMENTATION else 'Kapalı'}")
    print("=" * 80)
    
    # Model yükle
    print(f"\n📥 Model yükleniyor: {MODEL_NAME}")
    model = RTDETR(MODEL_NAME)
    
    # Eğitimi başlat
    print("\n🚀 Eğitim başlıyor...")
    start_time = datetime.now()
    start_timestamp = time.time()
    
    # Augmentation parametreleri
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
        }
    
    results = model.train(
        data=DATASET_YAML,
        epochs=NUM_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        lr0=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=0 if device == "cuda" else "cpu",
        project=OUTPUT_PROJECT,
        name=OUTPUT_NAME,
        exist_ok=True,
        seed=RANDOM_SEED,
        verbose=True,
        plots=True,
        save=True,
        val=True,
        iou=IOU_THRESHOLD,
        conf=CONF_THRESHOLD,
        **aug_params
    )
    
    end_time = datetime.now()
    total_duration = (time.time() - start_timestamp) / 60
    
    print("\n" + "=" * 80)
    print("EĞİTİM TAMAMLANDI!")
    print("=" * 80)
    print(f"⏱️  Toplam süre: {total_duration:.2f} dakika")
    print(f"📂 Sonuçlar: {OUTPUT_PROJECT}/{OUTPUT_NAME}/")
    
    # Sonuçları topla
    run_dir = os.path.join(OUTPUT_PROJECT, OUTPUT_NAME)
    best_model_src = os.path.join(run_dir, "weights", "best.pt")
    last_model_src = os.path.join(run_dir, "weights", "last.pt")
    
    # En iyi modeli kopyala
    if os.path.exists(best_model_src):
        shutil.copy2(best_model_src, BEST_MODEL_PATH)
        print(f"✅ En iyi model: {BEST_MODEL_PATH}")
    
    # Son epoch modelini kopyala
    if os.path.exists(last_model_src):
        shutil.copy2(last_model_src, LAST_MODEL_PATH)
        print(f"✅ Son model: {LAST_MODEL_PATH}")
    
    # Training info hazırla
    training_info = {
        "config": {
            "model": MODEL_NAME,
            "dataset_path": YOLO_ROOT,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "num_classes": 2,
            "augmentation": USE_AUGMENTATION,
            "device": device,
            "gpu_name": gpu_name,
            "seed": RANDOM_SEED,
        },
        "training_time": {
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": f"{total_duration:.2f} dakika",
        },
        "best_results": {
            "map50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        },
        "model_paths": {
            "best": BEST_MODEL_PATH,
            "last": LAST_MODEL_PATH,
            "run_dir": run_dir,
        }
    }
    
    # Sonuçları kaydet
    save_training_results(training_info, run_dir)
    
    return results


def validate(model_path: str = None):
    """Eğitilmiş modeli validation set üzerinde değerlendir"""
    if model_path is None:
        model_path = BEST_MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"❌ Model bulunamadı: {model_path}")
        return None
    
    print(f"📥 Model yükleniyor: {model_path}")
    model = RTDETR(model_path)
    
    print("\n🔍 Validation başlıyor...")
    results = model.val(data=DATASET_YAML)
    
    print("\n" + "=" * 80)
    print("VALIDATION SONUÇLARI:")
    print("=" * 80)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--val":
        validate()
    else:
        train()
