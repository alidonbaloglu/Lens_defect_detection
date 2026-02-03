"""
YOLO11 Eğitim Scripti - Based on RT-DETR Training Code
=======================================================
Kabin yüzey hatası tespiti için YOLO11 modeli eğitimi.
Mask R-CNN kodundaki parametreler ve sonuç kaydetme yapısı ile uyumlu.
Epoch bazlı metrik kaydetme ve mAP@0.25 desteği eklenmiştir.

Kullanım:
    python yolo11_train.py

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
    from ultralytics import YOLO  # ✅ YOLO11 için değiştirildi
    from ultralytics.utils.callbacks.base import add_integration_callbacks
    import torch
except ImportError:
    print("Ultralytics yüklü değil. Yüklemek için:")
    print("pip install ultralytics")
    exit(1)

# ----------------------------
# Konfigürasyon - Mask R-CNN ile Uyumlu
# ----------------------------
# Dataset yolları
YOLO_ROOT = r"C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Lens_Kabin_Aralik_Yolo"
DATASET_YAML = r"C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Lens_Kabin_Aralik_Yolo/data.yaml"

# Çıktı klasörleri - YOLO11 için güncellenmiş
OUTPUT_PROJECT = r"C:/Users/ali.donbaloglu/Desktop/Lens/results/YOLO11"
OUTPUT_NAME = f"Kabin_aralik_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
MODEL_DIR = r"C:/Users/ali.donbaloglu/Desktop/Lens/Modeller"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "YOLO11L_kabin_aralik_BEST.pt")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "YOLO11L_kabin_aralik_LAST.pt")

# MODEL SEÇİMİ - YOLO11 Large
MODEL_NAME = "yolo11l.pt"  # ✅ YOLO11 Large modeli

# ✅ İYİLEŞTİRİLMİŞ HİPERPARAMETRELER (YOLO11 için optimize edilmiş)
RANDOM_SEED = 42
NUM_EPOCHS = 150
BATCH_SIZE = 1  # ✅ YOLO daha hafif, batch size artırılabilir
IMAGE_SIZE = 1000  # ✅ YOLO için standart boyut (1000 yerine 640)
PATIENCE = 30
LEARNING_RATE = 0.01  # ✅ YOLO için standart LR
LR_FINAL = 0.01
WEIGHT_DECAY = 5e-4  # ✅ YOLO için standart weight decay
IOU_THRESHOLD = 0.25
CONF_THRESHOLD = 0.001

# ✅ YENİ EKLEMELER
GRADIENT_CLIP = 10.0
USE_AMP = True  # ✅ YOLO ile AMP genellikle stabil çalışır
WARMUP_EPOCHS = 3
WARMUP_LR = 0.0
CLS_WEIGHT = 0.5  # Classification loss weight

# ✅ İYİLEŞTİRİLMİŞ AUGMENTATION (YOLO için optimize edilmiş)
USE_AUGMENTATION = True  # ✅ YOLO augmentation ile iyi çalışır
AUG_HSVO = 0.015
AUG_HSVS = 0.7
AUG_HSVV = 0.4
AUG_DEGREES = 0.0
AUG_TRANSLATE = 0.1
AUG_SCALE = 0.5
AUG_SHEAR = 0.0
AUG_FLIPUD = 0.0
AUG_FLIPLR = 0.5
AUG_MOSAIC = 1.0  # ✅ YOLO Mosaic aktif
AUG_MIXUP = 0.0
AUG_COPY_PASTE = 0.0

# Global değişkenler - epoch logging için
epoch_history = []
epoch_logs_dir = None
epoch_start_time = None


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


# ----------------------------
# ✅ EPOCH LOGGING CALLBACKS
# ----------------------------
def on_train_epoch_start(trainer):
    """Her epoch başında çağrılır"""
    global epoch_start_time
    epoch_start_time = time.time()


def on_fit_epoch_end(trainer):
    """Her epoch sonunda çağrılır - metrikleri kaydet"""
    global epoch_history, epoch_logs_dir, epoch_start_time
    
    epoch = trainer.epoch + 1
    epoch_time = time.time() - epoch_start_time if epoch_start_time else 0
    
    # Metrikleri topla
    metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
    
    # Loss değerlerini al
    loss_items = trainer.loss_items if hasattr(trainer, 'loss_items') else None
    train_loss = float(loss_items.mean()) if loss_items is not None else 0.0
    
    # Learning rate
    lr = trainer.optimizer.param_groups[0]['lr'] if trainer.optimizer else LEARNING_RATE
    
    # Epoch bilgisi
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
    
    # Per-epoch JSON dosyası kaydet
    if epoch_logs_dir:
        epoch_json_path = os.path.join(epoch_logs_dir, f"epoch_{epoch:03d}.json")
        try:
            with open(epoch_json_path, "w", encoding="utf-8") as f:
                json.dump(epoch_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Epoch {epoch} JSON kaydı başarısız: {e}")
    
    # Konsola özet yazdır
    print(f"\n📊 Epoch {epoch} Özet:")
    print(f"   📉 Train Loss: {train_loss:.4f}")
    print(f"   🎯 Val mAP50: {epoch_info['val_map50']:.4f}")
    print(f"   📈 Val mAP50-95: {epoch_info['val_map50_95']:.4f}")
    print(f"   📈 Val Precision: {epoch_info['val_precision']:.4f}")
    print(f"   📈 Val Recall: {epoch_info['val_recall']:.4f}")
    print(f"   🔧 LR: {lr:.2e}")
    print(f"   ⏱️  Süre: {epoch_time:.2f}s")


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
        f.write("=" * 100 + "\n")
        f.write("YOLO11 EĞİTİM SONUÇLARI (EPOCH LOGGING ENABLED)\n")
        f.write("=" * 100 + "\n\n")
        
        # Konfigürasyon
        f.write("EĞİTİM KONFİGÜRASYONU:\n")
        f.write("-" * 50 + "\n")
        config = training_info.get('config', {})
        f.write(f"Model: {config.get('model', '-')}\n")
        f.write(f"Veri Seti: {config.get('dataset_path', '-')}\n")
        f.write(f"Epoch Sayısı: {config.get('num_epochs', '-')}\n")
        f.write(f"Batch Size: {config.get('batch_size', '-')}\n")
        f.write(f"Image Size: {config.get('image_size', '-')}\n")
        f.write(f"Learning Rate: {config.get('learning_rate', '-')}\n")
        f.write(f"LR Final (lrf): {config.get('lr_final', '-')}\n")
        f.write(f"Weight Decay: {config.get('weight_decay', '-')}\n")
        f.write(f"Patience: {config.get('patience', '-')}\n")
        f.write(f"IoU Threshold: {config.get('iou_threshold', '-')}\n")
        f.write(f"Conf Threshold: {config.get('conf_threshold', '-')}\n")
        f.write(f"Sınıf Sayısı: {config.get('num_classes', '-')}\n")
        f.write(f"Augmentation: {'Aktif' if config.get('augmentation', False) else 'Kapalı'}\n")
        if config.get('device'):
            f.write(f"Cihaz: {config.get('device')}\n")
        if config.get('gpu_name'):
            f.write(f"GPU: {config.get('gpu_name')}\n")
        f.write("\n")
        
        # Eğitim süresi
        f.write("EĞİTİM SÜRESİ:\n")
        f.write("-" * 50 + "\n")
        time_info = training_info.get('training_time', {})
        f.write(f"Başlangıç: {time_info.get('start_time', '-')}\n")
        f.write(f"Bitiş: {time_info.get('end_time', '-')}\n")
        f.write(f"Toplam Süre: {time_info.get('total_duration', '-')}\n")
        f.write("\n")
        
        # ✅ EPOCH GEÇMİŞİ TABLOSU
        f.write("EPOCH GEÇMİŞİ:\n")
        f.write("=" * 100 + "\n")
        epoch_hist = training_info.get('epoch_history', [])
        if epoch_hist:
            # Tablo başlığı
            f.write(f"{'Epoch':>5}  {'TrainLoss':>10}  {'mAP50':>8}  {'mAP50-95':>10}  {'Precision':>10}  {'Recall':>8}  {'LR':>12}  {'Time(s)':>8}\n")
            f.write("-" * 100 + "\n")
            for e in epoch_hist:
                ep = e.get('epoch', '-')
                tl = e.get('train_loss', 0.0)
                m50 = e.get('val_map50', 0.0)
                m50_95 = e.get('val_map50_95', 0.0)
                prec = e.get('val_precision', 0.0)
                rec = e.get('val_recall', 0.0)
                lr = e.get('learning_rate', 0.0)
                et = e.get('epoch_time', 0.0)
                f.write(f"{ep:>5}  {tl:10.4f}  {m50:8.4f}  {m50_95:10.4f}  {prec:10.4f}  {rec:8.4f}  {lr:12.2e}  {et:8.1f}\n")
        else:
            f.write("(Epoch history not available)\n")
        f.write("\n")
        
        # En iyi model sonuçları
        f.write("EN İYİ MODEL SONUÇLARI:\n")
        f.write("=" * 100 + "\n")
        best = training_info.get('best_results', {})
        f.write(f"En İyi Epoch: {best.get('epoch', '-')}\n")
        f.write(f"mAP50: {best.get('map50', 0):.4f}\n")
        f.write(f"mAP50-95: {best.get('map50_95', 0):.4f}\n")
        f.write(f"Precision: {best.get('precision', 0):.4f}\n")
        f.write(f"Recall: {best.get('recall', 0):.4f}\n")
        f.write("\n")
        
        # ✅ mAP@0.25 SONUÇLARI (Özel bölüm)
        map_025_results = training_info.get('map_025_results', {})
        if map_025_results and map_025_results.get('best'):
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"mAP@{IOU_THRESHOLD} SONUÇLARI (Özel IoU Threshold):\n")
            f.write("=" * 100 + "\n")
            
            best_map025 = map_025_results.get('best', {})
            last_map025 = map_025_results.get('last', {})
            
            f.write("EN İYİ MODEL (BEST):\n")
            f.write(f"  mAP@{IOU_THRESHOLD}: {best_map025.get(f'map@{IOU_THRESHOLD}', 0):.4f}\n")
            f.write(f"  mAP50: {best_map025.get('map50', 0):.4f}\n")
            f.write(f"  mAP50-95: {best_map025.get('map50_95', 0):.4f}\n")
            f.write(f"  Precision: {best_map025.get('precision', 0):.4f}\n")
            f.write(f"  Recall: {best_map025.get('recall', 0):.4f}\n")
            
            if last_map025 and last_map025.get('map50', 0) > 0:
                f.write("\nSON MODEL (LAST):\n")
                f.write(f"  mAP@{IOU_THRESHOLD}: {last_map025.get(f'map@{IOU_THRESHOLD}', 0):.4f}\n")
                f.write(f"  mAP50: {last_map025.get('map50', 0):.4f}\n")
                f.write(f"  mAP50-95: {last_map025.get('map50_95', 0):.4f}\n")
                f.write(f"  Precision: {last_map025.get('precision', 0):.4f}\n")
                f.write(f"  Recall: {last_map025.get('recall', 0):.4f}\n")
            
            f.write("\n")
        
        # Sınıf bazlı sonuçlar
        per_class = best.get('per_class', {})
        if per_class:
            f.write("SINIF BAZLI METRİKLER:\n")
            f.write("-" * 100 + "\n")
            for class_name, metrics in per_class.items():
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
                f.write(f"  Recall: {metrics.get('recall', 0):.4f}\n")
                f.write(f"  mAP50: {metrics.get('map50', 0):.4f}\n")
        
        # ✅ EN İYİ MODEL TEST SONUÇLARI
        best_test = training_info.get('best_model_test', {})
        if best_test:
            f.write("\n" + "=" * 100 + "\n")
            f.write("EN İYİ MODEL (BEST) TEST SONUÇLARI:\n")
            f.write("=" * 100 + "\n")
            f.write(f"mAP50: {best_test.get('map50', 0):.4f}\n")
            f.write(f"mAP50-95: {best_test.get('map50_95', 0):.4f}\n")
            f.write(f"Precision: {best_test.get('precision', 0):.4f}\n")
            f.write(f"Recall: {best_test.get('recall', 0):.4f}\n")
            if 'map_025' in best_test:
                f.write(f"mAP@{IOU_THRESHOLD}: {best_test.get('map_025', 0):.4f}\n")
        
        # ✅ SON MODEL TEST SONUÇLARI
        last_test = training_info.get('last_model_test', {})
        if last_test:
            f.write("\n" + "=" * 100 + "\n")
            f.write("SON MODEL (LAST) TEST SONUÇLARI:\n")
            f.write("=" * 100 + "\n")
            f.write(f"mAP50: {last_test.get('map50', 0):.4f}\n")
            f.write(f"mAP50-95: {last_test.get('map50_95', 0):.4f}\n")
            f.write(f"Precision: {last_test.get('precision', 0):.4f}\n")
            f.write(f"Recall: {last_test.get('recall', 0):.4f}\n")
            if 'map_025' in last_test:
                f.write(f"mAP@{IOU_THRESHOLD}: {last_test.get('map_025', 0):.4f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("NOT: Bu sonuçlar otomatik olarak oluşturulmuştur.\n")
        f.write(f"Epoch logları: {training_info.get('model_paths', {}).get('epoch_logs', '-')}\n")
        f.write("=" * 100 + "\n")
    
    print(f"✅ Eğitim sonuçları kaydedildi:")
    print(f"   JSON: {json_path}")
    print(f"   TXT:  {txt_path}")
    
    return json_path, txt_path


def evaluate_with_custom_iou(model, data_yaml: str, iou_threshold: float = 0.25):
    """Özel IoU threshold ile validation yap ve mAP hesapla"""
    print(f"\n🔍 mAP@{iou_threshold} hesaplanıyor...")
    try:
        results = model.val(
            data=data_yaml,
            iou=iou_threshold,
            conf=CONF_THRESHOLD,
            verbose=False,
            plots=False
        )
        
        # Güvenli attribute erişimi
        map50 = float(getattr(results.box, 'map50', 0.0))
        map50_95 = float(getattr(results.box, 'map', 0.0))
        precision = float(getattr(results.box, 'mp', 0.0))
        recall = float(getattr(results.box, 'mr', 0.0))
        
        # Sınıf bazlı metrikler (varsa)
        per_class_map = None
        if hasattr(results.box, 'maps') and results.box.maps is not None:
            per_class_map = results.box.maps.tolist()
        
        metrics = {
            f'map@{iou_threshold}': map50,
            'map50': map50,
            'map50_95': map50_95,
            'precision': precision,
            'recall': recall,
            'per_class_map': per_class_map
        }
        
        print(f"   ✅ mAP@{iou_threshold}: {map50:.4f}")
        print(f"      mAP50: {map50:.4f}")
        print(f"      mAP50-95: {map50_95:.4f}")
        print(f"      Precision: {precision:.4f}")
        print(f"      Recall: {recall:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"⚠️ mAP@{iou_threshold} hesaplanamadı: {e}")
        import traceback
        traceback.print_exc()
        return {
            f'map@{iou_threshold}': 0.0,
            'map50': 0.0,
            'map50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'per_class_map': None
        }


def train():
    """YOLO11 eğitimini başlat"""
    global epoch_history, epoch_logs_dir
    
    set_seed(RANDOM_SEED)
    ensure_dir(OUTPUT_PROJECT)
    ensure_dir(MODEL_DIR)
    
    # Epoch history'yi sıfırla
    epoch_history = []
    
    print("=" * 80)
    print("YOLO11 EĞİTİM (EPOCH LOGGING ENABLED)")
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
    print(f"📈 Learning Rate: {LEARNING_RATE} → {LEARNING_RATE * LR_FINAL} (cosine)")
    print(f"⏱️  Patience: {PATIENCE}")
    print(f"🎯 IoU Threshold: {IOU_THRESHOLD}")
    print(f"🎲 Augmentation: {'Aktif' if USE_AUGMENTATION else 'Kapalı'}")
    print("=" * 80)
    
    # Çıktı klasörünü hazırla
    run_dir = os.path.join(OUTPUT_PROJECT, OUTPUT_NAME)
    epoch_logs_dir = os.path.join(run_dir, "epoch_logs")
    ensure_dir(run_dir)
    ensure_dir(epoch_logs_dir)
    print(f"\n📂 Epoch logları: {epoch_logs_dir}")
    
    # Model yükle - YOLO11
    print(f"\n📥 Model yükleniyor: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)  # ✅ YOLO11 için değiştirildi
    
    # ✅ Callback'leri ekle
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    # Eğitimi başlat
    print("\n🚀 Eğitim başlıyor...")
    start_time = datetime.now()
    start_timestamp = time.time()
    
    # ✅ İyileştirilmiş Augmentation parametreleri
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
    
    results = model.train(
        data=DATASET_YAML,
        epochs=NUM_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        lr0=LEARNING_RATE,
        lrf=LR_FINAL,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=0.8,
        warmup_bias_lr=WARMUP_LR,
        box=7.5,  # ✅ YOLO için standart box loss gain
        cls=CLS_WEIGHT,  # Class loss gain
        dfl=1.5,  # ✅ YOLO için standart DFL loss gain
        label_smoothing=0.0,
        nbs=64,  # Nominal batch size
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        device=0 if device == "cuda" else "cpu",
        project=OUTPUT_PROJECT,
        name=OUTPUT_NAME,
        exist_ok=True,
        seed=RANDOM_SEED,
        verbose=True,
        plots=True,
        save=True,
        iou=IOU_THRESHOLD,
        conf=CONF_THRESHOLD,
        amp=USE_AMP,
        max_det=300,
        close_mosaic=10,  # ✅ Son 10 epoch'ta mosaic'i kapat
        **aug_params
    )
    
    end_time = datetime.now()
    total_duration = (time.time() - start_timestamp) / 60
    
    print("\n" + "=" * 80)
    print("EĞİTİM TAMAMLANDI!")
    print("=" * 80)
    print(f"⏱️  Toplam süre: {total_duration:.2f} dakika")
    print(f"📂 Sonuçlar: {run_dir}")
    print(f"📂 Epoch logları: {epoch_logs_dir}")
    
    # Sonuçları topla
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
    
    # ✅ EN İYİ MODEL İLE TEST DEĞERLENDİRME
    best_model_test = {}
    map_025_best = {}
    if os.path.exists(BEST_MODEL_PATH):
        print("\n" + "=" * 80)
        print("EN İYİ MODEL (BEST) DEĞERLENDİRME")
        print("=" * 80)
        best_model = YOLO(BEST_MODEL_PATH)  # ✅ YOLO11 için değiştirildi
        
        # Standart validation
        print("🔍 Standart validation...")
        best_val = best_model.val(data=DATASET_YAML, verbose=False)
        best_model_test = {
            'map50': float(best_val.box.map50),
            'map50_95': float(best_val.box.map),
            'precision': float(best_val.box.mp),
            'recall': float(best_val.box.mr),
        }
        print(f"   mAP50: {best_model_test['map50']:.4f}")
        print(f"   mAP50-95: {best_model_test['map50_95']:.4f}")
        print(f"   Precision: {best_model_test['precision']:.4f}")
        print(f"   Recall: {best_model_test['recall']:.4f}")
        
        # mAP@0.25 ile özel değerlendirme
        map_025_best = evaluate_with_custom_iou(best_model, DATASET_YAML, IOU_THRESHOLD)
        if map_025_best:
            best_model_test['map_025'] = map_025_best.get('map50', 0)
            print(f"   mAP@{IOU_THRESHOLD}: {best_model_test['map_025']:.4f}")
    
    # ✅ SON MODEL İLE TEST DEĞERLENDİRME
    last_model_test = {}
    map_025_last = {}
    if os.path.exists(LAST_MODEL_PATH):
        print("\n" + "=" * 80)
        print("SON MODEL (LAST) DEĞERLENDİRME")
        print("=" * 80)
        last_model = YOLO(LAST_MODEL_PATH)  # ✅ YOLO11 için değiştirildi
        
        # Standart validation
        print("🔍 Standart validation...")
        last_val = last_model.val(data=DATASET_YAML, verbose=False)
        last_model_test = {
            'map50': float(last_val.box.map50),
            'map50_95': float(last_val.box.map),
            'precision': float(last_val.box.mp),
            'recall': float(last_val.box.mr),
        }
        print(f"   mAP50: {last_model_test['map50']:.4f}")
        print(f"   mAP50-95: {last_model_test['map50_95']:.4f}")
        print(f"   Precision: {last_model_test['precision']:.4f}")
        print(f"   Recall: {last_model_test['recall']:.4f}")
        
        # mAP@0.25 ile özel değerlendirme
        map_025_last = evaluate_with_custom_iou(last_model, DATASET_YAML, IOU_THRESHOLD)
        if map_025_last:
            last_model_test['map_025'] = map_025_last.get('map50', 0)
            print(f"   mAP@{IOU_THRESHOLD}: {last_model_test['map_025']:.4f}")
    
    # En iyi epoch'u bul
    best_epoch = 0
    best_map50 = 0
    for e in epoch_history:
        if e.get('val_map50', 0) > best_map50:
            best_map50 = e.get('val_map50', 0)
            best_epoch = e.get('epoch', 0)
    
    # Training info hazırla
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
            "cls_weight": CLS_WEIGHT,
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
        "epoch_history": epoch_history,
        "best_results": {
            "epoch": best_epoch,
            "map50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        },
        "best_model_test": best_model_test,
        "last_model_test": last_model_test,
        
        # ✅ mAP@0.25 sonuçlarını ayrı bir bölüm olarak ekle
        "map_025_results": {
            "best": map_025_best,
            "last": map_025_last,
            "iou_threshold": IOU_THRESHOLD,
            "note": f"Custom IoU threshold evaluation at {IOU_THRESHOLD}"
        },
        
        "model_paths": {
            "best": BEST_MODEL_PATH,
            "last": LAST_MODEL_PATH,
            "run_dir": run_dir,
            "epoch_logs": epoch_logs_dir,
        }
    }
    
    # Sonuçları kaydet
    save_training_results(training_info, run_dir)
    
    # ✅ Eğitim sonrası otomatik validation
    print("\n" + "=" * 80)
    print("OTOMATİK VALİDASYON BAŞLIYOR...")
    print("=" * 80)
    validate(BEST_MODEL_PATH)
    
    print("\n" + "=" * 80)
    print("🎉 TÜM İŞLEMLER TAMAMLANDI!")
    print("=" * 80)
    print(f"📂 Sonuçlar: {run_dir}")
    print(f"📂 Epoch logları: {epoch_logs_dir}")
    print(f"🏆 En iyi model: {BEST_MODEL_PATH}")
    print(f"📦 Son model: {LAST_MODEL_PATH}")
    
    return results


def validate(model_path: str = None):
    """Eğitilmiş modeli validation set üzerinde değerlendir"""
    if model_path is None:
        model_path = BEST_MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"❌ Model bulunamadı: {model_path}")
        return None
    
    print(f"📥 Model yükleniyor: {model_path}")
    model = YOLO(model_path)  # ✅ YOLO11 için değiştirildi
    
    print("\n🔍 Validation başlıyor...")
    
    # Standart validation
    results = model.val(data=DATASET_YAML)
    
    print("\n" + "=" * 80)
    print("VALIDATION SONUÇLARI:")
    print("=" * 80)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    # mAP@0.25 ile özel değerlendirme
    print(f"\n🎯 mAP@{IOU_THRESHOLD} ile değerlendirme...")
    map_025 = evaluate_with_custom_iou(model, DATASET_YAML, IOU_THRESHOLD)
    if map_025:
        print(f"\nmAP@{IOU_THRESHOLD} SONUÇLARI:")
        print("-" * 40)
        print(f"mAP50: {map_025.get('map50', 0):.4f}")
        print(f"Precision: {map_025.get('precision', 0):.4f}")
        print(f"Recall: {map_025.get('recall', 0):.4f}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--val":
        validate()
    else:
        train()
