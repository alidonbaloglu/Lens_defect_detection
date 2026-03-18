"""
Active Learning ve HITL (Human-in-the-Loop) Modülü
RT-DETR Ensemble sistemi için belirsizlik tabanlı veri toplama ve eğitim
"""

import os
import json
import shutil
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
import torch
from pathlib import Path


class UncertaintySampler:
    """
    Belirsizlik örneklemesi için sınıf.
    Güven skorlarına göre modelin kararsız kaldığı örnekleri seçer.
    """
    
    def __init__(
        self,
        low_conf_threshold: float = 0.30,
        high_conf_threshold: float = 0.70,
        save_high_conf_ratio: float = 0.05  # Yüksek güvenli olanlardan da %5 sakla
    ):
        """
        Args:
            low_conf_threshold: Bu değerin altındaki tespitler "kesin yanlış" kabul edilir
            high_conf_threshold: Bu değerin üstündeki tespitler "kesin doğru" kabul edilir
            save_high_conf_ratio: Yüksek güvenli görüntülerden kaçının saklanacağı
        """
        self.low_conf = low_conf_threshold
        self.high_conf = high_conf_threshold
        self.save_high_conf_ratio = save_high_conf_ratio
        self.high_conf_counter = 0
        
    def should_annotate(self, predictions: List[Tuple]) -> Tuple[bool, str, float]:
        """
        Bir görüntünün manuel kontrole gönderilip gönderilmeyeceğini belirler.
        
        Args:
            predictions: [(x1, y1, x2, y2, conf, cls_idx), ...]
            
        Returns:
            (should_annotate, reason, uncertainty_score)
        """
        if not predictions:
            # Hiç tespit yok - bu da belirsizlik
            return True, "no_detection", 1.0
        
        # Tüm güven skorlarını al
        confidences = [pred[4] if len(pred) >= 5 else pred[2] for pred in predictions]
        
        if not confidences:
            return True, "invalid_predictions", 1.0
        
        max_conf = max(confidences)
        min_conf = min(confidences)
        avg_conf = np.mean(confidences)
        
        # Belirsizlik skoru (0-1 arası, 1 = çok belirsiz)
        uncertainty = 1.0 - avg_conf
        
        # Karar mantığı
        if avg_conf < self.low_conf:
            # Düşük güven - kesinlikle kontrol et
            return True, "low_confidence", uncertainty
        
        elif avg_conf > self.high_conf:
            # Yüksek güven - ama bazen bunları da kontrol et
            self.high_conf_counter += 1
            if (self.high_conf_counter % int(1/self.save_high_conf_ratio)) == 0:
                return True, "high_confidence_sample", uncertainty
            return False, "high_confidence_skip", uncertainty
        
        else:
            # Orta güven - belirsizlik bölgesi, kesinlikle kontrol et
            return True, "uncertain", uncertainty


class ActiveLearningDataManager:
    """
    Aktif öğrenme için veri yönetimi.
    Inference sonuçlarını saklar ve etiketleme havuzunu yönetir.
    """
    
    def __init__(self, root_dir: str = "active_learning_data"):
        """
        Args:
            root_dir: Tüm AL verilerinin saklanacağı ana dizin
        """
        self.root_dir = Path(root_dir)
        self.setup_directories()
        
    def setup_directories(self):
        """Gerekli dizin yapısını oluşturur"""
        dirs = [
            "inference_results",      # Tüm inference sonuçları
            "annotation_queue",       # Etiketleme bekleyen görüntüler
            "annotated",              # Etiketlenmiş veriler
            "training_ready",         # Eğitime hazır YOLO formatı
            "models",                 # Model versiyonları
            "metrics"                 # Performans metrikleri
        ]
        
        for d in dirs:
            (self.root_dir / d).mkdir(parents=True, exist_ok=True)
            
    def save_inference_result(
        self,
        image: np.ndarray,
        predictions: List[Tuple],
        metadata: Dict,
        uncertainty_info: Tuple[bool, str, float]
    ) -> str:
        """
        Inference sonucunu ve metadata'yı kaydeder.
        
        Args:
            image: BGR görüntü
            predictions: Model tahminleri
            metadata: Kamera bilgisi, timestamp vb.
            uncertainty_info: (should_annotate, reason, uncertainty_score)
            
        Returns:
            Kaydedilen dosyanın ID'si
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_id = f"{metadata.get('camera_idx', 'unk')}_{timestamp}"
        
        # Görüntüyü kaydet
        img_dir = self.root_dir / "inference_results" / "images"
        img_dir.mkdir(exist_ok=True)
        img_path = img_dir / f"{file_id}.png"
        cv2.imwrite(str(img_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        # Metadata ve predictions'ı JSON olarak kaydet
        json_dir = self.root_dir / "inference_results" / "predictions"
        json_dir.mkdir(exist_ok=True)
        json_path = json_dir / f"{file_id}.json"
        
        data = {
            "file_id": file_id,
            "timestamp": timestamp,
            "image_path": str(img_path),
            "metadata": metadata,
            "predictions": [
                {
                    "bbox": pred[:4],
                    "confidence": float(pred[4] if len(pred) >= 5 else pred[2]),
                    "class": int(pred[5] if len(pred) >= 6 else pred[3])
                }
                for pred in predictions
            ],
            "uncertainty": {
                "should_annotate": uncertainty_info[0],
                "reason": uncertainty_info[1],
                "score": float(uncertainty_info[2])
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Eğer etiketleme gerekiyorsa, kopyasını annotation_queue'ya ekle
        if uncertainty_info[0]:
            self.add_to_annotation_queue(file_id, img_path, data)
        
        return file_id
    
    def add_to_annotation_queue(self, file_id: str, img_path: Path, data: Dict):
        """Etiketleme kuyruğuna görüntü ekler"""
        queue_dir = self.root_dir / "annotation_queue"
        
        # Görüntüyü kopyala
        dst_img = queue_dir / f"{file_id}.png"
        shutil.copy2(img_path, dst_img)
        
        # JSON'ı kopyala (etiketleme arayüzü için)
        dst_json = queue_dir / f"{file_id}.json"
        with open(dst_json, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_annotation_queue_stats(self) -> Dict:
        """Etiketleme kuyruğu istatistiklerini döndürür"""
        queue_dir = self.root_dir / "annotation_queue"
        images = list(queue_dir.glob("*.png"))
        
        if not images:
            return {"count": 0, "reasons": {}}
        
        # Belirsizlik nedenlerini say
        reasons = {}
        for img in images:
            json_path = img.with_suffix('.json')
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    reason = data.get('uncertainty', {}).get('reason', 'unknown')
                    reasons[reason] = reasons.get(reason, 0) + 1
        
        return {
            "count": len(images),
            "reasons": reasons
        }
    
    def convert_annotations_to_yolo(self, class_names: List[str]) -> int:
        """
        Etiketlenmiş verileri YOLO formatına çevirir.
        
        Returns:
            Dönüştürülen dosya sayısı
        """
        annotated_dir = self.root_dir / "annotated"
        training_dir = self.root_dir / "training_ready"
        
        img_train_dir = training_dir / "images"
        lbl_train_dir = training_dir / "labels"
        img_train_dir.mkdir(exist_ok=True)
        lbl_train_dir.mkdir(exist_ok=True)
        
        converted = 0
        
        # Etiketlenmiş JSON dosyalarını işle
        for json_file in annotated_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Görüntü boyutlarını al
                img_path = annotated_dir / f"{data['file_id']}.png"
                if not img_path.exists():
                    continue
                
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                
                # YOLO formatında label oluştur
                yolo_labels = []
                for pred in data.get('corrected_predictions', data.get('predictions', [])):
                    x1, y1, x2, y2 = pred['bbox']
                    cls = pred['class']
                    
                    # YOLO formatı: class x_center y_center width height (normalize)
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    yolo_labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # Dosyaları kaydet
                dst_img = img_train_dir / f"{data['file_id']}.png"
                dst_lbl = lbl_train_dir / f"{data['file_id']}.txt"
                
                shutil.copy2(img_path, dst_img)
                with open(dst_lbl, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                
                converted += 1
                
            except Exception as e:
                print(f"Hata - {json_file}: {e}")
                continue
        
        # data.yaml oluştur
        yaml_path = training_dir / "data.yaml"
        yaml_content = f"""
path: {training_dir.absolute()}
train: images
val: images  # Ayrı validation set eklenebilir

names:
"""
        for idx, name in enumerate(class_names):
            yaml_content += f"  {idx}: {name}\n"
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        return converted


class IncrementalTrainer:
    """
    Artımlı eğitim için yardımcı sınıf.
    Catastrophic forgetting'i önlemek için eski verilerle birleştirme yapar.
    """
    
    def __init__(
        self,
        old_dataset_path: str,
        new_dataset_path: str,
        old_data_ratio: float = 0.3
    ):
        """
        Args:
            old_dataset_path: Eski eğitim setinin yolu
            new_dataset_path: Yeni (AL'den gelen) verinin yolu
            old_data_ratio: Eski veriden ne kadarının kullanılacağı (0-1)
        """
        self.old_path = Path(old_dataset_path)
        self.new_path = Path(new_dataset_path)
        self.old_ratio = old_data_ratio
        
    def merge_datasets(self, output_path: str) -> Path:
        """
        Eski ve yeni veri setlerini birleştirir.
        
        Returns:
            Birleştirilmiş veri setinin yolu
        """
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)
        
        img_dir = output / "images"
        lbl_dir = output / "labels"
        img_dir.mkdir(exist_ok=True)
        lbl_dir.mkdir(exist_ok=True)
        
        # Yeni verileri tamamen kopyala
        new_imgs = list((self.new_path / "images").glob("*.png"))
        for img in new_imgs:
            shutil.copy2(img, img_dir / img.name)
            lbl = (self.new_path / "labels" / img.stem).with_suffix('.txt')
            if lbl.exists():
                shutil.copy2(lbl, lbl_dir / lbl.name)
        
        # Eski verilerden rastgele örnekle
        old_imgs = list((self.old_path / "images").glob("*.png"))
        n_old_samples = int(len(old_imgs) * self.old_ratio)
        
        if n_old_samples > 0:
            selected_old = np.random.choice(old_imgs, size=n_old_samples, replace=False)
            for img in selected_old:
                dst_img = img_dir / f"old_{img.name}"
                shutil.copy2(img, dst_img)
                lbl = (self.old_path / "labels" / img.stem).with_suffix('.txt')
                if lbl.exists():
                    shutil.copy2(lbl, lbl_dir / f"old_{img.stem}.txt")
        
        # data.yaml kopyala veya oluştur
        old_yaml = self.new_path / "data.yaml"
        if old_yaml.exists():
            shutil.copy2(old_yaml, output / "data.yaml")
        
        print(f"Veri seti birleştirildi: {len(new_imgs)} yeni + {n_old_samples} eski = {len(new_imgs) + n_old_samples}")
        
        return output
    
    def train_model(
        self,
        base_model_path: str,
        merged_dataset_path: str,
        output_dir: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        lr0: float = 0.0001  # Düşük learning rate - fine-tuning için
    ):
        """
        Artımlı eğitim başlatır.
        
        Not: Bu fonksiyon ultralytics YOLO API'sini kullanır.
        """
        try:
            from ultralytics import RTDETR
            
            # Mevcut modeli yükle
            model = RTDETR(base_model_path)
            
            # Fine-tuning parametreleri
            results = model.train(
                data=str(Path(merged_dataset_path) / "data.yaml"),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                lr0=lr0,
                project=output_dir,
                name=f"incremental_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                patience=10,
                save=True,
                plots=True,
                verbose=True
            )
            
            return results
            
        except Exception as e:
            print(f"Eğitim hatası: {e}")
            return None
