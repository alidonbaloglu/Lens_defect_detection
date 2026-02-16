"""
Active Learning Uygulaması - Max Score Ensemble
RT-DETR Ensemble sistemi için görüntü tahmin ve etiket düzeltme aracı

Ensemble Stratejisi:
- İki model de aynı nesneyi tespit ederse → En yüksek confidence skora sahip tahmini al
- Sadece bir model tespit ederse → O modelin tahminini doğrudan kullan

İş Akışı:
1. Görüntü klasörü seçimi
2. Ensemble ile toplu tahmin
3. YOLO formatında kayıt
4. Etiketleme penceresi ile düzeltme
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime

import cv2
import numpy as np
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from ultralytics import RTDETR

# Active Learning modülünü import et
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from active_learning.active_learning_module import UncertaintySampler, ActiveLearningDataManager

# ----------------------------
# Config
# ----------------------------
SCRATCH_MODEL_PATH = r"C:/Users/ali.donbaloglu/Desktop/Lens/RT_Detr_Ensemble/model/best_cizik.pt"
BLACKDOT_MODEL_PATH = r"C:/Users/ali.donbaloglu/Desktop/Lens/RT_Detr_Ensemble/model/best_siyahnokta.pt"

ENSEMBLE_CLASS_NAMES = ["scratch", "black_dot"]
DEFAULT_CONF_THRESHOLD = 0.40
NMS_IOU_THRESHOLD = 0.40

# Active Learning Ayarları
AL_LOW_CONF_THRESHOLD = 0.30
AL_HIGH_CONF_THRESHOLD = 0.70
AL_SAMPLE_HIGH_CONF = 0.05

# Renkler (BGR)
COLORS = {
    'scratch': (0, 255, 0),     # Yeşil
    'black_dot': (255, 0, 0),   # Mavi
    'selected': (0, 255, 255),  # Sarı - seçili kutu
    'hover': (255, 255, 0),     # Cyan - hover
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Ensemble Fonksiyonları
# ----------------------------
def calculate_iou(boxA, boxB):
    """İki bounding box arasındaki IoU değerini hesapla."""
    def to_corners(box):
        x, y, w, h = box
        return [x - w/2, y - h/2, x + w/2, y + h/2]

    A = to_corners(boxA)
    B = to_corners(boxB)

    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]

    denominator = areaA + areaB - inter
    return inter / denominator if denominator > 0 else 0


def ensemble_predictions(scratch_results, blackdot_results, img_shape, nms_iou=0.5, conf_threshold=0.40):
    """
    İki modelin tahminlerini birleştir.
    
    Max Score Stratejisi:
    - İki model de aynı nesneyi tespit ederse → En yüksek confidence skora sahip tahmini seç
    - Sadece bir model tespit ederse → O modelin tahminini doğrudan kullan
    """
    h, w = img_shape[:2]
    
    scratch_preds = []
    blackdot_preds = []
    
    # Çizik modelinden tahminleri al
    if scratch_results and scratch_results[0].boxes:
        for box in scratch_results[0].boxes:
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            
            if hasattr(box, 'xywhn'):
                pred_box = box.xywhn[0].tolist()
            else:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pred_box = [
                    ((x1 + x2) / 2) / w,
                    ((y1 + y2) / 2) / h,
                    (x2 - x1) / w,
                    (y2 - y1) / h
                ]
            
            scratch_preds.append({
                'cls_id': cls_id,
                'box': pred_box,
                'conf': conf,
                'matched': False
            })
    
    # Siyah nokta modelinden tahminleri al
    if blackdot_results and blackdot_results[0].boxes:
        for box in blackdot_results[0].boxes:
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            
            if hasattr(box, 'xywhn'):
                pred_box = box.xywhn[0].tolist()
            else:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pred_box = [
                    ((x1 + x2) / 2) / w,
                    ((y1 + y2) / 2) / h,
                    (x2 - x1) / w,
                    (y2 - y1) / h
                ]
            
            blackdot_preds.append({
                'cls_id': cls_id,
                'box': pred_box,
                'conf': conf,
                'matched': False
            })
    
    ensemble_results = []
    
    # Eşleştirme ve birleştirme
    for sp in scratch_preds:
        best_match = None
        best_iou = 0
        
        for bp in blackdot_preds:
            if bp['matched']:
                continue
            if sp['cls_id'] != bp['cls_id']:
                continue
            
            iou = calculate_iou(sp['box'], bp['box'])
            if iou >= nms_iou and iou > best_iou:
                best_iou = iou
                best_match = bp
        
        if best_match:
            # İKİ MODEL DE TESPİT ETTİ → EN YÜKSEK SKORU SEÇ
            best_match['matched'] = True
            sp['matched'] = True
            cls_id = sp['cls_id']
            
            # Hangi modelin confidence skoru daha yüksekse onu al
            if sp['conf'] >= best_match['conf']:
                # Scratch modeli daha yüksek skor verdi
                ensemble_conf = sp['conf']
                ensemble_box = sp['box']
                source = 'S'  # Scratch modelinden geldi
            else:
                # BlackDot modeli daha yüksek skor verdi
                ensemble_conf = best_match['conf']
                ensemble_box = best_match['box']
                source = 'B'  # BlackDot modelinden geldi
            
            # (cls_id, box, conf, model_source) - 'E' yerine kazanan modelin kaynağını yaz
            ensemble_results.append((cls_id, ensemble_box, ensemble_conf, source))
        else:
            # SADECE ÇİZİK MODELİ TESPİT ETTİ
            sp['matched'] = True
            cls_id = sp['cls_id']
            ensemble_results.append((cls_id, sp['box'], sp['conf'], 'S'))
    
    for bp in blackdot_preds:
        if not bp['matched']:
            # SADECE SİYAH NOKTA MODELİ TESPİT ETTİ
            cls_id = bp['cls_id']
            ensemble_results.append((cls_id, bp['box'], bp['conf'], 'B'))
    
    # Confidence filtreleme
    filtered_results = [r for r in ensemble_results if r[2] >= conf_threshold]
    return filtered_results


def load_ensemble_models():
    """RT-DETR modellerini yükle."""
    device = torch.device(DEVICE)
    
    if not os.path.exists(SCRATCH_MODEL_PATH):
        raise RuntimeError(f"Çizik modeli bulunamadı: {SCRATCH_MODEL_PATH}")
    if not os.path.exists(BLACKDOT_MODEL_PATH):
        raise RuntimeError(f"Siyah nokta modeli bulunamadı: {BLACKDOT_MODEL_PATH}")
    
    scratch_model = RTDETR(SCRATCH_MODEL_PATH)
    blackdot_model = RTDETR(BLACKDOT_MODEL_PATH)
    
    if device.type == "cuda":
        scratch_model = scratch_model.to(device)
        blackdot_model = blackdot_model.to(device)
    
    return scratch_model, blackdot_model


# ----------------------------
# YOLO Format Fonksiyonları
# ----------------------------
def save_yolo_labels(label_path: str, predictions: List[Tuple], img_shape: Tuple[int, int]):
    """Tahminleri standart YOLO formatında kaydet + metadata dosyasına ekle."""
    # Standart YOLO formatında kaydet
    with open(label_path, 'w') as f:
        for pred in predictions:
            cls_id, box = pred[0], pred[1]
            x_center, y_center, width, height = box
            # Standart YOLO format: cls x y w h
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    # Tek metadata dosyasına ekle (labels klasöründe)
    labels_dir = os.path.dirname(label_path)
    meta_path = os.path.join(labels_dir, "metadata.txt")
    label_name = os.path.basename(label_path)
    
    # Mevcut metadata'yı oku
    existing_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            for line in f:
                if '|' in line:
                    parts = line.strip().split('|')
                    existing_meta[parts[0]] = parts[1] if len(parts) > 1 else ''
    
    # Bu dosyanın metadata'ını oluştur
    meta_entries = []
    for pred in predictions:
        conf = pred[2] if len(pred) > 2 else 1.0
        model_source = pred[3] if len(pred) > 3 else 'E'
        meta_entries.append(f"{conf:.4f},{model_source}")
    
    existing_meta[label_name] = ';'.join(meta_entries)
    
    # Metadata dosyasını yeniden yaz
    with open(meta_path, 'w') as f:
        for name, data in sorted(existing_meta.items()):
            f.write(f"{name}|{data}\n")


def load_yolo_labels(label_path: str) -> List[Tuple]:
    """Standart YOLO formatındaki etiketleri + metadata yükle."""
    labels = []
    metadata = []
    
    # Tek metadata dosyasından bu dosyanın verisini oku
    labels_dir = os.path.dirname(label_path)
    meta_path = os.path.join(labels_dir, "metadata.txt")
    label_name = os.path.basename(label_path)
    
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            for line in f:
                if '|' in line:
                    parts = line.strip().split('|')
                    if parts[0] == label_name and len(parts) > 1 and parts[1]:
                        for entry in parts[1].split(';'):
                            if ',' in entry:
                                conf_str, source = entry.split(',')
                                metadata.append((float(conf_str), source))
                        break
    
    # YOLO label dosyasını yükle
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Metadata varsa kullan, yoksa varsayılan
                    if i < len(metadata):
                        conf, model_source = metadata[i]
                    else:
                        conf, model_source = 1.0, 'M'
                    
                    labels.append((cls_id, [x_center, y_center, width, height], conf, model_source))
    return labels


# ----------------------------
# Ana Pencere
# ----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Active Learning - RT-DETR Max Score Ensemble")
        self.setMinimumSize(800, 600)
        
        # Modeller
        self.scratch_model = None
        self.blackdot_model = None
        self.models_loaded = False
        
        # Veriler
        self.image_folder = None
        self.image_files = []
        self.predictions_dict = {}  # {filename: [(cls_id, box, conf), ...]}
        
        # AL bileşenleri
        self.al_sampler = UncertaintySampler(
            low_conf_threshold=AL_LOW_CONF_THRESHOLD,
            high_conf_threshold=AL_HIGH_CONF_THRESHOLD,
            save_high_conf_ratio=AL_SAMPLE_HIGH_CONF
        )
        
        self.init_ui()
    
    def init_ui(self):
        """Arayüzü oluştur."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        
        layout = QtWidgets.QVBoxLayout(central)
        
        # Başlık
        title = QtWidgets.QLabel("Active Learning - Max Score Ensemble")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Açıklama
        desc = QtWidgets.QLabel("Strateji: İki model de tespit ederse → en yüksek skoru al")
        desc.setStyleSheet("font-size: 12px; color: #aaa; padding: 2px;")
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)
        
        # Klasör seçimi
        folder_layout = QtWidgets.QHBoxLayout()
        self.folder_label = QtWidgets.QLabel("Klasör seçilmedi")
        self.folder_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 5px;")
        folder_layout.addWidget(self.folder_label, stretch=1)
        
        btn_select = QtWidgets.QPushButton("Klasör Seç")
        btn_select.clicked.connect(self.select_folder)
        folder_layout.addWidget(btn_select)
        layout.addLayout(folder_layout)
        
        # İstatistikler
        stats_group = QtWidgets.QGroupBox("İstatistikler")
        stats_layout = QtWidgets.QGridLayout(stats_group)
        
        self.label_total_images = QtWidgets.QLabel("Toplam Görüntü: 0")
        self.label_processed = QtWidgets.QLabel("İşlenen: 0")
        self.label_detections = QtWidgets.QLabel("Toplam Tespit: 0")
        self.label_uncertain = QtWidgets.QLabel("Belirsiz: 0")
        
        stats_layout.addWidget(self.label_total_images, 0, 0)
        stats_layout.addWidget(self.label_processed, 0, 1)
        stats_layout.addWidget(self.label_detections, 1, 0)
        stats_layout.addWidget(self.label_uncertain, 1, 1)
        layout.addWidget(stats_group)
        
        # Progress bar
        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        
        # Status
        self.status_label = QtWidgets.QLabel("Hazır")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)
        
        # Butonlar
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.btn_load_models = QtWidgets.QPushButton("Modelleri Yükle")
        self.btn_load_models.clicked.connect(self.load_models)
        btn_layout.addWidget(self.btn_load_models)
        
        self.btn_predict = QtWidgets.QPushButton("Tahmin Yap ve Kaydet")
        self.btn_predict.clicked.connect(self.run_predictions)
        self.btn_predict.setEnabled(False)
        btn_layout.addWidget(self.btn_predict)
        
        self.btn_annotate = QtWidgets.QPushButton("Etiketleme Penceresi")
        self.btn_annotate.clicked.connect(self.open_annotation_window)
        self.btn_annotate.setEnabled(False)
        btn_layout.addWidget(self.btn_annotate)
        
        layout.addLayout(btn_layout)
        
        # Log alanı
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)
        
        layout.addStretch()
    
    def log(self, message: str):
        """Log mesajı ekle."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        QtWidgets.QApplication.processEvents()
    
    def select_folder(self):
        """Görüntü klasörü seç."""
        folder = QFileDialog.getExistingDirectory(self, "Görüntü Klasörü Seç")
        if folder:
            self.image_folder = folder
            self.folder_label.setText(folder)
            
            # Görüntüleri bul
            self.image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                self.image_files.extend(Path(folder).glob(ext))
            self.image_files = sorted([str(f) for f in self.image_files])
            
            self.label_total_images.setText(f"Toplam Görüntü: {len(self.image_files)}")
            self.log(f"Klasör seçildi: {folder} ({len(self.image_files)} görüntü)")
            
            if self.models_loaded:
                self.btn_predict.setEnabled(True)
    
    def load_models(self):
        """Modelleri yükle."""
        self.status_label.setText("Modeller yükleniyor...")
        self.log("Modeller yükleniyor...")
        QtWidgets.QApplication.processEvents()
        
        try:
            self.scratch_model, self.blackdot_model = load_ensemble_models()
            self.models_loaded = True
            self.btn_load_models.setEnabled(False)
            self.btn_load_models.setText("✓ Modeller Yüklendi")
            self.status_label.setText("Modeller yüklendi")
            self.log(f"Modeller başarıyla yüklendi (device: {DEVICE})")
            
            if self.image_folder:
                self.btn_predict.setEnabled(True)
                
        except Exception as e:
            self.log(f"HATA: Model yüklenemedi - {e}")
            QMessageBox.critical(self, "Hata", f"Model yüklenemedi:\n{e}")
    
    @torch.no_grad()
    def run_predictions(self, checked=None):
        """Tüm görüntülerde tahmin yap ve YOLO formatında kaydet."""
        if not self.image_files:
            QMessageBox.warning(self, "Uyarı", "Önce bir klasör seçin!")
            return
        
        if not self.models_loaded:
            QMessageBox.warning(self, "Uyarı", "Önce modelleri yükleyin!")
            return
        
        # Labels klasörü oluştur
        labels_dir = os.path.join(self.image_folder, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        
        # Predictions klasörü oluştur (tahmin görselleri için)
        predictions_dir = os.path.join(self.image_folder, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        self.predictions_dict = {}
        total_detections = 0
        uncertain_count = 0
        
        self.progress.setMaximum(len(self.image_files))
        self.btn_predict.setEnabled(False)
        
        # Model kaynağına göre renkler (BGR)
        SOURCE_COLORS = {
            'E': (0, 200, 255),   # Turuncu-sarı - Ensemble
            'S': (0, 255, 0),     # Yeşil - Scratch modeli
            'B': (255, 0, 0),     # Mavi - BlackDot modeli
        }
        
        for i, img_path in enumerate(self.image_files):
            self.status_label.setText(f"İşleniyor: {os.path.basename(img_path)}")
            self.progress.setValue(i + 1)
            QtWidgets.QApplication.processEvents()
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    self.log(f"Okunamadı: {img_path}")
                    continue
                
                # Tahmin yap
                scratch_results = self.scratch_model(img, conf=DEFAULT_CONF_THRESHOLD, verbose=False)
                blackdot_results = self.blackdot_model(img, conf=DEFAULT_CONF_THRESHOLD, verbose=False)
                
                # Ensemble birleştir (Max Score)
                ensemble_preds = ensemble_predictions(
                    scratch_results,
                    blackdot_results,
                    img.shape,
                    nms_iou=NMS_IOU_THRESHOLD,
                    conf_threshold=DEFAULT_CONF_THRESHOLD
                )
                
                # Belirsizlik kontrolü
                should_annotate, reason, uncertainty = self.al_sampler.should_annotate(ensemble_preds)
                if should_annotate:
                    uncertain_count += 1
                
                # YOLO formatında kaydet
                filename = os.path.basename(img_path)
                label_name = os.path.splitext(filename)[0] + ".txt"
                label_path = os.path.join(labels_dir, label_name)
                
                save_yolo_labels(label_path, ensemble_preds, img.shape[:2])
                
                # Tahmin yapılmış görseli kaydet
                pred_img = img.copy()
                h, w = pred_img.shape[:2]
                
                for pred in ensemble_preds:
                    cls_id, box, conf = pred[:3]
                    model_source = pred[3] if len(pred) > 3 else 'E'
                    
                    x_center, y_center, bw, bh = box
                    
                    # YOLO formatından piksel koordinatlarına çevir
                    x1 = int((x_center - bw/2) * w) - 5
                    y1 = int((y_center - bh/2) * h) - 5
                    x2 = int((x_center + bw/2) * w) + 5
                    y2 = int((y_center + bh/2) * h) + 5
                    
                    # Sınırları kontrol et
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w - 1, x2)
                    y2 = min(h - 1, y2)
                    
                    # Renk ve sınıf adı
                    color = SOURCE_COLORS.get(model_source, (0, 200, 255))
                    cls_name = ENSEMBLE_CLASS_NAMES[cls_id] if cls_id < len(ENSEMBLE_CLASS_NAMES) else 'scratch'
                    
                    # Kutu çiz
                    cv2.rectangle(pred_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Etiket metni
                    label_text = f"[{model_source}] {cls_name} {conf:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 2
                    
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
                    text_x = x1
                    text_y = y2 + text_h + 8
                    
                    if text_y > h - 5:
                        text_y = y1 - 8
                    
                    cv2.putText(pred_img, label_text, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
                
                # Görseli kaydet
                pred_filename = os.path.splitext(filename)[0] + "_pred.jpg"
                pred_path = os.path.join(predictions_dir, pred_filename)
                cv2.imwrite(pred_path, pred_img)
                
                self.predictions_dict[img_path] = ensemble_preds
                total_detections += len(ensemble_preds)
                
            except Exception as e:
                self.log(f"Hata ({img_path}): {e}")
        
        self.label_processed.setText(f"İşlenen: {len(self.predictions_dict)}")
        self.label_detections.setText(f"Toplam Tespit: {total_detections}")
        self.label_uncertain.setText(f"Belirsiz: {uncertain_count}")
        
        self.status_label.setText("Tahmin tamamlandı!")
        self.log(f"Tamamlandı! {len(self.predictions_dict)} görüntü işlendi, {total_detections} tespit")
        self.log(f"Etiketler kaydedildi: {labels_dir}")
        self.log(f"Tahmin görselleri kaydedildi: {predictions_dir}")
        
        self.btn_predict.setEnabled(True)
        self.btn_annotate.setEnabled(True)
        
        QMessageBox.information(
            self, "Tamamlandı",
            f"Tahmin tamamlandı!\n\n"
            f"İşlenen: {len(self.predictions_dict)} görüntü\n"
            f"Toplam tespit: {total_detections}\n"
            f"Belirsiz: {uncertain_count}\n\n"
            f"Etiketler: {labels_dir}\n"
            f"Tahmin görselleri: {predictions_dir}"
        )
    
    def open_annotation_window(self):
        """Etiketleme penceresini aç."""
        if not self.image_files:
            QMessageBox.warning(self, "Uyarı", "Önce bir klasör seçin!")
            return
        
        labels_dir = os.path.join(self.image_folder, "labels")
        self.annotation_window = AnnotationWindow(self.image_files, labels_dir)
        self.annotation_window.show()
        self.log("Etiketleme penceresi açıldı")


# ----------------------------
# Etiketleme Penceresi
# ----------------------------
class AnnotationWindow(QtWidgets.QMainWindow):
    def __init__(self, image_files: List[str], labels_dir: str):
        super().__init__()
        self.setWindowTitle("Etiket Düzenleme - Max Score")
        self.setMinimumSize(1200, 800)
        
        self.image_files = image_files
        self.labels_dir = labels_dir
        self.current_idx = 0
        self.current_image = None
        self.current_labels = []  # [(cls_id, [x, y, w, h], conf, source), ...]
        self.selected_box_idx = None
        self.dragging = False
        self.drag_start = None
        self.new_box_start = None
        self.hover_box_idx = None
        
        # Zoom ayarları
        self.zoom_level = 1.0
        self.zoom_min = 0.5
        self.zoom_max = 5.0
        self.zoom_step = 0.1
        
        # Pan (kaydırma) ayarları
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.panning = False
        self.pan_start = None
        
        self.init_ui()
        self.load_current_image()
    
    def init_ui(self):
        """Arayüzü oluştur."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        
        main_layout = QtWidgets.QHBoxLayout(central)
        
        # Sol panel - Görüntü
        left_panel = QtWidgets.QVBoxLayout()
        
        # Scroll area - kaydırma için
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setStyleSheet("background-color: black;")
        self.scroll_area.setMinimumSize(800, 600)
        
        # Görüntü alanı
        self.image_label = QtWidgets.QLabel()
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.on_mouse_press
        self.image_label.mouseReleaseEvent = self.on_mouse_release
        self.image_label.mouseMoveEvent = self.on_mouse_move
        self.image_label.wheelEvent = self.on_wheel
        
        self.scroll_area.setWidget(self.image_label)
        left_panel.addWidget(self.scroll_area)
        
        # Navigasyon
        nav_layout = QtWidgets.QHBoxLayout()
        
        self.btn_prev = QtWidgets.QPushButton("◀ Önceki")
        self.btn_prev.clicked.connect(self.prev_image)
        nav_layout.addWidget(self.btn_prev)
        
        self.label_nav = QtWidgets.QLabel("1 / 1")
        self.label_nav.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.label_nav, stretch=1)
        
        self.btn_next = QtWidgets.QPushButton("Sonraki ▶")
        self.btn_next.clicked.connect(self.next_image)
        nav_layout.addWidget(self.btn_next)
        
        left_panel.addLayout(nav_layout)
        main_layout.addLayout(left_panel, stretch=3)
        
        # Sağ panel - Kontroller
        right_panel = QtWidgets.QVBoxLayout()
        
        # Dosya bilgisi
        self.label_filename = QtWidgets.QLabel("Dosya: -")
        self.label_filename.setWordWrap(True)
        right_panel.addWidget(self.label_filename)
        
        # Etiket listesi
        right_panel.addWidget(QtWidgets.QLabel("Etiketler:"))
        self.label_list = QtWidgets.QListWidget()
        self.label_list.itemClicked.connect(self.on_label_selected)
        right_panel.addWidget(self.label_list)
        
        # Düzenleme butonları
        self.btn_delete = QtWidgets.QPushButton("Seçili Etiketi Sil")
        self.btn_delete.clicked.connect(self.delete_selected_label)
        self.btn_delete.setEnabled(False)
        right_panel.addWidget(self.btn_delete)
        
        # Sınıf seçimi (yeni kutu için)
        class_layout = QtWidgets.QHBoxLayout()
        class_layout.addWidget(QtWidgets.QLabel("Yeni Etiket Sınıfı:"))
        self.class_combo = QtWidgets.QComboBox()
        for i, name in enumerate(ENSEMBLE_CLASS_NAMES):
            self.class_combo.addItem(f"{i}: {name}")
        class_layout.addWidget(self.class_combo)
        right_panel.addLayout(class_layout)
        
        right_panel.addWidget(QtWidgets.QLabel("💡 İpucu: Görüntüde sürükleyerek yeni kutu ekleyin"))
        
        # Kaydet butonu
        self.btn_save = QtWidgets.QPushButton("💾 Değişiklikleri Kaydet")
        self.btn_save.clicked.connect(self.save_current_labels)
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        right_panel.addWidget(self.btn_save)
        
        right_panel.addStretch()
        main_layout.addLayout(right_panel, stretch=1)
    
    def load_current_image(self):
        """Mevcut görüntüyü ve etiketleri yükle."""
        if not self.image_files:
            return
        
        img_path = self.image_files[self.current_idx]
        self.current_image = cv2.imread(img_path)
        
        # Etiket dosyasını yükle
        filename = os.path.basename(img_path)
        label_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)
        
        self.current_labels = load_yolo_labels(label_path)
        
        # UI güncelle
        self.label_filename.setText(f"Dosya: {filename}")
        self.label_nav.setText(f"{self.current_idx + 1} / {len(self.image_files)}")
        
        self.update_label_list()
        
        # Zoom ve pan sıfırla
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        
        self.draw_image()
        
        # Buton durumları
        self.btn_prev.setEnabled(self.current_idx > 0)
        self.btn_next.setEnabled(self.current_idx < len(self.image_files) - 1)
    
    def update_label_list(self):
        """Etiket listesini güncelle."""
        self.label_list.clear()
        for i, label_data in enumerate(self.current_labels):
            cls_id, box, conf = label_data[:3]
            model_source = label_data[3] if len(label_data) > 3 else 'E'
            cls_name = ENSEMBLE_CLASS_NAMES[cls_id] if cls_id < len(ENSEMBLE_CLASS_NAMES) else str(cls_id)
            self.label_list.addItem(f"{i+1}. [{model_source}] {cls_name} ({conf:.2f})")
    
    def draw_image(self):
        """Görüntüyü etiketlerle birlikte çiz."""
        if self.current_image is None:
            return
        
        img = self.current_image.copy()
        h, w = img.shape[:2]
        
        # Model kaynağına göre renkler (BGR)
        SOURCE_COLORS = {
            'E': (0, 200, 255),   # Turuncu-sarı - Ensemble
            'S': (0, 255, 0),     # Yeşil - Scratch modeli
            'B': (255, 0, 0),     # Mavi - BlackDot modeli
            'M': (255, 0, 255),   # Magenta - Manuel eklenen
        }
        
        # Kutuları çiz
        for i, label_data in enumerate(self.current_labels):
            cls_id, box, conf = label_data[:3]
            model_source = label_data[3] if len(label_data) > 3 else 'E'
            
            x_center, y_center, bw, bh = box
            
            x1 = int((x_center - bw/2) * w)
            y1 = int((y_center - bh/2) * h)
            x2 = int((x_center + bw/2) * w)
            y2 = int((y_center + bh/2) * h)
            
            # Sınırları kontrol et
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            
            # Sınıf adı
            cls_name = ENSEMBLE_CLASS_NAMES[cls_id] if cls_id < len(ENSEMBLE_CLASS_NAMES) else 'scratch'
            
            # Seçim ve hover durumları
            if i == self.selected_box_idx:
                color = (0, 255, 255)  # Sarı - seçili
                thickness = 2
            elif i == self.hover_box_idx:
                color = (255, 255, 0)  # Cyan - hover
                thickness = 1
            else:
                # Model kaynağına göre renk
                color = SOURCE_COLORS.get(model_source, (0, 200, 255))
                thickness = 1
            
            # Kutu çiz
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Etiket metni
            label_text = f"[{model_source}] {cls_name} {conf:.2f}"
            
            # Font ayarları
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Metin boyutu
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            # Metin konumu - kutunun altında
            text_x = x1
            text_y = y2 + text_h + 8
            
            # Ekran dışına taşarsa yukarı al
            if text_y > h - 5:
                text_y = y1 - 8
            
            # Metin çiz (renk model kaynağına göre)
            cv2.putText(img, label_text, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
        
        # Yeni kutu çizme (sürükleme sırasında)
        if self.new_box_start and self.drag_start:
            x1, y1 = self.new_box_start
            x2, y2 = self.drag_start
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        # QImage'e dönüştür ve göster
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        
        # Zoom uygula
        pixmap = QtGui.QPixmap.fromImage(qimg)
        
        # Scroll area boyutuna göre temel ölçekleme
        target_size = self.scroll_area.size()
        base_scaled = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Zoom seviyesine göre son ölçekleme
        zoomed_w = int(base_scaled.width() * self.zoom_level)
        zoomed_h = int(base_scaled.height() * self.zoom_level)
        zoomed = pixmap.scaled(zoomed_w, zoomed_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Label boyutunu ayarla ve pixmap'i göster
        self.image_label.setFixedSize(zoomed.size())
        self.image_label.setPixmap(zoomed)
    
    def get_image_coords(self, event_pos) -> Tuple[int, int]:
        """Widget koordinatlarını görüntü koordinatlarına çevir."""
        if self.current_image is None:
            return (0, 0)
        
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return (0, 0)
        
        # Label boyutu ve pixmap boyutu
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()
        
        # Offset (ortalama)
        offset_x = (label_w - pixmap_w) // 2
        offset_y = (label_h - pixmap_h) // 2
        
        # Görüntü içindeki koordinat
        img_x = event_pos.x() - offset_x
        img_y = event_pos.y() - offset_y
        
        # Gerçek görüntü boyutuna ölçekle
        img_h, img_w = self.current_image.shape[:2]
        real_x = int(img_x * img_w / pixmap_w)
        real_y = int(img_y * img_h / pixmap_h)
        
        return (real_x, real_y)
    
    def find_box_at(self, x: int, y: int) -> Optional[int]:
        """Verilen koordinattaki kutuyu bul."""
        if self.current_image is None:
            return None
        
        h, w = self.current_image.shape[:2]
        
        for i, label_data in enumerate(self.current_labels):
            cls_id, box, conf = label_data[:3]
            x_center, y_center, bw, bh = box
            
            x1 = int((x_center - bw/2) * w)
            y1 = int((y_center - bh/2) * h)
            x2 = int((x_center + bw/2) * w)
            y2 = int((y_center + bh/2) * h)
            
            if x1 <= x <= x2 and y1 <= y <= y2:
                return i
        
        return None
    
    def on_mouse_press(self, event):
        """Mouse basıldığında."""
        if event.button() == Qt.LeftButton:
            x, y = self.get_image_coords(event.pos())
            box_idx = self.find_box_at(x, y)
            
            if box_idx is not None:
                # Mevcut kutu seçildi
                self.selected_box_idx = box_idx
                self.label_list.setCurrentRow(box_idx)
                self.btn_delete.setEnabled(True)
            else:
                # Yeni kutu çizmeye başla
                self.selected_box_idx = None
                self.btn_delete.setEnabled(False)
                self.new_box_start = (x, y)
                self.dragging = True
            
            self.draw_image()
        
        elif event.button() == Qt.RightButton:
            # Sağ tık ile kaydırma başlat
            self.panning = True
            self.pan_start = event.globalPos()
    
    def on_mouse_release(self, event):
        """Mouse bırakıldığında."""
        if event.button() == Qt.LeftButton and self.dragging:
            x, y = self.get_image_coords(event.pos())
            
            if self.new_box_start:
                x1, y1 = self.new_box_start
                x2, y2 = x, y
                
                # Minimum boyut kontrolü
                if abs(x2 - x1) > 2 and abs(y2 - y1) > 2:
                    h, w = self.current_image.shape[:2]
                    
                    # YOLO formatına çevir
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    bw = abs(x2 - x1) / w
                    bh = abs(y2 - y1) / h
                    
                    cls_id = self.class_combo.currentIndex()
                    # Manual olarak eklenen kutular 'M' kaynağına sahip
                    self.current_labels.append((cls_id, [x_center, y_center, bw, bh], 1.0, 'M'))
                    
                    self.update_label_list()
            
            self.new_box_start = None
            self.dragging = False
            self.drag_start = None
            self.draw_image()
        
        elif event.button() == Qt.RightButton:
            # Kaydırma bitti
            self.panning = False
            self.pan_start = None
    
    def on_mouse_move(self, event):
        """Mouse hareket ettiğinde."""
        # Sağ tık ile kaydırma
        if self.panning and self.pan_start:
            delta = event.globalPos() - self.pan_start
            self.pan_start = event.globalPos()
            
            # Scroll bar'ları hareket ettir
            h_bar = self.scroll_area.horizontalScrollBar()
            v_bar = self.scroll_area.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            return
        
        x, y = self.get_image_coords(event.pos())
        
        if self.dragging and self.new_box_start:
            self.drag_start = (x, y)
            self.draw_image()
        else:
            # Hover effect
            new_hover = self.find_box_at(x, y)
            if new_hover != self.hover_box_idx:
                self.hover_box_idx = new_hover
                self.draw_image()
    
    def on_label_selected(self, item):
        """Liste öğesi seçildiğinde."""
        idx = self.label_list.currentRow()
        self.selected_box_idx = idx
        self.btn_delete.setEnabled(True)
        self.draw_image()
    
    def delete_selected_label(self):
        """Seçili etiketi sil."""
        if self.selected_box_idx is not None and 0 <= self.selected_box_idx < len(self.current_labels):
            del self.current_labels[self.selected_box_idx]
            self.selected_box_idx = None
            self.btn_delete.setEnabled(False)
            self.update_label_list()
            self.draw_image()
    
    def save_current_labels(self):
        """Mevcut etiketleri kaydet."""
        if not self.image_files:
            return
        
        img_path = self.image_files[self.current_idx]
        filename = os.path.basename(img_path)
        label_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)
        
        save_yolo_labels(label_path, self.current_labels, self.current_image.shape[:2])
        
        QMessageBox.information(self, "Kaydedildi", f"Etiketler kaydedildi:\n{label_path}")
    
    def prev_image(self):
        """Önceki görüntüye geç."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.selected_box_idx = None
            self.btn_delete.setEnabled(False)
            self.load_current_image()
    
    def next_image(self):
        """Sonraki görüntüye geç."""
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.selected_box_idx = None
            self.btn_delete.setEnabled(False)
            self.load_current_image()
    
    def keyPressEvent(self, event):
        """Klavye kısayolları."""
        if event.key() == Qt.Key_Delete:
            self.delete_selected_label()
        elif event.key() == Qt.Key_Left:
            self.prev_image()
        elif event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            self.save_current_labels()
        elif event.key() == Qt.Key_R:
            # Zoom ve pan sıfırla
            self.zoom_level = 1.0
            self.pan_offset_x = 0
            self.pan_offset_y = 0
            self.draw_image()
    
    def on_wheel(self, event):
        """Mouse tekerleği ile yakınlaştırma/uzaklaştırma."""
        delta = event.angleDelta().y()
        
        if delta > 0:
            # Yakınlaştır
            self.zoom_level = min(self.zoom_max, self.zoom_level + self.zoom_step)
        else:
            # Uzaklaştır
            self.zoom_level = max(self.zoom_min, self.zoom_level - self.zoom_step)
        
        self.draw_image()


# ----------------------------
# Ana Program
# ----------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Koyu tema
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ToolTipBase, Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, Qt.white)
    palette.setColor(QtGui.QPalette.Text, Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, Qt.red)
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
