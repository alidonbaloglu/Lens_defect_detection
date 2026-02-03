import os
import sys
import time
import csv
from datetime import datetime
from typing import List, Tuple, Optional
import torch
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ultralytics import RTDETR


# ----------------------------
# Config (edit here)
# ----------------------------
DATA_ROOT = os.path.join("results", "DataCollection")
CAMERA_INDICES = [0,1,2,3]
USE_DSHOW = True
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# --- RT-DETR ENSEMBLE AYARLARI ---
# Model Yolları
SCRATCH_MODEL_PATH = r"C:/Users/ali.donbaloglu/Desktop/Lens/RT_Detr_Ensemble/model/best_cizik.pt"
BLACKDOT_MODEL_PATH = r"C:/Users/ali.donbaloglu/Desktop/Lens/RT_Detr_Ensemble/model/best_siyahnokta.pt"

# Sınıf İsimleri
ENSEMBLE_CLASS_NAMES = ["scratch", "black_dot"]

# Eşik Değerleri
DEFAULT_CONF_THRESHOLD = 0.40
DEFAULT_IOU_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.40

# Ensemble Ağırlık Ayarları
SCRATCH_MODEL_STRONG_WEIGHT = 1.0
SCRATCH_MODEL_WEAK_WEIGHT = 0.5
BLACKDOT_MODEL_STRONG_WEIGHT = 1.0
BLACKDOT_MODEL_WEAK_WEIGHT = 0.5

CAMERA_SETTINGS = {
    'width': CAMERA_WIDTH,
    'height': CAMERA_HEIGHT,
    'fps': 30,
    'brightness': 128,
    'contrast': 128,
    'saturation': 128,
    'hue': 0,
    'gain': 0,
    'exposure': 0,
    'focus': 30,
    'sharpness': 128,
    'white_balance': 4500,
    'autofocus': False,
    'auto_exposure': False,
    'auto_white_balance': False,
}
# Sağ (right) per-camera overrides
CAMERA_SETTINGS_PER_CAMERA_SAG: dict[int, dict] = {
    2: {
        'brightness': 100,
        'contrast': 126,
        'saturation': 122,
        'hue': 20,
        'focus': 23,
        'sharpness': 255,
        'white_balance': 3008,
    },
    0: {
        'brightness': 128,
        'contrast': 155,
        'saturation': 59,
        'focus': 30,
        'sharpness': 255,
        'white_balance': 2635,
    },
    1: {
        'brightness': 104,
        'contrast': 154,
        'saturation': 73,
        'focus': 30,
        'sharpness': 253,
        'white_balance': 3141,
    },
    3: {
        'brightness': 128,
        'contrast': 128,
        'saturation': 128,
        'focus': 30,
        'sharpness': 255,
        'white_balance': 3008,
    }
}


# Sol (left) per-camera overrides
CAMERA_SETTINGS_PER_CAMERA_SOL: dict[int, dict] = {
    2: {
        'width': 3840,
        'height': 2160,
        'brightness': 120,
        'contrast': 180,
        'saturation': 35,
        'focus': 64,
        'sharpness': 230,
        'white_balance': 4900,
    },
    0: {
        'brightness': 120,
        'contrast': 160,
        'saturation': 30,
        'focus': 30,
        'sharpness': 225,
        'white_balance': 4900,
    },
    1: {
        'brightness': 120,
        'contrast': 160,
        'saturation': 30,
        'focus': 30,
        'sharpness': 225,
        'white_balance': 4900,
    },
    3: {
        'brightness': 120,
        'contrast': 160,
        'saturation': 30,
        'focus': 30,
        'sharpness': 225,
        'white_balance': 4900,
    }
}
def get_settings_for(camera_idx: int, side: str = "Sol") -> dict:
    """Return merged settings for given camera index and side ('Sol' or 'Sag').
    Defaults to 'Sol'.
    """
    s = dict(CAMERA_SETTINGS)
    if side == "Sag":
        s.update(CAMERA_SETTINGS_PER_CAMERA_SAG.get(camera_idx, {}))
    else:
        s.update(CAMERA_SETTINGS_PER_CAMERA_SOL.get(camera_idx, {}))
    return s


def apply_camera_settings(cap: cv2.VideoCapture, settings: dict) -> None:
    if cap is None:
        return
    width = settings.get('width', CAMERA_WIDTH)
    height = settings.get('height', CAMERA_HEIGHT)
    fps = settings.get('fps', 30)
    autofocus = settings.get('autofocus', False)
    auto_exposure = settings.get('auto_exposure', False)
    auto_white_balance = settings.get('auto_white_balance', False)
    focus = settings.get('focus', 30)
    white_balance = settings.get('white_balance', 4500)
    brightness = settings.get('brightness', 128)
    contrast = settings.get('contrast', 128)
    saturation = settings.get('saturation', 128)
    hue = settings.get('hue', 0)
    sharpness = settings.get('sharpness', 128)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if auto_exposure else 1)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1 if auto_white_balance else 0)
    if not autofocus:
        cap.set(cv2.CAP_PROP_FOCUS, focus)
    if not auto_white_balance:
        # Birçok sürücü 2800-6500 aralığını bekler; aşım durumunda kırp
        try:
            wb_val = int(max(2800, min(6500, int(white_balance))))
        except Exception:
            wb_val = 4500
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, wb_val)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    cap.set(cv2.CAP_PROP_SATURATION, saturation)
    cap.set(cv2.CAP_PROP_HUE, hue)
    cap.set(cv2.CAP_PROP_SHARPNESS, sharpness)


def ensure_data_root() -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, DATA_ROOT)
    os.makedirs(data_root, exist_ok=True)
    return data_root


def list_existing_parts(data_root: str) -> List[Tuple[int, str]]:
    parts: List[Tuple[int, str]] = []
    if not os.path.isdir(data_root):
        return parts
    for name in os.listdir(data_root):
        if not os.path.isdir(os.path.join(data_root, name)):
            continue
        if name.lower().startswith("part"):
            num_str = name[4:]
            try:
                num = int(num_str)
                parts.append((num, name))
            except Exception:
                continue
    parts.sort(key=lambda x: x[0])
    return parts


def next_part_dir(data_root: str) -> Tuple[int, str]:
    parts = list_existing_parts(data_root)
    next_num = 1 if not parts else parts[-1][0] + 1
    name = f"Part{next_num}"
    path = os.path.join(data_root, name)
    os.makedirs(path, exist_ok=True)
    return next_num, path


def get_image_index_start(folder: str) -> int:
    if not os.path.isdir(folder):
        return 1
    max_idx = 0
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        stem = os.path.splitext(fname)[0]
        parts = stem.split("_")
        try:
            candidate: Optional[int] = None
            if len(parts) >= 4 and parts[-2].isdigit():
                candidate = int(parts[-2])
            elif len(parts) >= 3 and parts[-1].isdigit():
                candidate = int(parts[-1])
            if candidate is not None and candidate > max_idx:
                max_idx = candidate
        except Exception:
            pass
    return max_idx + 1


def setup_cameras(camera_indices: List[int]) -> dict[int, cv2.VideoCapture]:
    caps: dict[int, cv2.VideoCapture] = {}
    for idx in camera_indices:
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if USE_DSHOW else cv2.VideoCapture(idx)
        except Exception:
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Uyari: Kamera {idx} acilamadi")
            continue
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        # Ilk acilista varsayilan (default) surucu ayarlariyla calissin; sadece buffer flush yap
        for _ in range(5):
            cap.read()
            time.sleep(0.005)
        caps[idx] = cap
        print(f"[OK] Kamera {idx} varsayilan ayarlarla acildi")
    return caps


def bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)


# ----------------------------
# RT-DETR Ensemble helpers
# ----------------------------
DEVICE = "cuda"
LINE_THICKNESS = 1
FONT_SCALE = 0.6

# Renk Haritası (BGR)
COLORS = {
    'gt': (0, 0, 255),       # Kırmızı - Ground Truth
    'scratch': (0, 255, 0),   # Yeşil - Çizik tahminleri
    'black_dot': (255, 0, 0), # Mavi - Siyah nokta tahminleri
    'ensemble': (0, 255, 255) # Sarı - Ensemble sonucu
}


def calculate_iou(boxA, boxB):
    """
    İki bounding box arasındaki IoU değerini hesapla.
    Box format: [center_x, center_y, width, height] (normalized)
    """
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


def ensemble_predictions(scratch_results, blackdot_results, img_shape, nms_iou=0.5, conf_threshold=DEFAULT_CONF_THRESHOLD):
    """
    İki modelin tahminlerini GERÇEK ENSEMBLE olarak birleştir.
    
    Strateji:
    - İki model aynı yerde (IoU >= nms_iou) aynı sınıfı tespit ederse:
      Skorları ağırlıklı olarak birleştir: (score1 * w1 + score2 * w2)
    - Sadece bir model tespit ettiyse: O modelin skorunu kullan
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
    
    # Her çizik modeli tahmini için siyah nokta modelinde eşleşme ara
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
            # İKİ MODEL DE TESPİT ETTİ - ENSEMBLE SKORU HESAPLA
            best_match['matched'] = True
            sp['matched'] = True
            
            cls_id = sp['cls_id']
            
            if cls_id == 0:  # scratch
                w1 = SCRATCH_MODEL_STRONG_WEIGHT
                w2 = BLACKDOT_MODEL_WEAK_WEIGHT
            else:  # black_dot
                w1 = SCRATCH_MODEL_WEAK_WEIGHT
                w2 = BLACKDOT_MODEL_STRONG_WEIGHT
            
            ensemble_conf = (sp['conf'] * w1 + best_match['conf'] * w2) / (w1 + w2)
            
            ensemble_box = [
                (sp['box'][0] + best_match['box'][0]) / 2,
                (sp['box'][1] + best_match['box'][1]) / 2,
                (sp['box'][2] + best_match['box'][2]) / 2,
                (sp['box'][3] + best_match['box'][3]) / 2
            ]
            
            ensemble_results.append((
                cls_id,
                ensemble_box,
                ensemble_conf,
                'ensemble',
                f"S:{sp['conf']:.3f}*{w1}+B:{best_match['conf']:.3f}*{w2}"
            ))
        else:
            # SADECE ÇİZİK MODELİ TESPİT ETTİ
            sp['matched'] = True
            cls_id = sp['cls_id']
            
            if cls_id == 0:
                weight = SCRATCH_MODEL_STRONG_WEIGHT
            else:
                weight = SCRATCH_MODEL_WEAK_WEIGHT
            
            weighted_conf = sp['conf'] * weight
            
            ensemble_results.append((
                cls_id,
                sp['box'],
                weighted_conf,
                'scratch_only',
                f"S:{sp['conf']:.3f}*{weight}"
            ))
    
    # Eşleşmeyen siyah nokta modeli tahminleri
    for bp in blackdot_preds:
        if not bp['matched']:
            cls_id = bp['cls_id']
            
            if cls_id == 1:
                weight = BLACKDOT_MODEL_STRONG_WEIGHT
            else:
                weight = BLACKDOT_MODEL_WEAK_WEIGHT
            
            weighted_conf = bp['conf'] * weight
            
            ensemble_results.append((
                cls_id,
                bp['box'],
                weighted_conf,
                'blackdot_only',
                f"B:{bp['conf']:.3f}*{weight}"
            ))
    
    # Ağırlıklı confidence threshold altındaki sonuçları filtrele
    filtered_results = [r for r in ensemble_results if r[2] >= conf_threshold]
    
    return filtered_results


def draw_boxes_ensemble(image, ensemble_preds, class_names):
    """
    Ensemble tahminlerini görüntü üzerine çiz.
    """
    h, w = image.shape[:2]

    for pred in ensemble_preds:
        if len(pred) == 5:
            cls_id, box, conf, model_source, details = pred
        else:
            cls_id, box, conf, model_source = pred
            details = ""
        
        x, y, bw, bh = box
        
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        OFFSET = 5
        x1 = max(0, x1 - OFFSET)
        y1 = max(0, y1 - OFFSET)
        x2 = min(w, x2 + OFFSET)
        y2 = min(h, y2 + OFFSET)

        label_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        
        if model_source == 'ensemble':
            color = (0, 255, 255)  # Sarı
            tag = "E"
        elif model_source == 'scratch_only':
            color = COLORS.get(label_name, (0, 255, 0))
            tag = "S"
        elif model_source == 'blackdot_only':
            color = COLORS.get(label_name, (255, 0, 0))
            tag = "B"
        else:
            color = COLORS.get(label_name, COLORS['ensemble'])
            tag = "?"
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, LINE_THICKNESS + 1)
        
        cv2.putText(image, f"[{tag}] {label_name} {conf:.2f}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, LINE_THICKNESS + 1)

    return image


def draw_ok_nok_flag(img_bgr: np.ndarray, has_error: bool) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    text = "NOK" if has_error else "OK"
    color = (0, 0, 255) if has_error else (0, 200, 0)
    scale = max(0.9, min(2.0, h / 480.0))
    thickness = max(2, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = 12, 18 + th
    cv2.rectangle(img_bgr, (x - 8, y - th - 10), (x + tw + 8, y + 6), (0, 0, 0), -1)
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return img_bgr


def load_ensemble_models(device: torch.device):
    """
    İki RT-DETR modelini yükle ve döndür.
    Returns: (scratch_model, blackdot_model)
    """
    if not os.path.exists(SCRATCH_MODEL_PATH):
        raise RuntimeError(f"Çizik modeli bulunamadı: {SCRATCH_MODEL_PATH}")
    
    if not os.path.exists(BLACKDOT_MODEL_PATH):
        raise RuntimeError(f"Siyah nokta modeli bulunamadı: {BLACKDOT_MODEL_PATH}")
    
    scratch_model = RTDETR(SCRATCH_MODEL_PATH)
    if device.type == "cuda":
        scratch_model = scratch_model.to(device)
    
    blackdot_model = RTDETR(BLACKDOT_MODEL_PATH)
    if device.type == "cuda":
        blackdot_model = blackdot_model.to(device)
    
    return scratch_model, blackdot_model


@torch.no_grad()
def analyze_frame_ensemble(scratch_model, blackdot_model, frame_bgr: np.ndarray, 
                           conf_thr: float = DEFAULT_CONF_THRESHOLD) -> Tuple[np.ndarray, list, int, bool]:
    """
    Ensemble analizi yap.
    Returns: (vis_bgr, ensemble_preds, error_count, has_error)
    """
    # Her iki modelden tahmin al
    scratch_results = scratch_model(frame_bgr, conf=conf_thr, verbose=False)
    blackdot_results = blackdot_model(frame_bgr, conf=conf_thr, verbose=False)
    
    # Tahminleri birleştir
    ensemble_preds = ensemble_predictions(
        scratch_results,
        blackdot_results,
        frame_bgr.shape,
        nms_iou=NMS_IOU_THRESHOLD,
        conf_threshold=conf_thr
    )
    
    # Görselleştir
    vis_bgr = frame_bgr.copy()
    vis_bgr = draw_boxes_ensemble(vis_bgr, ensemble_preds, ENSEMBLE_CLASS_NAMES)
    
    has_error = len(ensemble_preds) > 0
    error_count = len(ensemble_preds)
    
    return vis_bgr, ensemble_preds, error_count, has_error


class DataCollectorWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Collector")
        self.data_root = ensure_data_root()
        self.current_part_num, self.current_part_dir = next_part_dir(self.data_root)
        self.next_image_idx = get_image_index_start(self.current_part_dir)

        self.caps = setup_cameras(CAMERA_INDICES)
        if not self.caps:
            raise RuntimeError("Hiçbir kamera açılamadı")

        # Which side/part is selected: 'Sol' or 'Sag'
        self.selected_side = "Sol"

        # Kamera 0 için dinamik focus ayarları
        self.camera0_focus = 5  # Başlangıç focus değeri
        self.focus_timer = QtCore.QTimer(self)
        self.focus_timer.timeout.connect(self.increase_camera0_focus)
        self.focus_timer.setInterval(800)  # 0.5 saniye

        # Auto-exposure ayarlama: once ac, sonra kapat; kapatinca ayarlar uygulanacak
        QtCore.QTimer.singleShot(3000, self.enable_auto_exposure)
        QtCore.QTimer.singleShot(8000, self.disable_auto_exposure)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)

        # Top row: live camera views (4)
        self.view0 = QtWidgets.QLabel()
        self.view1 = QtWidgets.QLabel()
        self.view2 = QtWidgets.QLabel()
        self.view3 = QtWidgets.QLabel()
        # Bottom row: analysis results for each camera (4)
        self.ana0 = QtWidgets.QLabel()
        self.ana1 = QtWidgets.QLabel()
        self.ana2 = QtWidgets.QLabel()
        self.ana3 = QtWidgets.QLabel()

        for v in [self.view0, self.view1, self.view2, self.view3, self.ana0, self.ana1, self.ana2, self.ana3]:
            v.setMinimumSize(320, 180)
            v.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            v.setAlignment(QtCore.Qt.AlignCenter)
            v.setStyleSheet("background-color: black;")

        grid = QtWidgets.QGridLayout()
        # 4 columns, 2 rows -> top row live, bottom row analysis
        grid.addWidget(self.view0, 0, 0)
        grid.addWidget(self.view1, 0, 1)
        grid.addWidget(self.view2, 0, 2)
        grid.addWidget(self.view3, 0, 3)

        grid.addWidget(self.ana0, 1, 0)
        grid.addWidget(self.ana1, 1, 1)
        grid.addWidget(self.ana2, 1, 2)
        grid.addWidget(self.ana3, 1, 3)

        # Make columns stretch evenly
        for c in range(4):
            grid.setColumnStretch(c, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        self.label_status = QtWidgets.QLabel()
        self.label_status.setText("Hazır")

        # Side selector (Sol/Sag)
        self.side_selector = QtWidgets.QComboBox()
        self.side_selector.addItems(["Sol", "Sag"])
        self.side_selector.setCurrentText(self.selected_side)
        self.side_selector.currentTextChanged.connect(self.on_side_changed)

        self.btn_save = QtWidgets.QPushButton("Kaydet")
        self.btn_batch_analyze = QtWidgets.QPushButton("Kayıtları Analiz Et")
        self.btn_next = QtWidgets.QPushButton("Sonraki Part")
        self.btn_quit = QtWidgets.QPushButton("Çıkış")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_batch_analyze.clicked.connect(self.on_analyze_saved)
        self.btn_next.clicked.connect(self.on_next)
        self.btn_quit.clicked.connect(self.close)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.btn_save)
        buttons.addWidget(self.btn_next)
        buttons.addWidget(self.btn_quit)

        # Top controls layout
        top_controls = QtWidgets.QHBoxLayout()
        top_controls.addWidget(QtWidgets.QLabel("Parça:"))
        top_controls.addWidget(self.side_selector)
        top_controls.addStretch()

        right = QtWidgets.QVBoxLayout()
        right.addLayout(top_controls)
        right.addLayout(grid)
        right.addWidget(self.label_status)
        right.addWidget(self.btn_batch_analyze)
        right.addLayout(buttons)

        self.setLayout(right)

        # Overlay banner for operation result
        self.overlay = QtWidgets.QLabel(self)
        self.overlay.setAlignment(QtCore.Qt.AlignCenter)
        self.overlay.setStyleSheet(
            "background-color: rgba(0,0,0,140); color: white; font-size: 36px; padding: 20px; border-radius: 12px;"
        )
        self.overlay.hide()
        self.overlay_timer = QtCore.QTimer(self)
        self.overlay_timer.setSingleShot(True)
        self.overlay_timer.timeout.connect(self.overlay.hide)

        # RT-DETR Ensemble model lazy state
        self._scratch_model = None
        self._blackdot_model = None
        self._ensemble_device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")
        # store last analyzed image path per camera and error counts: {cam_idx: (path, error_count)}
        self._best_analyzed_per_cam = {}

    def apply_all_camera_settings(self) -> None:
        for cam_idx, cap in self.caps.items():
            try:
                settings = get_settings_for(cam_idx, side=self.selected_side)
                # Kamera 0 için dinamik focus değerini kullan
                if cam_idx == 0:
                    settings['focus'] = self.camera0_focus
                apply_camera_settings(cap, settings)
                print(f"[OK] Kamera {cam_idx}: Ayarlar uygulandi -> {settings}")
            except Exception as e:
                print(f"[WARNING] Kamera {cam_idx}: Ayarlar uygulanamadi: {e}")

    def on_side_changed(self, text: str) -> None:
        """Called when user switches between 'Sol' and 'Sag'. Re-apply camera settings."""
        self.selected_side = text
        print(f"[INFO] Seçilen parça değişti: {text}. Kamera ayarları uygulanıyor...")
        # Re-apply settings for all open cameras
        self.apply_all_camera_settings()

    def increase_camera0_focus(self) -> None:
        """Kamera 0 için focus değerini artırır ve fotoğraf çeker"""
        if self.camera0_focus < 30:
            self.camera0_focus += 1
            # Sadece kamera 0'ın focus ayarını güncelle
            if 0 in self.caps:
                cap = self.caps[0]
                try:
                    cap.set(cv2.CAP_PROP_FOCUS, self.camera0_focus)
                    print(f"[OK] Kamera 0: Focus {self.camera0_focus} olarak güncellendi")
                    
                    # Focus ayarlandıktan sonra kısa bir bekleme
                    QtCore.QThread.msleep(300)
                    
                    # Bu focus değeri için fotoğraf çek
                    self.capture_focus_photo()
                    
                except Exception as e:
                    print(f"[WARNING] Kamera 0: Focus güncellenemedi: {e}")
        else:
            # Maksimum değere ulaştığında timer'ı durdur
            self.focus_timer.stop()
            print(f"[INFO] Kamera 0: Focus maksimum değere (16) ulaştı, timer durduruldu")

    def update_frames(self) -> None:
        views = [self.view0, self.view1, self.view2, self.view3]
        for i, cam_idx in enumerate(sorted(self.caps.keys())[:4]):
            cap = self.caps[cam_idx]
            ret, frame = cap.read()
            if not ret:
                continue
            # Draw label before scaling
            cv2.putText(frame, f"Kamera {cam_idx}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            img = bgr_to_qimage(frame)
            target_size = views[i].size()
            pm = QtGui.QPixmap.fromImage(img).scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            views[i].setPixmap(pm)

        self.label_status.setText(
            f"Part{self.current_part_num} | Index: {self.next_image_idx} | Klasör: {os.path.basename(self.current_part_dir)}"
        )

        # Center overlay if visible
        if self.overlay.isVisible():
            g = self.geometry()
            ow = min(int(g.width() * 0.5), 800)
            oh = 100
            self.overlay.setFixedSize(ow, oh)
            self.overlay.move((g.width() - ow) // 2, (g.height() - oh) // 2)

    def on_next(self) -> None:
        self.current_part_num, self.current_part_dir = next_part_dir(self.data_root)
        self.next_image_idx = get_image_index_start(self.current_part_dir)
        self.label_status.setText(f"Yeni klasör: Part{self.current_part_num}")

    def on_save(self) -> None:
        # Tüm kameralar için focus taraması yap ve her değerde fotoğraf çek (±5)
        total_cams = len(self.caps)
        total_saved = 0
        for cam_idx, cap in self.caps.items():
            total_saved += self.sweep_focus_and_capture(cam_idx)

        if total_saved > 0:
            self.next_image_idx += 1
            self.show_banner("Tamamlandı", ok=True)
        else:
            self.show_banner("Hata: Kayıt yok", ok=False)

        self.label_status.setText(
            f"Kaydedildi: {total_saved} görüntü | Kameralar: {total_cams} -> {os.path.basename(self.current_part_dir)}"
        )
    
    def capture_focus_photo(self) -> None:
        """Kamera 0 için mevcut focus değeriyle fotoğraf çeker"""
        if 0 not in self.caps:
            return
            
        cap = self.caps[0]
        # Birkaç frame okuyarak en güncel görüntüyü al
        frame = None
        for _ in range(3):
            ret, frame = cap.read()
            if ret and frame is not None:
                break
            QtCore.QThread.msleep(10)
            
        if frame is None:
            print(f"[WARNING] Kamera 0: Focus {self.camera0_focus} için fotoğraf çekilemedi")
            return
            
        # Focus değerini dosya adına ekle
        raw_dir = os.path.join(self.current_part_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        fname = f"cam_0_focus_{self.camera0_focus:02d}_{self.next_image_idx:05d}.png"
        out_path = os.path.join(raw_dir, fname)
        # PNG formatında kaydet
        ok = cv2.imwrite(out_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        if ok:
            print(f"[OK] Kamera 0: Focus {self.camera0_focus} fotoğrafı kaydedildi -> {fname}")
        else:
            print(f"[ERROR] Kamera 0: Focus {self.camera0_focus} fotoğrafı kaydedilemedi")

    def ensure_model_loaded(self) -> None:
        if self._scratch_model is None or self._blackdot_model is None:
            try:
                self.label_status.setText("RT-DETR Ensemble modelleri yükleniyor...")
                QtWidgets.QApplication.processEvents()
                self._scratch_model, self._blackdot_model = load_ensemble_models(self._ensemble_device)
                self.label_status.setText("Ensemble modeller yüklendi")
                print("[OK] RT-DETR Ensemble modelleri yüklendi")
            except Exception as e:
                self._scratch_model = None
                self._blackdot_model = None
                self.label_status.setText(f"Model yüklenemedi: {e}")
                print(f"[ERROR] Model yüklenemedi: {e}")

    def on_analyze(self) -> None:
        # Live analysis removed. Use batch analysis (on_analyze_saved) to process RAW images.
        # This method is intentionally left empty to avoid accidental live inference.
        return

    def on_analyze_saved(self) -> None:
        """Batch analyze: mevcut Part klasöründeki tüm kayıtlı görüntüleri model ile analiz et
        ve sonuçları `analyzed` alt klasörüne kaydet."""
        # Load ensemble models if needed
        self.ensure_model_loaded()
        if self._scratch_model is None or self._blackdot_model is None:
            self.show_banner("Ensemble modeller yüklenemedi", ok=False)
            return

        part_dir = self.current_part_dir
        if not os.path.isdir(part_dir):
            self.show_banner("Part klasörü bulunamadı", ok=False)
            return

        raw_dir = os.path.join(part_dir, "raw")
        if not os.path.isdir(raw_dir):
            self.show_banner("RAW klasörü bulunamadı", ok=False)
            return

        out_dir = os.path.join(part_dir, "analyzed")
        os.makedirs(out_dir, exist_ok=True)

        # Find image files inside raw/
        files = [f for f in sorted(os.listdir(raw_dir)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not files:
            self.show_banner("Kayıtlı resim yok", ok=False)
            return

        total = 0
        failed = 0
        # reset per-camera bests: store (path, err_count, avg_score)
        self._best_analyzed_per_cam = {}

        # Helper to extract camera index from filename. Falls back to None.
        def _extract_cam_idx(name: str) -> Optional[int]:
            # expected pattern: cam_<idx>_...
            parts = name.split("_")
            try:
                if parts[0].lower().startswith("cam") and parts[1].isdigit():
                    return int(parts[1])
            except Exception:
                pass
            return None

        # Prepare CSV for metrics
        metrics_csv = os.path.join(part_dir, f"analysis_metrics_part{self.current_part_num}.csv")
        csv_exists = os.path.isfile(metrics_csv)
        try:
            csv_fh = open(metrics_csv, "a", newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_fh)
            if not csv_exists:
                csv_writer.writerow(["image_id", "camera_idx", "error_count", "avg_score", "timestamp"])
        except Exception:
            csv_fh = None
            csv_writer = None

        # Process each file
        for fname in files:
            fpath = os.path.join(raw_dir, fname)
            try:
                img = cv2.imread(fpath)
                if img is None:
                    failed += 1
                    continue

                # RT-DETR Ensemble analizi
                vis_bgr, ensemble_preds, err_count, has_error = analyze_frame_ensemble(
                    self._scratch_model, self._blackdot_model, img, DEFAULT_CONF_THRESHOLD
                )
                vis_bgr = draw_ok_nok_flag(vis_bgr, has_error)
                out_name = os.path.splitext(fname)[0] + "_analyzed.png"
                out_path = os.path.join(out_dir, out_name)
                if cv2.imwrite(out_path, vis_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                    total += 1
                    print(f"[OK] Analiz kaydedildi -> {out_name}")
                    # update best per camera by number of detections (errors)
                    cam_idx = _extract_cam_idx(fname)
                    # avg_score: ortalama confidence
                    avg_score = 0.0
                    if ensemble_preds:
                        scores_list = [p[2] for p in ensemble_preds if len(p) >= 3]
                        avg_score = float(np.mean(scores_list)) if scores_list else 0.0
                    # write metrics to CSV if possible
                    if csv_writer is not None:
                        try:
                            csv_writer.writerow([os.path.basename(fpath), cam_idx if cam_idx is not None else "", err_count, f"{avg_score:.4f}", datetime.utcnow().isoformat()])
                        except Exception:
                            pass
                    if cam_idx is not None:
                        prev = self._best_analyzed_per_cam.get(cam_idx)
                        # choose image with larger error count; tie-breaker higher avg_score
                        if prev is None or err_count > prev[1] or (err_count == prev[1] and avg_score > prev[2]):
                            # store path, err_count, avg_score
                            self._best_analyzed_per_cam[cam_idx] = (out_path, err_count, avg_score)
                else:
                    failed += 1
                    print(f"[ERROR] Analiz kaydedilemedi -> {out_name}")

                # Update status every 5 images
                if total % 5 == 0:
                    self.label_status.setText(f"Analiz ediliyor: {total}/{len(files)}")
                    QtWidgets.QApplication.processEvents()

            except Exception as e:
                failed += 1
                print(f"[ERROR] Analiz sırasında hata ({fname}): {e}")

        # close csv
        try:
            if csv_fh is not None:
                csv_fh.close()
        except Exception:
            pass

        msg = f"Analiz tamam: {total} başarılı, {failed} hatalı"
        self.show_banner(msg, ok=(total > 0))
        self.label_status.setText(msg)

        # After analysis, update bottom-row analysis labels with the worst image per camera
        ana_widgets = {0: self.ana0, 1: self.ana1, 2: self.ana2, 3: self.ana3}
        for cam_idx, widget in ana_widgets.items():
            best = self._best_analyzed_per_cam.get(cam_idx)
            if best is None:
                # clear / show placeholder
                widget.setPixmap(QtGui.QPixmap())
                widget.setText("No data")
                widget.setStyleSheet("background-color: #222; color: #ccc;")
            else:
                path, cnt, avg_score = best
                img = cv2.imread(path)
                if img is None:
                    widget.setPixmap(QtGui.QPixmap())
                    widget.setText("Err")
                    continue
                qimg = bgr_to_qimage(img)
                pm = QtGui.QPixmap.fromImage(qimg).scaled(widget.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                widget.setPixmap(pm)
                widget.setToolTip(f"{os.path.basename(path)} | errors: {cnt} | avg_score: {avg_score:.3f}")

    def get_baseline_focus(self, cam_idx: int) -> int:
        """Her kamera için başlangıç (baseline) focus:
        1) CAMERA_SETTINGS_PER_CAMERA içindeki focus varsa onu kullan
        2) Yoksa cihazın mevcut CAP_PROP_FOCUS değerini dene
        3) Son çare 30
        """
        # pick the appropriate per-camera overrides depending on selected side
        if self.selected_side == "Sag":
            settings_focus = CAMERA_SETTINGS_PER_CAMERA_SAG.get(cam_idx, {}).get('focus')
        else:
            settings_focus = CAMERA_SETTINGS_PER_CAMERA_SOL.get(cam_idx, {}).get('focus')
        if settings_focus is not None:
            base = int(settings_focus)
        else:
            base = 30
            try:
                cap = self.caps.get(cam_idx)
                if cap is not None:
                    current = int(cap.get(cv2.CAP_PROP_FOCUS))
                    if current > 0:
                        base = current
            except Exception:
                pass
        # Makul sınırlar içinde tut (UVC sürücüler için 0-255 aralığı yaygın)
        return max(0, min(255, int(base)))

    def sweep_focus_and_capture(self, cam_idx: int) -> int:
        """Belirtilen kamera için focus'u ±5 aralığında tarar ve her değerde fotoğraf çeker."""
        if cam_idx not in self.caps:
            return 0
        cap = self.caps[cam_idx]

        base = self.get_baseline_focus(cam_idx)
        min_f = max(0, base - 3)
        max_f = min(255, base + 3)

        saved = 0
        # Önce -5'ten başlayarak base'e kadar, sonra base'den +5'e kadar
        focus_values = list(range(min_f, base)) + list(range(base, max_f + 1))
        for fval in focus_values:
            try:
                cap.set(cv2.CAP_PROP_FOCUS, fval)
                # odaklama motorunun oturması için kısa bekleme
                QtCore.QThread.msleep(120)
            except Exception:
                pass

            # taze kare al
            frame = None
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
                QtCore.QThread.msleep(10)
            if frame is None:
                continue

            raw_dir = os.path.join(self.current_part_dir, "raw")
            os.makedirs(raw_dir, exist_ok=True)
            fname = f"cam_{cam_idx}_focus_{fval:02d}_{self.next_image_idx:05d}.png"
            out_path = os.path.join(raw_dir, fname)
            if cv2.imwrite(out_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                saved += 1
                print(f"[OK] Kamera {cam_idx}: Focus {fval} fotoğrafı kaydedildi -> {fname}")
            else:
                print(f"[ERROR] Kamera {cam_idx}: Focus {fval} fotoğrafı kaydedilemedi")

        return saved
    
    def show_banner(self, text: str, ok: bool) -> None:
        color = "#28a745" if ok else "#dc3545"
        self.overlay.setText(text)
        self.overlay.setStyleSheet(
            f"background-color: rgba(0,0,0,140); color: white; font-size: 36px; padding: 20px; border: 3px solid {color}; border-radius: 12px;"
        )
        self.overlay.show()
        # Position immediately
        g = self.geometry()
        ow = min(int(g.width() * 0.5), 800)
        oh = 100
        self.overlay.setFixedSize(ow, oh)
        self.overlay.move((g.width() - ow) // 2, (g.height() - oh) // 2)
        self.overlay_timer.start(1500)

    def enable_auto_exposure(self) -> None:
        """Tüm kameralar için auto-exposure'ı açar"""
        for cam_idx, cap in self.caps.items():
            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 3 = auto mode
                print(f"[OK] Kamera {cam_idx}: Auto-exposure acildi")
            except Exception as e:
                print(f"[WARNING] Kamera {cam_idx}: Auto-exposure acilamadi: {e}")

    def disable_auto_exposure(self) -> None:
        """Tüm kameralar için auto-exposure'ı kapatır"""
        for cam_idx, cap in self.caps.items():
            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode
                # Bazı sürücüler manuel moda geçişten sonra kısa bir bekleme ister
                QtCore.QThread.msleep(200)
                # Auto-exposure kapatildiktan sonra mevcut ayarlari uygula ve logla
                settings = get_settings_for(cam_idx, side=self.selected_side)
                apply_camera_settings(cap, settings)
                QtCore.QThread.msleep(100)
                apply_camera_settings(cap, settings)
                print(f"[OK] Kamera {cam_idx}: Auto-exposure kapatildi, manuel mod aktif. Uygulanan ayarlar: {settings}")
            except Exception as e:
                print(f"[WARNING] Kamera {cam_idx}: Auto-exposure kapatilamadi: {e}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        for cap in self.caps.values():
            try:
                cap.release()
            except Exception:
                pass
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = DataCollectorWindow()
    w.showFullScreen()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


