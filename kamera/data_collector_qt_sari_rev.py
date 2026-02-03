import os
import sys
import time
from typing import List, Tuple, Optional
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


# ----------------------------
# Config (edit here)
# ----------------------------
DATA_ROOT = os.path.join("results", "DataCollection")
CAMERA_INDICES = [0,1,2,3]
USE_DSHOW = True
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

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
        'brightness': 0,
        'contrast': 255,
        'saturation': 114,
        'focus': 110,
        'sharpness': 255,
        'white_balance': 3025,
    },
    0: {
        'brightness': 128,
        'contrast': 155,
        'saturation': 124,
        'focus': 30,
        'sharpness': 255,
        'white_balance': 3025,
    },
    1: {
        'brightness': 129,
        'contrast': 154,
        'saturation': 127,
        'focus': 30,
        'sharpness': 255,
        'white_balance': 3025,
    },
    3: {
        'brightness': 101,
        'contrast': 128,
        'saturation': 158,
        'focus': 30,
        'sharpness': 255,
        'white_balance': 3053,
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
# Mask R-CNN helpers (copied/adapted from cnn_kamera_maskrcnn_only.py)
# ----------------------------
SCORE_THR = 0.5
DEVICE = "cuda"
NUM_CLASSES = 8  # background + 7 classes
CLASS_NAMES = [
    "background",
    "defect",
    "cizik",
    "enjeksiyon_noktasi",
    "kirik",
    "siyah_nokta",
    "siyahlk",
    "yabanci",
]


def build_model(num_classes: int = NUM_CLASSES) -> torchvision.models.detection.MaskRCNN:
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=ResNet101_Weights.DEFAULT,
        trainable_layers=3,
    )
    model = torchvision.models.detection.MaskRCNN(backbone=backbone, num_classes=num_classes)
    return model


def load_model(device: torch.device) -> torchvision.models.detection.MaskRCNN:
    project_root = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(project_root, "..", "Modeller", "MaskRCNN_COCO_sinif_v1.pt")
    # fallback to project-local Modeller if relative path different
    if not os.path.exists(weights_path):
        weights_path = os.path.join(project_root, "Modeller", "MaskRCNN_COCO_sinif_v1.pt")
    ckpt = torch.load(weights_path, map_location=device)
    num_classes = ckpt.get("num_classes", NUM_CLASSES)
    model = build_model(num_classes)
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except Exception:
        model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()
    return model


def draw_predictions(rgb: np.ndarray, boxes: np.ndarray, masks: np.ndarray, scores: np.ndarray, labels_idx: Optional[np.ndarray] = None) -> np.ndarray:
    overlay = rgb.copy()
    h, w = rgb.shape[:2]
    font_scale = max(0.7, min(2.5, h / 720.0 * 1.2))
    thickness = max(2, int(round(font_scale * 2)))

    # Masks
    for m in (masks if masks.size > 0 else []):
        m_bin = (m[0] >= 0.5).astype(np.uint8)
        color = (0, 255, 0)
        contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=2)
        overlay[m_bin == 1] = (overlay[m_bin == 1] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

    # Boxes and labels
    for idx, ((x1, y1, x2, y2), sc) in enumerate(zip(boxes, scores)):
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        lbl_txt = ""
        if labels_idx is not None and idx < len(labels_idx):
            cls_id = int(labels_idx[idx])
            name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
            lbl_txt = f"{name} {float(sc):.2f}"
        else:
            lbl_txt = f"{float(sc):.2f}"
        (tw, th), _ = cv2.getTextSize(lbl_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx, ty = int(x1), max(int(y1) - 8, th + 4)
        cv2.rectangle(overlay, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(overlay, lbl_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    blended = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)
    return blended


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


@torch.no_grad()
def analyze_frame(model, frame_bgr: np.ndarray, score_thr: float = SCORE_THR) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_t = F.to_tensor(rgb).to(next(model.parameters()).device)

    outputs = model([img_t])
    out = outputs[0]

    scores_all = out.get("scores", torch.empty(0)).detach().cpu().numpy()
    boxes_all = out.get("boxes", torch.empty(0, 4)).detach().cpu().numpy()
    masks_all = out.get("masks", torch.empty(0, 1, rgb.shape[0], rgb.shape[1])).detach().cpu().numpy()
    labels_all = out.get("labels", torch.empty(0)).detach().cpu().numpy()

    keep = scores_all >= score_thr if scores_all.size > 0 else np.array([], dtype=bool)
    boxes = boxes_all[keep] if boxes_all.size > 0 else boxes_all
    masks = masks_all[keep] if masks_all.size > 0 else masks_all
    scores = scores_all[keep] if scores_all.size > 0 else scores_all
    labels_idx = labels_all[keep] if labels_all.size > 0 else labels_all

    vis = draw_predictions(rgb, boxes, masks, scores, labels_idx)
    has_error = boxes.shape[0] > 0
    return vis, boxes, masks, labels_idx, has_error


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

        self.view0 = QtWidgets.QLabel()
        self.view1 = QtWidgets.QLabel()
        self.view2 = QtWidgets.QLabel()
        self.view3 = QtWidgets.QLabel()
        for v in [self.view0, self.view1, self.view2, self.view3]:
            v.setMinimumSize(320, 180)
            v.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            v.setAlignment(QtCore.Qt.AlignCenter)
            v.setStyleSheet("background-color: black;")

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.view0, 0, 0)
        grid.addWidget(self.view1, 0, 1)
        grid.addWidget(self.view2, 1, 0)
        grid.addWidget(self.view3, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
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

        # CNN model lazy state
        self._cnn_model = None
        self._cnn_device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")

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
        if self._cnn_model is None:
            try:
                self.label_status.setText("Model yükleniyor...")
                QtWidgets.QApplication.processEvents()
                self._cnn_model = load_model(self._cnn_device)
                self.label_status.setText("Model yüklendi")
                print("[OK] CNN model yüklendi")
            except Exception as e:
                self._cnn_model = None
                self.label_status.setText(f"Model yüklenemedi: {e}")
                print(f"[ERROR] Model yüklenemedi: {e}")

    def on_analyze(self) -> None:
        # Live analysis removed. Use batch analysis (on_analyze_saved) to process RAW images.
        # This method is intentionally left empty to avoid accidental live inference.
        return

    def on_analyze_saved(self) -> None:
        """Batch analyze: mevcut Part klasöründeki tüm kayıtlı görüntüleri model ile analiz et
        ve sonuçları `analyzed` alt klasörüne kaydet."""
        # Load model if needed
        self.ensure_model_loaded()
        if self._cnn_model is None:
            self.show_banner("Model yok", ok=False)
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
        # Process each file
        for fname in files:
            fpath = os.path.join(raw_dir, fname)
            try:
                img = cv2.imread(fpath)
                if img is None:
                    failed += 1
                    continue

                vis_rgb, boxes, masks, labels_idx, has_error = analyze_frame(self._cnn_model, img, SCORE_THR)
                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                vis_bgr = draw_ok_nok_flag(vis_bgr, has_error)

                out_name = os.path.splitext(fname)[0] + "_analyzed.png"
                out_path = os.path.join(out_dir, out_name)
                if cv2.imwrite(out_path, vis_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                    total += 1
                    print(f"[OK] Analiz kaydedildi -> {out_name}")
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

        msg = f"Analiz tamam: {total} başarılı, {failed} hatalı"
        self.show_banner(msg, ok=(total > 0))
        self.label_status.setText(msg)

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
        min_f = max(0, base - 5)
        max_f = min(255, base + 5)

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


