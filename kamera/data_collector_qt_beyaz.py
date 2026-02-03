import os
import sys
import time
from typing import List, Tuple, Optional

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
    'focus': 30,
    'sharpness': 255,
    'white_balance': 7000,
    'autofocus': False,
    'auto_exposure': False,
    'auto_white_balance': False,
}

CAMERA_SETTINGS_PER_CAMERA: dict[int, dict] = {
    0:{
        'focus': 16,
        #'sharpness': 255,
        #'white_balance': 7000,
    },
    
}



def get_settings_for(camera_idx: int) -> dict:
    s = dict(CAMERA_SETTINGS)
    s.update(CAMERA_SETTINGS_PER_CAMERA.get(camera_idx, {}))
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

        self.btn_save = QtWidgets.QPushButton("Kaydet")
        self.btn_next = QtWidgets.QPushButton("Sonraki Part")
        self.btn_quit = QtWidgets.QPushButton("Çıkış")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_next.clicked.connect(self.on_next)
        self.btn_quit.clicked.connect(self.close)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.btn_save)
        buttons.addWidget(self.btn_next)
        buttons.addWidget(self.btn_quit)

        right = QtWidgets.QVBoxLayout()
        right.addLayout(grid)
        right.addWidget(self.label_status)
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

    def apply_all_camera_settings(self) -> None:
        for cam_idx, cap in self.caps.items():
            try:
                settings = get_settings_for(cam_idx)
                apply_camera_settings(cap, settings)
                print(f"[OK] Kamera {cam_idx}: Ayarlar uygulandi -> {settings}")
            except Exception as e:
                print(f"[WARNING] Kamera {cam_idx}: Ayarlar uygulanamadi: {e}")

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
        total_expected = len(self.caps)
        saved = 0
        # Per-camera: capture 1 shot
        for cam_idx, cap in self.caps.items():
            # Try a few reads in case of immediate failure
            frame = None
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
                QtCore.QThread.msleep(10)
            if frame is None:
                continue
            fname = f"cam_{cam_idx}_{self.next_image_idx:05d}.jpg"
            out_path = os.path.join(self.current_part_dir, fname)
            ok = cv2.imwrite(out_path, frame)
            if ok:
                saved += 1
        if saved == total_expected and saved > 0:
            self.next_image_idx += 1
            self.show_banner("Tamamlandı", ok=True)
        else:
            if saved > 0:
                self.next_image_idx += 1
            self.show_banner("Hata: Eksik kayıt", ok=False)
        self.label_status.setText(f"Kaydedildi: {saved}/{total_expected} görüntü -> {os.path.basename(self.current_part_dir)}")

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
                settings = get_settings_for(cam_idx)
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


