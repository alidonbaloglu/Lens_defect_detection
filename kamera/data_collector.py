import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# Config (edit here)
# ----------------------------
DATA_ROOT = os.path.join("results", "DataCollection")
CAMERA_INDICES = [0, 1, 2, 3]
USE_DSHOW = True  # on Windows, DirectShow backend reduces latency
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# Main window grid layout
MAIN_WIN_W = 1920
MAIN_WIN_H = 1080
TOP_H = 1080
TILE_W = 480
TILE_H = 540
WIN_MAIN = "DataCollector"


# ----------------------------
# Per-camera settings (copied style from other module)
# ----------------------------
CAMERA_SETTINGS = {
    'width': CAMERA_WIDTH,
    'height': CAMERA_HEIGHT,
    'fps': 30,
    'brightness': 128,
    'contrast': 128,
    'saturation': 128,
    'hue': 0,
    'gain': 50,
    'exposure': 150,
    'focus': 30,
    'sharpness': 128,
    'white_balance': 4500,
    'autofocus': False,
    'auto_exposure': False,
    'auto_white_balance': False,
}

# Optional per-camera overrides: { cam_index: { 'exposure': 120, ... } }
CAMERA_SETTINGS_PER_CAMERA: dict[int, dict] = {
    0:{
        'brightness':88,
        'contrast':126,
        'saturation':122,
        'hue':20,
        'focus': 12,
        'sharpness': 255,
        'white_balance': 3008,
    },
    1:{
        'brightness':128,
        'contrast':155,
        'saturation':59,
        'focus': 30,
        'sharpness': 255,
        'white_balance': 2635,
    },
    2:{
        'brightness':104,
        'contrast':154,
        'saturation':73,
        'focus': 30,
        'sharpness': 253,
        'white_balance': 3141,
    },
    3:{
        'brightness':56,
        'contrast':52,
        'saturation':148,
        'focus': 14,
        'sharpness': 255,
        'white_balance': 3008,
    }
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
    exposure = settings.get('exposure', 150)
    white_balance = settings.get('white_balance', 4500)
    brightness = settings.get('brightness', 128)
    contrast = settings.get('contrast', 128)
    saturation = settings.get('saturation', 128)
    hue = settings.get('hue', 0)
    gain = settings.get('gain', 50)
    sharpness = settings.get('sharpness', 128)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if auto_exposure else 1)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1 if auto_white_balance else 0)
    if not autofocus:
        cap.set(cv2.CAP_PROP_FOCUS, focus)
    if not auto_exposure:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    if not auto_white_balance:
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, white_balance)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    cap.set(cv2.CAP_PROP_SATURATION, saturation)
    cap.set(cv2.CAP_PROP_HUE, hue)
    cap.set(cv2.CAP_PROP_GAIN, gain)
    cap.set(cv2.CAP_PROP_SHARPNESS, sharpness)


# ----------------------------
# Camera utils
# ----------------------------
def setup_cameras(camera_indices: List[int]) -> dict[int, cv2.VideoCapture]:
    caps: dict[int, cv2.VideoCapture] = {}
    for idx in camera_indices:
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if USE_DSHOW else cv2.VideoCapture(idx)
        except Exception:
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Uyarı: Kamera {idx} açılamadı")
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
        apply_camera_settings(cap, get_settings_for(idx))
        for _ in range(5):
            cap.read()
            time.sleep(0.005)
        caps[idx] = cap
        print(f"✓ Kamera {idx} açıldı ve ayarlar uygulandı")
    return caps


# ----------------------------
# Part folder helpers
# ----------------------------
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
        # expected patterns:
        #   cam_<idx>_<nnnn>
        #   cam_<idx>_<nnnn>_<shot>
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


# ----------------------------
# Main loop
# ----------------------------
def draw_info(canvas: np.ndarray, text_lines: List[str]) -> np.ndarray:
    out = canvas.copy()
    y = 30
    for line in text_lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 30
    return out


def run() -> None:
    data_root = ensure_data_root()
    caps = setup_cameras(CAMERA_INDICES)
    if not caps:
        raise RuntimeError("Hiçbir kamera açılamadı")

    current_part_num, current_part_dir = next_part_dir(data_root)
    next_image_idx = get_image_index_start(current_part_dir)

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, MAIN_WIN_W, MAIN_WIN_H)

    try:
        while True:
            frames = {}
            for cam_idx, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    frame_resized = cv2.resize(frame, (TILE_W, TILE_H))
                    cv2.putText(frame_resized, f"Kamera {cam_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    frames[cam_idx] = frame_resized

            if len(frames) >= 4:
                row1 = np.hstack([
                    frames.get(0, np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)),
                    frames.get(1, np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)),
                    frames.get(2, np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)),
                    frames.get(3, np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)),
                ])
                live_grid = row1
            elif len(frames) >= 2:
                tiles = [frames[k] for k in sorted(frames.keys())]
                live_grid = np.hstack(tiles)
            elif len(frames) == 1:
                live_grid = frames[list(frames.keys())[0]]
            else:
                live_grid = np.zeros((TILE_H, TILE_W * 4, 3), dtype=np.uint8)

            canvas = cv2.resize(live_grid, (MAIN_WIN_W, TOP_H))
            info_lines = [
                "VERI TOPLAMA MODU",
                f"Aktif klasor: Part{current_part_num} ({os.path.basename(current_part_dir)})",
                f"Kayit index: {next_image_idx}",
                "Tuslar: 's' = kaydet, 'n' = sonraki Part, 'q' = cikis",
                f"Klasor: {os.path.relpath(current_part_dir, os.path.dirname(os.path.abspath(__file__)))}",
            ]
            overlay = draw_info(canvas, info_lines)
            cv2.imshow(WIN_MAIN, overlay)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                current_part_num, current_part_dir = next_part_dir(data_root)
                next_image_idx = get_image_index_start(current_part_dir)
                print(f"Yeni klasor: {current_part_dir}")
            elif key == ord('s'):
                saved = 0
                for cam_idx, cap in caps.items():
                    for shot in range(1, 4):  # 3 shots per camera
                        ret, frame = cap.read()
                        if not ret:
                            print(f"  Kamera {cam_idx}: frame okunamadı (shot {shot})")
                            continue
                        fname = f"cam_{cam_idx}_{next_image_idx:05d}_{shot}.jpg"
                        out_path = os.path.join(current_part_dir, fname)
                        cv2.imwrite(out_path, frame)
                        saved += 1
                        cv2.waitKey(50)
                if saved > 0:
                    next_image_idx += 1
                print(f"Kaydedildi: {saved} goruntu -> {current_part_dir}")

    finally:
        for cap in caps.values():
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("\n✓ Kameralar kapatıldı.")


def main() -> None:
    run()


if __name__ == "__main__":
    main()


