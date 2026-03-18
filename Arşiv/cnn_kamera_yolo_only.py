import os
import time
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ----------------------------
# Config (edit here)
# ----------------------------
DEVICE = "cuda"
CAMERA_INDICES = [0,1,2,3]
USE_DSHOW = True
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# Ana pencere ve grid ayarları (üst canlı, alt analiz)
MAIN_WIN_W = 1920
MAIN_WIN_H = 1080
TOP_H = 540
BOT_H = 540
TILE_W = 480
TILE_H_TOP = 540
TILE_H_BOT = 540

WIN_MAIN = "FourCamsMain_YOLO"
WIN_CTRL = "CameraControl_YOLO"

# YOLO model ayarları
YOLO_MODEL_PATH = os.path.join("Modeller", "Lens_v2_1400", "weights", "best.pt")
YOLO_CONF = 0.15
YOLO_IMGSZ = 1024


def load_yolo(device_str: str) -> YOLO:
    model = YOLO(YOLO_MODEL_PATH)
    return model


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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        caps[idx] = cap
        print(f"✓ Kamera {idx} açıldı")
    return caps


def yolo_annotate_and_flag(yolo: YOLO, frame_bgr: np.ndarray, device_str: str) -> Tuple[np.ndarray, bool]:
    device_arg = 0 if (device_str == "cuda") else "cpu"
    results = yolo.predict(
        source=frame_bgr,
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        device=device_arg,
        verbose=False,
        augment=True,
    )
    vis = frame_bgr.copy()
    has_error = False
    if len(results) > 0 and getattr(results[0], "boxes", None) is not None and results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        names = results[0].names if hasattr(results[0], "names") else {}
        has_error = True
        for (x1,y1,x2,y2), c, k in zip(boxes, confs, clss):
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            name = names.get(int(k), str(int(k))) if isinstance(names, dict) else str(int(k))
            label = f"{name} {float(c):.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tx, ty = int(x1), max(int(y1)-8, th+4)
            cv2.rectangle(vis, (tx-2, ty-th-4), (tx+tw+2, ty+2), (0,0,0), -1)
            cv2.putText(vis, label, (tx,ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    vis = draw_ok_nok_flag(vis, has_error)
    return vis, has_error


def process_cameras_yolo_only(camera_indices: List[int] = CAMERA_INDICES, device_str: str = DEVICE) -> None:
    yolo = load_yolo(device_str)
    caps = setup_cameras(camera_indices)
    if not caps:
        raise RuntimeError("Hiçbir kamera açılamadı")

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, MAIN_WIN_W, MAIN_WIN_H)
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 800, 600)

    try:
        analysis_tiles: dict[int, np.ndarray] = {}
        while True:
            # Üst satır canlı grid
            live_tiles = []
            for cam_idx, cap in caps.items():
                ret, frame = cap.read()
                if not ret:
                    tile = np.zeros((TILE_H_TOP, TILE_W, 3), dtype=np.uint8)
                else:
                    tile = cv2.resize(frame, (TILE_W, TILE_H_TOP))
                    cv2.putText(tile, f"Cam {cam_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                live_tiles.append(tile)
            while len(live_tiles) < 4:
                live_tiles.append(np.zeros((TILE_H_TOP, TILE_W, 3), dtype=np.uint8))
            live_row = np.hstack(live_tiles[:4])

            # Alt satır analiz grid (son analiz)
            analysis_row = np.hstack([analysis_tiles.get(i, np.zeros((TILE_H_BOT, TILE_W, 3), dtype=np.uint8)) for i in range(4)])

            main_canvas = np.vstack([
                cv2.resize(live_row, (MAIN_WIN_W, TOP_H)),
                cv2.resize(analysis_row, (MAIN_WIN_W, BOT_H)),
            ])
            cv2.imshow(WIN_MAIN, main_canvas)

            info_img = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(info_img, "YOLO ONLY", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.putText(info_img, "'a' - Tum kameralardan analiz", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(info_img, "'q' - Cikis", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow(WIN_CTRL, info_img)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                print("YOLO analiz başlıyor...")
                i = 0
                for cam_idx, cap in caps.items():
                    ret, frame = cap.read()
                    if not ret:
                        analysis_tiles[i] = np.zeros((TILE_H_BOT, TILE_W, 3), dtype=np.uint8)
                        i += 1
                        continue
                    vis, has_error = yolo_annotate_and_flag(yolo, frame, device_str)
                    analysis_tiles[i] = cv2.resize(vis, (TILE_W, TILE_H_BOT))
                    i += 1
                print("YOLO analiz bitti.")
    finally:
        for cap in caps.values():
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()


def main() -> None:
    process_cameras_yolo_only(camera_indices=CAMERA_INDICES, device_str=DEVICE)


if __name__ == "__main__":
    main()


