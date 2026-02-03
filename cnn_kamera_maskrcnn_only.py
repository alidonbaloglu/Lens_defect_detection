import os
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights


# ----------------------------
# Config (edit here)
# ----------------------------
VIDEO_NAME = "Videolar/Kabin/cizik-6.mp4"
SCORE_THR = 0.5
DEVICE = "cuda"
OUTPUT_DIR = os.path.join("results", "MaskRCNN", "video_preds")
OUTPUT_DIR_LIVE = os.path.join("results", "MaskRCNN", "live_preds")
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

# Window names (ASCII-safe)
WIN_MAIN = "FourCamsMain_CNN"
WIN_CTRL = "CameraControl_CNN"

# ----------------------------
# Model utils (MaskRCNN_COCO_sinif_v1)
# ----------------------------
NUM_CLASSES = 8  # background + 7 classes

# Class names (index-aligned with model labels)
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
        # Build label text from model class and score
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


@torch.no_grad()
def process_cameras_cnn_only(camera_indices: List[int] = CAMERA_INDICES, score_thr: float = SCORE_THR, device_str: str = DEVICE) -> None:
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    model = load_model(device)

    caps = setup_cameras(camera_indices)
    if not caps:
        raise RuntimeError("Hiçbir kamera açılamadı")

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, MAIN_WIN_W, MAIN_WIN_H)
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 800, 600)

    try:
        frame_count = 0
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
            # Tam 4 karo yap
            while len(live_tiles) < 4:
                live_tiles.append(np.zeros((TILE_H_TOP, TILE_W, 3), dtype=np.uint8))
            live_row = np.hstack(live_tiles[:4])

            # Alt satır analiz grid (son analizden)
            analysis_tiles_list = []
            for i in range(4):
                analysis_tiles_list.append(analysis_tiles.get(i, np.zeros((TILE_H_BOT, TILE_W, 3), dtype=np.uint8)))
            analysis_row = np.hstack(analysis_tiles_list)

            main_canvas = np.vstack([
                cv2.resize(live_row, (MAIN_WIN_W, TOP_H)),
                cv2.resize(analysis_row, (MAIN_WIN_W, BOT_H)),
            ])
            cv2.imshow(WIN_MAIN, main_canvas)

            info_img = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(info_img, "CNN ONLY - MASK R-CNN", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.putText(info_img, "'a' - Tum kameralardan analiz", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(info_img, "'q' - Cikis", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow(WIN_CTRL, info_img)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                # Tüm kameralardan anlık analiz yap, alt satırı güncelle
                print("Analiz başlıyor...")
                i = 0
                for cam_idx, cap in caps.items():
                    ret, frame = cap.read()
                    if not ret:
                        analysis_tiles[i] = np.zeros((TILE_H_BOT, TILE_W, 3), dtype=np.uint8)
                        i += 1
                        continue
                    vis, boxes, masks, labels_idx, has_error = analyze_frame(model, frame, score_thr)
                    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                    vis_bgr = draw_ok_nok_flag(vis_bgr, has_error)
                    analysis_tiles[i] = cv2.resize(vis_bgr, (TILE_W, TILE_H_BOT))
                    i += 1
                print("Analiz bitti.")
                frame_count += 1
    finally:
        for cap in caps.values():
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()


def main() -> None:
    process_cameras_cnn_only(camera_indices=CAMERA_INDICES, score_thr=SCORE_THR, device_str=DEVICE)


if __name__ == "__main__":
    main()


