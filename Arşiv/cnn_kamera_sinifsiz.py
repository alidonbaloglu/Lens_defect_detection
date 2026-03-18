import os
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights
from ultralytics import YOLO


# ----------------------------
# Config (edit here)
# ----------------------------
VIDEO_NAME = "Videolar/Kabin/cizik-6.mp4"  # only filename; will be searched recursively in project folders
SCORE_THR = 0.5
DEVICE = "cuda"  # or "cpu"
OUTPUT_DIR = os.path.join("results", "MaskRCNN", "video_preds")
OUTPUT_DIR_LIVE = os.path.join("results", "MaskRCNN", "live_preds")
LIVE_CROPS_DIR = os.path.join("results", "MaskRCNN", "crops_live")
CAMERA_INDICES = [0,1,2,3]
USE_DSHOW = True  # on Windows, DirectShow backend reduces latency
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
# Ana pencere ve grid ayarları (üstte canlı, altta analiz)
MAIN_WIN_W = 1920
MAIN_WIN_H = 1080
TOP_H = 540
BOT_H = 540
# Four tiles per row
TILE_W = 480
TILE_H_TOP = 540
TILE_H_BOT = 540
# YOLO kaldırıldı
# Window names (ASCII-safe to avoid platform issues)
WIN_MAIN = "FourCamsMain"          # 1920x1080; üst canlı, alt analiz
WIN_CTRL = "CameraControl"

# ----------------------------
# Per-camera settings (similar to kamera/multi_camera.py)
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

# Kamera bazlı başlangıç ayarları: { cam_index: {override_key: value} }
# Örnek:
# CAMERA_SETTINGS_PER_CAMERA = {
#     0: { 'exposure': 120, 'focus': 40 },
#     1: { 'autofocus': True },
# }
CAMERA_SETTINGS_PER_CAMERA: dict[int, dict] = {}


def get_settings_for(camera_idx: int) -> dict:
    s = dict(CAMERA_SETTINGS)
    s.update(CAMERA_SETTINGS_PER_CAMERA.get(camera_idx, {}))
    return s


def apply_camera_settings(cap: cv2.VideoCapture, settings: dict) -> None:
    if cap is None:
        return
    # Read with safe defaults in case keys are missing
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

    # Çözünürlük ve FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # Otomatik özellikler
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
    # 3 = auto, 1 = manual (OpenCV/DirectShow)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if auto_exposure else 1)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1 if auto_white_balance else 0)
    # Manuel ayarlar (otomatik kapalıysa)
    if not autofocus:
        cap.set(cv2.CAP_PROP_FOCUS, focus)
    if not auto_exposure:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    if not auto_white_balance:
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, white_balance)
    # Diğerleri
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    cap.set(cv2.CAP_PROP_SATURATION, saturation)
    cap.set(cv2.CAP_PROP_HUE, hue)
    cap.set(cv2.CAP_PROP_GAIN, gain)
    cap.set(cv2.CAP_PROP_SHARPNESS, sharpness)

# ----------------------------
# Model utils (single-class Mask R-CNN)
# ----------------------------
# Tek sınıf (background + 1 sınıf): NUM_CLASSES = 2
NUM_CLASSES = 3


def build_model(num_classes: int = NUM_CLASSES) -> torchvision.models.detection.MaskRCNN:
    # Build Mask R-CNN with ResNet-101 FPN backbone to match training
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=ResNet101_Weights.DEFAULT,
        trainable_layers=3,
    )
    model = torchvision.models.detection.MaskRCNN(backbone=backbone, num_classes=num_classes)
    return model


def load_model(device: torch.device) -> torchvision.models.detection.MaskRCNN:
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Tek sınıflı ağırlık (ResNet-101, background + defect)
    weights_path = os.path.join(project_root, "Modeller", "MaskRCNN_COCO_v4.pt")
    ckpt = torch.load(weights_path, map_location=device)

    # Determine number of classes from checkpoint if available (fallback to 8)
    num_classes = ckpt.get("num_classes", NUM_CLASSES)
    model = build_model(num_classes)

    # Prefer strict loading; if it fails (e.g., minor key mismatch), fall back to non-strict
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except Exception:
        model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device).eval()
    return model


# ----------------------------
# YOLO classifier utils
# ----------------------------
# kaldırıldı


def clamp_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int, pad_ratio: float = 0.45, min_size: int = 64) -> Tuple[int, int, int, int]:
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio
    x1p = int(max(0, np.floor(x1 - pad_x)))
    y1p = int(max(0, np.floor(y1 - pad_y)))
    x2p = int(min(w - 1, np.ceil(x2 + pad_x)))
    y2p = int(min(h - 1, np.ceil(y2 + pad_y)))
    # enforce minimum crop size
    if (x2p - x1p) < min_size:
        delta = (min_size - (x2p - x1p)) // 2 + 1
        x1p = max(0, x1p - delta)
        x2p = min(w - 1, x2p + delta)
    if (y2p - y1p) < min_size:
        delta = (min_size - (y2p - y1p)) // 2 + 1
        y1p = max(0, y1p - delta)
        y2p = min(h - 1, y2p + delta)
    return x1p, y1p, x2p, y2p


def classify_boxes_with_yolo(yolo_model: YOLO, frame_rgb: np.ndarray, boxes: np.ndarray, use_cuda: bool) -> List[str]:
    return []


# ----------------------------
# Geometry utils
# ----------------------------
def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-8
    return float(inter / union)


def parse_label_text(label_txt: str) -> Tuple[str, Optional[float]]:
    """Parse label text like "class conf" where class may contain spaces.
    Returns (class_name, conf_float or None).
    """
    if not label_txt:
        return "", None
    parts = label_txt.strip().split()
    if not parts:
        return "", None
    # Try to interpret the last token as confidence
    conf_val: Optional[float] = None
    try:
        conf_val = float(parts[-1])
        cls_name = " ".join(parts[:-1]).strip()
        if cls_name == "":
            cls_name = parts[-1]
    except Exception:
        # No numeric confidence; treat entire string as class name
        cls_name = " ".join(parts)
    return cls_name, conf_val

def draw_fitted_label(crop_bgr: np.ndarray, text: str) -> np.ndarray:
    if not text:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    # Start from a large scale and decrease until it fits
    scale = max(0.5, min(2.0, h / 200.0))
    thickness = max(1, int(round(scale * 2)))
    for _ in range(10):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw <= w - 10:  # fits with margin
            break
        scale *= 0.9
        thickness = max(1, int(round(scale * 2)))
    tx, ty = 5, max(25, th + 8)
    cv2.rectangle(crop_bgr, (tx - 2, ty - th - 6), (tx + tw + 2, ty + 4), (0, 0, 0), -1)
    cv2.putText(crop_bgr, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (200, 200, 200), thickness, cv2.LINE_AA)
    return crop_bgr


# ----------------------------
# Search utils
# ----------------------------
def find_video_by_name(filename: str) -> Optional[str]:
    # If a full path is provided
    if os.path.isfile(filename):
        return filename

    project_root = os.path.dirname(os.path.abspath(__file__))
    search_dirs: List[str] = [
        os.path.join(project_root, "Videolar"),
        os.path.join(project_root, "Sonuç Videolar"),
        os.path.join(project_root, "MyData"),
        os.path.join(project_root, "data"),
        project_root,
    ]
    exts = [".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"]

    # If filename has no extension, try all
    candidates: List[str] = []
    has_ext = os.path.splitext(filename)[1] != ""
    if has_ext:
        candidates.append(filename)
    else:
        name_wo_ext = filename
        for e in exts:
            candidates.append(name_wo_ext + e)

    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            files_set = set(files)
            for cand in candidates:
                if cand in files_set:
                    return os.path.join(root, cand)
    return None

# ----------------------------
# Drawing utils
# ----------------------------
def draw_predictions(rgb: np.ndarray, boxes: np.ndarray, masks: np.ndarray, scores: np.ndarray, labels: Optional[List[str]] = None) -> np.ndarray:
    overlay = rgb.copy()
    # Dynamic font size based on frame height
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

    # Boxes
    for idx, ((x1, y1, x2, y2), sc) in enumerate(zip(boxes, scores)):
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        # Draw only YOLO label (class + conf). If empty, draw nothing.
        label = ""
        if labels is not None and idx < len(labels) and labels[idx] != "":
            label = labels[idx]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            tx, ty = int(x1), max(int(y1) - 8, th + 4)
            cv2.rectangle(overlay, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
            cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

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


# Small utility to stamp OK/NOK on analysis images
def draw_ok_nok_flag(img_bgr: np.ndarray, has_error: bool) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    text = "NOK" if has_error else "OK"
    color = (0, 0, 255) if has_error else (0, 200, 0)
    scale = max(0.9, min(2.0, h / 480.0))
    thickness = max(2, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = 12, 18 + th
    # black background rectangle
    cv2.rectangle(img_bgr, (x - 8, y - th - 10), (x + tw + 8, y + 6), (0, 0, 0), -1)
    cv2.putText(img_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return img_bgr


# ----------------------------
# Main video processing
# ----------------------------
@torch.no_grad()
def process_video(filename: str, score_thr: float = SCORE_THR, device_str: str = DEVICE) -> str:
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    model = load_model(device)
    project_root = os.path.dirname(os.path.abspath(__file__))
    # yolo_model kaldırıldı
    use_cuda = torch.cuda.is_available() and device_str == "cuda"

    video_path = find_video_by_name(filename)
    if video_path is None:
        raise FileNotFoundError(f"Video not found by filename: {filename}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(project_root, OUTPUT_DIR), exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(project_root, OUTPUT_DIR, f"{stem}_pred.mp4")
    crops_dir = os.path.join(project_root, "results", "MaskRCNN", "crops", stem)
    os.makedirs(crops_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    try:
        frame_idx = 0
        # Keep best crop per spatial cluster across the video
        clusters: List[dict] = []  # each: {"box": (x1,y1,x2,y2), "conf": float, "label": str, "crop": np.ndarray}
        IOU_MERGE_THR = 0.3
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_t = F.to_tensor(rgb).to(device)

            outputs = model([img_t])
            out = outputs[0]

            scores_all = out.get("scores", torch.empty(0)).detach().cpu().numpy()
            boxes_all = out.get("boxes", torch.empty(0, 4)).detach().cpu().numpy()
            masks_all = out.get("masks", torch.empty(0, 1, rgb.shape[0], rgb.shape[1])).detach().cpu().numpy()

            keep = scores_all >= score_thr if scores_all.size > 0 else np.array([], dtype=bool)
            boxes = boxes_all[keep] if boxes_all.size > 0 else boxes_all
            masks = masks_all[keep] if masks_all.size > 0 else masks_all
            scores = scores_all[keep] if scores_all.size > 0 else scores_all

            labels = []

            # Track best-confidence detection per unique location (cluster) across the video
            h, w = rgb.shape[:2]
            for det_idx, (x1, y1, x2, y2) in enumerate(boxes):
                label_txt = "" if det_idx >= len(labels) else labels[det_idx]
                if not label_txt:
                    continue
                parts = label_txt.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    cls_name = parts[0]
                    conf_val = float(parts[1])
                except Exception:
                    continue
                # Expand to crop and prepare annotation
                xi, yi, xj, yj = clamp_box(x1, y1, x2, y2, w, h)
                if xj <= xi or yj <= yi:
                    continue
                crop_bgr = cv2.cvtColor(rgb[yi:yj, xi:xj, :], cv2.COLOR_RGB2BGR)
                disp = f"{cls_name} {conf_val:.2f}"
                crop_bgr_annot = draw_fitted_label(crop_bgr.copy(), disp)

                # Cluster by IoU
                assigned = False
                cur_box = (float(xi), float(yi), float(xj), float(yj))
                for c in clusters:
                    iou = iou_xyxy(cur_box, c["box"]) 
                    if iou >= IOU_MERGE_THR:
                        # Update if this one is better
                        if conf_val > c["conf"]:
                            c["conf"] = conf_val
                            c["label"] = disp
                            c["box"] = cur_box
                            c["crop"] = crop_bgr_annot
                        assigned = True
                        break
                if not assigned:
                    clusters.append({"box": cur_box, "conf": conf_val, "label": disp, "crop": crop_bgr_annot})

            vis = draw_predictions(rgb, boxes, masks, scores, labels)
            writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        # Save one image per unique location with full label fitted
        for idx, c in enumerate(clusters):
            name = f"{stem}_best_{idx:02d}.png"
            cv2.imwrite(os.path.join(crops_dir, name), c["crop"])

    return out_path


# ----------------------------
# Multi-camera setup and analysis
# ----------------------------
def setup_cameras(camera_indices: List[int] = CAMERA_INDICES) -> dict[int, cv2.VideoCapture]:
    """Kameraları açar ve her kamera için özel ayarları uygular."""
    caps: dict[int, cv2.VideoCapture] = {}
    
    for idx in camera_indices:
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if USE_DSHOW else cv2.VideoCapture(idx)
        except Exception:
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Uyarı: Kamera {idx} açılamadı")
            continue
        # Düşük gecikme
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        # Kamera özel ayarlarını uygula
        apply_camera_settings(cap, get_settings_for(idx))
        # Isınma
        for _ in range(5):
            cap.read()
            time.sleep(0.005)
        caps[idx] = cap
        print(f"✓ Kamera {idx} açıldı ve ayarlar uygulandı")
    return caps


@torch.no_grad()
def analyze_frame(model, yolo_model, frame_bgr: np.ndarray, score_thr: float = SCORE_THR, use_cuda: bool = False) -> Tuple[np.ndarray, List[str], List[dict], bool]:
    """Tek bir frame üzerinde hata tespiti yap"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_t = F.to_tensor(rgb).to(next(model.parameters()).device)

    outputs = model([img_t])
    out = outputs[0]

    scores_all = out.get("scores", torch.empty(0)).detach().cpu().numpy()
    boxes_all = out.get("boxes", torch.empty(0, 4)).detach().cpu().numpy()
    masks_all = out.get("masks", torch.empty(0, 1, rgb.shape[0], rgb.shape[1])).detach().cpu().numpy()

    keep = scores_all >= score_thr if scores_all.size > 0 else np.array([], dtype=bool)
    boxes = boxes_all[keep] if boxes_all.size > 0 else boxes_all
    masks = masks_all[keep] if masks_all.size > 0 else masks_all
    scores = scores_all[keep] if scores_all.size > 0 else scores_all

    labels: List[str] = []
    
    # Crop bilgilerini hazırla
    crops_info = []
    h, w = rgb.shape[:2]
    for det_idx, (x1, y1, x2, y2) in enumerate(boxes):
        label_txt = "" if det_idx >= len(labels) else labels[det_idx]
        if not label_txt:
            # Default label from score only
            label_txt = f"{float(scores[det_idx]):.2f}" if det_idx < len(scores) else ""
        parts = label_txt.strip().split()
        if len(parts) < 1:
            continue
        try:
            conf_val = float(parts[-1]) if parts else float(scores[det_idx])
        except Exception:
            conf_val = float(scores[det_idx]) if det_idx < len(scores) else 0.0
        
        xi, yi, xj, yj = clamp_box(x1, y1, x2, y2, w, h)
        if xj <= xi or yj <= yi:
            continue
            
        crop_bgr = cv2.cvtColor(rgb[yi:yj, xi:xj, :], cv2.COLOR_RGB2BGR)
        disp = f"defect {conf_val:.2f}"
        crop_bgr_annot = draw_fitted_label(crop_bgr.copy(), disp)
        
        crops_info.append({
            "box": (float(xi), float(yi), float(xj), float(yj)),
            "conf": conf_val,
            "label": disp,
            "crop": crop_bgr_annot,
            "class": "defect"
        })

    vis = draw_predictions(rgb, boxes, masks, scores, labels)
    has_error = boxes.shape[0] > 0
    return vis, labels, crops_info, has_error


@torch.no_grad()
def process_cameras_on_demand(camera_indices: List[int] = CAMERA_INDICES, score_thr: float = SCORE_THR, device_str: str = DEVICE) -> None:
    """Kamerları hazırla ve 'a' tuşuna basınca analiz yap"""
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    model = load_model(device)
    project_root = os.path.dirname(os.path.abspath(__file__))
    # yolo_model kaldırıldı
    use_cuda = torch.cuda.is_available() and device_str == "cuda"

    os.makedirs(os.path.join(project_root, OUTPUT_DIR_LIVE), exist_ok=True)
    os.makedirs(os.path.join(project_root, LIVE_CROPS_DIR), exist_ok=True)

    # Kamerları aç
    caps = setup_cameras(camera_indices)
    if not caps:
        raise RuntimeError("Hiçbir kamera açılamadı")

    # Aktif kamera takibi (trackbar'lar bu kameraya uygulanır)
    active = {'idx': sorted(caps.keys())[0] if caps else None}

    def get_active_cap() -> Optional[cv2.VideoCapture]:
        if active['idx'] is None:
            return None
        return caps.get(active['idx'])

    # Trackbar callback'leri
    def tb_set_brightness(v):
        cap = get_active_cap();  cap and cap.set(cv2.CAP_PROP_BRIGHTNESS, v)
    def tb_set_contrast(v):
        cap = get_active_cap();  cap and cap.set(cv2.CAP_PROP_CONTRAST, v)
    def tb_set_saturation(v):
        cap = get_active_cap();  cap and cap.set(cv2.CAP_PROP_SATURATION, v)
    def tb_set_hue(v):
        cap = get_active_cap();  cap and cap.set(cv2.CAP_PROP_HUE, v)
    def tb_set_gain(v):
        cap = get_active_cap();  cap and cap.set(cv2.CAP_PROP_GAIN, v)
    def tb_set_exposure(v):
        cap = get_active_cap();  cap and cap.set(cv2.CAP_PROP_EXPOSURE, v)
    def tb_set_focus(v):
        cap = get_active_cap();  cap and cap.set(cv2.CAP_PROP_FOCUS, v)
    def tb_set_sharpness(v):
        cap = get_active_cap();  cap and cap.set(cv2.CAP_PROP_SHARPNESS, v)
    def tb_set_white_balance(v):
        cap = get_active_cap();  cap and cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, v)
    def tb_toggle_autofocus(v):
        cap = get_active_cap();  cap and cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if v else 0)
    def tb_toggle_auto_exposure(v):
        cap = get_active_cap();  
        if cap:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if v else 1)

    # Ayar penceresi ve trackbar'ları oluştur
    cv2.namedWindow('Ayarlar', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Ayarlar', 420, 420)
    s0 = get_settings_for(active['idx']) if active['idx'] is not None else CAMERA_SETTINGS
    cv2.createTrackbar('Parlak', 'Ayarlar', int(s0.get('brightness',128)), 255, tb_set_brightness)
    cv2.createTrackbar('Kontrast', 'Ayarlar', int(s0.get('contrast',128)), 255, tb_set_contrast)
    cv2.createTrackbar('Doygun', 'Ayarlar', int(s0.get('saturation',128)), 255, tb_set_saturation)
    cv2.createTrackbar('Renk', 'Ayarlar', int(s0.get('hue',0)), 180, tb_set_hue)
    cv2.createTrackbar('Kazanc', 'Ayarlar', int(s0.get('gain',50)), 255, tb_set_gain)
    cv2.createTrackbar('Pozlama', 'Ayarlar', int(s0.get('exposure',150)), 2000, tb_set_exposure)
    cv2.createTrackbar('Focus', 'Ayarlar', int(s0.get('focus',30)), 255, tb_set_focus)
    cv2.createTrackbar('Keskin', 'Ayarlar', int(s0.get('sharpness',128)), 255, tb_set_sharpness)
    cv2.createTrackbar('B.Denge', 'Ayarlar', int(s0.get('white_balance',4500)), 7000, tb_set_white_balance)
    cv2.createTrackbar('AutoF', 'Ayarlar', 1 if s0.get('autofocus',False) else 0, 1, tb_toggle_autofocus)
    cv2.createTrackbar('AutoE', 'Ayarlar', 1 if s0.get('auto_exposure',False) else 0, 1, tb_toggle_auto_exposure)

    # Ana pencereyi oluştur (üst: canlı, alt: analiz)
    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, MAIN_WIN_W, MAIN_WIN_H)
    
    # Kontrol penceresi
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 800, 600)
    
    print("\n" + "=" * 60)
    print("HATA TESPİT SİSTEMİ - KAMERA MODU")
    print("=" * 60)
    print(f"Açılan kameralar: {list(caps.keys())}")
    print(f"Çözünürlük: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"Model: Mask R-CNN")
    print("\nKontroller:")
    print("  'a' - Tüm kameralardan fotoğraf çek ve analiz et")
    print("  'q' - Çıkış")
    print("  '1-4' - Belirli kameradan tek fotoğraf çek")
    print("\nKlasör yapısı:")
    print("  results/MaskRCNN/live_preds/session_XXXXXX/")
    print("    ├── görüntü/ (orijinal görüntüler)")
    print("    └── sonuçlar/ (analiz sonuçları)")
    print("=" * 60 + "\n")

    frame_count = 0
    
    try:
        while True:
            # Her kameradan frame oku ve birleştir
            frames = {}
            for cam_idx, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    # Karo boyutuna ölçekle (canlı üst satır)
                    frame_resized = cv2.resize(frame, (TILE_W, TILE_H_TOP))
                    
                    # Kamera bilgisini frame üzerine ekle
                    cv2.putText(frame_resized, f"Kamera {cam_idx}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame_resized, f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    frames[cam_idx] = frame_resized
            
            # 4'lü grid oluştur (üst kısım - canlı): 4 karo yan yana
            if len(frames) >= 4:
                live_grid = np.hstack([
                    frames.get(0, np.zeros((TILE_H_TOP, TILE_W, 3), dtype=np.uint8)),
                    frames.get(1, np.zeros((TILE_H_TOP, TILE_W, 3), dtype=np.uint8)),
                    frames.get(2, np.zeros((TILE_H_TOP, TILE_W, 3), dtype=np.uint8)),
                    frames.get(3, np.zeros((TILE_H_TOP, TILE_W, 3), dtype=np.uint8)),
                ])
            elif len(frames) >= 2:
                # 2-3 kamera varsa yan yana diz
                tiles = [frames.get(i, np.zeros((TILE_H_TOP, TILE_W, 3), dtype=np.uint8)) for i in range(len(frames))]
                live_grid = np.hstack(tiles)
            elif len(frames) == 1:
                # Tek kamera varsa
                live_grid = frames[list(frames.keys())[0]]
            else:
                # Kamera yoksa siyah ekran
                live_grid = np.zeros((TILE_H_TOP, TILE_W * 4, 3), dtype=np.uint8)
                cv2.putText(live_grid, "Kamera Bulunamadi", (200, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Alt analiz grid'i başlangıçta siyah
            analysis_grid = np.zeros((TILE_H_BOT, TILE_W * 4, 3), dtype=np.uint8)
            # Üst + alt tek pencerede göster
            main_canvas = np.vstack([
                cv2.resize(live_grid, (MAIN_WIN_W, TOP_H)),
                cv2.resize(analysis_grid, (MAIN_WIN_W, BOT_H))
            ])
            cv2.imshow(WIN_MAIN, main_canvas)
            
            # Ana pencereye bilgi yaz
            info_img = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(info_img, "HATA TESPIT SISTEMI", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(info_img, f"Acilan kameralar: {list(caps.keys())}", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(info_img, f"Cozunurluk: {CAMERA_WIDTH}x{CAMERA_HEIGHT}", (50, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(info_img, f"Aktif Kamera: {active['idx']}", (50, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(info_img, "TAB - Aktif kamerayı değiştir", (50, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(info_img, "'a' - Tum kameralardan analiz", (50, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(info_img, "'1-4' - Tek kameradan analiz", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(info_img, "'q' - Cikis", (50, 230), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(info_img, f"Toplam analiz: {frame_count}", (50, 270), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow(WIN_CTRL, info_img)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('a'):
                # Tüm kameralardan analiz yap
                print(f"\n[{frame_count}] Tüm kameralardan analiz başlıyor...")
                
                # Her işlem için ayrı klasör oluştur
                timestamp = int(time.time() * 1000) % 100000
                session_dir = os.path.join(project_root, OUTPUT_DIR_LIVE, f"session_{timestamp}")
                goruntu_dir = os.path.join(session_dir, "görüntü")
                sonuclar_dir = os.path.join(session_dir, "sonuçlar")
                
                os.makedirs(goruntu_dir, exist_ok=True)
                os.makedirs(sonuclar_dir, exist_ok=True)
                
                # Analiz sonuçları için frame'leri sakla
                analysis_frames = {}
                
                for cam_idx, cap in caps.items():
                    ret, frame = cap.read()
                    if not ret:
                        print(f"  Kamera {cam_idx}: Frame okunamadı")
                        continue
                    
                    # Önce orijinal görüntüyü kaydet
                    cv2.imwrite(os.path.join(goruntu_dir, f"kamera_{cam_idx}_{timestamp}.jpg"), frame)
                    print(f"  Kamera {cam_idx}: Görüntü kaydedildi")
                    
                    # Sonra analiz yap
                    print(f"  Kamera {cam_idx}: Analiz ediliyor...")
                    vis, labels, crops_info, has_error = analyze_frame(model, None, frame, score_thr, use_cuda)
                    
                    # Analiz sonucunu kaydet (NOK/OK damgası ile)
                    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                    vis_bgr = draw_ok_nok_flag(vis_bgr, has_error)
                    cv2.imwrite(os.path.join(sonuclar_dir, f"kamera_{cam_idx}_analiz_{timestamp}.jpg"), vis_bgr)
                    
                    print(f"    {len(crops_info)} hata tespit edildi")
                    
                    # Analiz sonucunu karo boyutuna ölçekle (alt satır)
                    vis_resized = cv2.resize(vis_bgr, (TILE_W, TILE_H_BOT))
                    analysis_frames[cam_idx] = vis_resized
                
                frame_count += 1
                print(f"Analiz tamamlandı. Toplam frame: {frame_count}")
                print(f"Klasör: {session_dir}")
                
                # Analiz grid'ini alt yarıda göster
                if len(analysis_frames) >= 4:
                    analysis_grid = np.hstack([
                        analysis_frames.get(0, np.zeros((TILE_H_BOT, TILE_W, 3), dtype=np.uint8)),
                        analysis_frames.get(1, np.zeros((TILE_H_BOT, TILE_W, 3), dtype=np.uint8)),
                        analysis_frames.get(2, np.zeros((TILE_H_BOT, TILE_W, 3), dtype=np.uint8)),
                        analysis_frames.get(3, np.zeros((TILE_H_BOT, TILE_W, 3), dtype=np.uint8)),
                    ])
                elif len(analysis_frames) >= 2:
                    tiles = [analysis_frames.get(i, np.zeros((TILE_H_BOT, TILE_W, 3), dtype=np.uint8)) for i in range(len(analysis_frames))]
                    analysis_grid = np.hstack(tiles)
                elif len(analysis_frames) == 1:
                    analysis_grid = analysis_frames[list(analysis_frames.keys())[0]]
                else:
                    analysis_grid = np.zeros((TILE_H_BOT, TILE_W * 4, 3), dtype=np.uint8)

                main_canvas = np.vstack([
                    cv2.resize(live_grid, (MAIN_WIN_W, TOP_H)),
                    cv2.resize(analysis_grid, (MAIN_WIN_W, BOT_H))
                ])
                cv2.imshow(WIN_MAIN, main_canvas)
                
                # Analiz penceresini 5 saniye boyunca göster (GUI event'lerini işle)
                print("Analiz sonuçları 5 saniye gösterilecek...")
                t0 = time.time()
                while (time.time() - t0) < 10.0:
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                # Devam: ana pencere zaten açık kalır
                
            elif key == 9:  # TAB ile aktif kamerayı değiştir
                keys_sorted = sorted(caps.keys())
                if active['idx'] in keys_sorted:
                    i = keys_sorted.index(active['idx'])
                    active['idx'] = keys_sorted[(i + 1) % len(keys_sorted)]
                else:
                    active['idx'] = keys_sorted[0]
                # Trackbar'ları aktif kameranın mevcut değerlerine güncelle
                cap = get_active_cap()
                if cap is not None:
                    try:
                        cv2.setTrackbarPos('Parlak', 'Ayarlar', int(cap.get(cv2.CAP_PROP_BRIGHTNESS)))
                        cv2.setTrackbarPos('Kontrast', 'Ayarlar', int(cap.get(cv2.CAP_PROP_CONTRAST)))
                        cv2.setTrackbarPos('Doygun', 'Ayarlar', int(cap.get(cv2.CAP_PROP_SATURATION)))
                        cv2.setTrackbarPos('Renk', 'Ayarlar', int(cap.get(cv2.CAP_PROP_HUE)))
                        cv2.setTrackbarPos('Kazanc', 'Ayarlar', int(cap.get(cv2.CAP_PROP_GAIN)))
                        cv2.setTrackbarPos('Pozlama', 'Ayarlar', int(cap.get(cv2.CAP_PROP_EXPOSURE)))
                        cv2.setTrackbarPos('Focus', 'Ayarlar', int(cap.get(cv2.CAP_PROP_FOCUS)))
                        cv2.setTrackbarPos('Keskin', 'Ayarlar', int(cap.get(cv2.CAP_PROP_SHARPNESS)))
                        cv2.setTrackbarPos('B.Denge', 'Ayarlar', int(cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)))
                        cv2.setTrackbarPos('AutoF', 'Ayarlar', 1 if cap.get(cv2.CAP_PROP_AUTOFOCUS) else 0)
                        cv2.setTrackbarPos('AutoE', 'Ayarlar', 1 if cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 3 else 0)
                    except Exception:
                        pass
            elif key in [ord(str(d)) for d in range(10)]:
                target = int(chr(key))
                if target in caps:
                    active['idx'] = target
                    cap = get_active_cap()
                    if cap is not None:
                        try:
                            cv2.setTrackbarPos('Parlak', 'Ayarlar', int(cap.get(cv2.CAP_PROP_BRIGHTNESS)))
                            cv2.setTrackbarPos('Kontrast', 'Ayarlar', int(cap.get(cv2.CAP_PROP_CONTRAST)))
                            cv2.setTrackbarPos('Doygun', 'Ayarlar', int(cap.get(cv2.CAP_PROP_SATURATION)))
                            cv2.setTrackbarPos('Renk', 'Ayarlar', int(cap.get(cv2.CAP_PROP_HUE)))
                            cv2.setTrackbarPos('Kazanc', 'Ayarlar', int(cap.get(cv2.CAP_PROP_GAIN)))
                            cv2.setTrackbarPos('Pozlama', 'Ayarlar', int(cap.get(cv2.CAP_PROP_EXPOSURE)))
                            cv2.setTrackbarPos('Focus', 'Ayarlar', int(cap.get(cv2.CAP_PROP_FOCUS)))
                            cv2.setTrackbarPos('Keskin', 'Ayarlar', int(cap.get(cv2.CAP_PROP_SHARPNESS)))
                            cv2.setTrackbarPos('B.Denge', 'Ayarlar', int(cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)))
                            cv2.setTrackbarPos('AutoF', 'Ayarlar', 1 if cap.get(cv2.CAP_PROP_AUTOFOCUS) else 0)
                            cv2.setTrackbarPos('AutoE', 'Ayarlar', 1 if cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 3 else 0)
                        except Exception:
                            pass
    
    finally:
        for cap in caps.values():
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("\n✓ Kameralar kapatıldı.")

def main() -> None:
    # On-demand 4-camera processing
    try:
        process_cameras_on_demand(camera_indices=CAMERA_INDICES, score_thr=SCORE_THR, device_str=DEVICE)
    except Exception as e:
        print(f"Kamera isleminde hata: {e}. Video moduna geciliyor...")
        out_path = process_video(VIDEO_NAME, score_thr=SCORE_THR, device_str=DEVICE)
        print(f"Saved annotated video to: {out_path}")


if __name__ == "__main__":
    main()


