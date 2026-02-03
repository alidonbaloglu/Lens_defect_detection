import os
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
YOLO_MODEL_PATH = os.path.join("Modeller", "Hibrit_yolo","weights","best.pt")
YOLO_CONF = 0.15  # lower for better recall
YOLO_IMGSZ = 1024  # larger for small details
YOLO_CLASS_NAMES = ["artik malzeme", "cizik","enjeksiyon noktasi", "siyah nokta","	yirtik"]  # fallback if model names missing
DEBUG_SAVE_UNKNOWN = True  # enable to save unknown crops for debugging

# ----------------------------
# Model utils
# ----------------------------
NUM_CLASSES = 3  # background + classes (will be overridden by checkpoint if available)


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
    # Use latest checkpoint trained with ResNet-101
    weights_path = os.path.join(project_root, "Modeller", "MaskRCNN_COCO_v4.pt")
    ckpt = torch.load(weights_path, map_location=device)

    # Determine number of classes from checkpoint if available
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
def load_yolo_classifier(project_root: str, device_str: str) -> YOLO:
    model_path = os.path.join(project_root, YOLO_MODEL_PATH)
    yolo_model = YOLO(model_path)
    # ultralytics uses 0/"cpu" for device; pass at predict time
    return yolo_model


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
    if boxes.size == 0:
        return []
    h, w = frame_rgb.shape[:2]
    labels: List[str] = []
    device_arg = 0 if use_cuda else "cpu"

    for (x1, y1, x2, y2) in boxes:
        xi, yi, xj, yj = clamp_box(x1, y1, x2, y2, w, h)
        if xj <= xi or yj <= yi:
            labels.append("Unknown")
            continue
        crop_rgb = frame_rgb[yi:yj, xi:xj, :]
        # Convert to BGR for Ultralytics which expects OpenCV images
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        # Run YOLO on the crop
        results = yolo_model.predict(
            source=crop_bgr,
            conf=YOLO_CONF,
            imgsz=YOLO_IMGSZ,
            device=device_arg,
            verbose=False,
            augment=True,
        )
        if len(results) == 0:
            labels.append("")
            continue
        r0 = results[0]
        # If detection/seg model
        if getattr(r0, "boxes", None) is not None and r0.boxes is not None and len(r0.boxes) > 0:
            boxes_res = r0.boxes
            confs = boxes_res.conf.cpu().numpy()
            clss = boxes_res.cls.cpu().numpy().astype(int)
            best_idx = int(np.argmax(confs))
            class_id = clss[best_idx]
            conf = float(confs[best_idx])
            # Prefer model-provided names; fallback to YOLO_CLASS_NAMES
            name_map = getattr(r0, "names", None)
            if name_map is None:
                name_map = getattr(getattr(yolo_model, "model", yolo_model), "names", getattr(yolo_model, "names", {}))
            if isinstance(name_map, dict) and len(name_map) > 0:
                class_name = name_map.get(class_id, str(class_id))
            elif 0 <= class_id < len(YOLO_CLASS_NAMES):
                class_name = YOLO_CLASS_NAMES[class_id]
            else:
                class_name = str(class_id)
            labels.append(f"{class_name} {conf:.2f}")
        # If classification model
        elif getattr(r0, "probs", None) is not None and r0.probs is not None and getattr(r0.probs, "top1", None) is not None:
            class_id = int(r0.probs.top1)
            conf = float(r0.probs.top1conf)
            name_map = getattr(r0, "names", None)
            if name_map is None:
                name_map = getattr(getattr(yolo_model, "model", yolo_model), "names", getattr(yolo_model, "names", {}))
            if isinstance(name_map, dict) and len(name_map) > 0:
                class_name = name_map.get(class_id, str(class_id))
            elif 0 <= class_id < len(YOLO_CLASS_NAMES):
                class_name = YOLO_CLASS_NAMES[class_id]
            else:
                class_name = str(class_id)
            labels.append(f"{class_name} {conf:.2f}")
        else:
            # Second-pass attempt with lower threshold and larger size
            results2 = yolo_model.predict(
                source=crop_bgr,
                conf=max(0.05, YOLO_CONF * 0.5),
                imgsz=max(1280, YOLO_IMGSZ),
                device=device_arg,
                verbose=False,
                augment=True,
            )
            if len(results2) > 0 and getattr(results2[0], "boxes", None) is not None and results2[0].boxes is not None and len(results2[0].boxes) > 0:
                boxes_res = results2[0].boxes
                confs = boxes_res.conf.cpu().numpy()
                clss = boxes_res.cls.cpu().numpy().astype(int)
                best_idx = int(np.argmax(confs))
                class_id = clss[best_idx]
                conf = float(confs[best_idx])
                name_map = getattr(results2[0], "names", None)
                if name_map is None:
                    name_map = getattr(getattr(yolo_model, "model", yolo_model), "names", getattr(yolo_model, "names", {}))
                if isinstance(name_map, dict) and len(name_map) > 0:
                    class_name = name_map.get(class_id, str(class_id))
                elif 0 <= class_id < len(YOLO_CLASS_NAMES):
                    class_name = YOLO_CLASS_NAMES[class_id]
                else:
                    class_name = str(class_id)
                labels.append(f"{class_name} {conf:.2f}")
            else:
                if DEBUG_SAVE_UNKNOWN:
                    dbg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "MaskRCNN", "unknown_crops")
                    os.makedirs(dbg_dir, exist_ok=True)
                    idx_name = f"crop_{yi}_{xi}_{yj}_{xj}.png"
                    cv2.imwrite(os.path.join(dbg_dir, idx_name), crop_bgr)
                labels.append("")
    return labels


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


# ----------------------------
# Main video processing
# ----------------------------
@torch.no_grad()
def process_video(filename: str, score_thr: float = SCORE_THR, device_str: str = DEVICE) -> str:
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    model = load_model(device)
    project_root = os.path.dirname(os.path.abspath(__file__))
    yolo_model = load_yolo_classifier(project_root, device_str)
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

            labels = classify_boxes_with_yolo(yolo_model, rgb, boxes, use_cuda)

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


def main() -> None:
    out_path = process_video(VIDEO_NAME, score_thr=SCORE_THR, device_str=DEVICE)
    print(f"Saved annotated video to: {out_path}")


if __name__ == "__main__":
    main()


