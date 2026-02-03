import os
from typing import Optional, List

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN


# ----------------------------
# Config (edit here)
# ----------------------------
VIDEO_NAME = "Videolar/Kabin/cizik-2.mp4"  # only filename; will be searched recursively in project folders
SCORE_THR = 0.5
DEVICE = "cuda"  # or "cpu"
OUTPUT_DIR = os.path.join("results", "MaskRCNN", "video_predictions")
BACKBONE_NAME = "resnet101"  # choose between "resnet50" and "resnet101" to match your checkpoint

# ----------------------------
# Model utils
# ----------------------------
NUM_CLASSES = 8  # background + defect (Lens model)

# Sınıf isimleri (background dahil değil, sadece gerçek sınıflar)
CLASS_NAMES = [
    "background",  # 0 - bu gösterilmez
    "defect",
    "cizik",    # 1
    "enjeksiyon_noktasi",    # 2
    "kirik",    # 3
    "siyah_nokta",    # 4
    "siyahlk",    # 5
    "yabanci",    # 6

]


def get_weights_path() -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(project_root, "Modeller", "MaskRCNN_COCO_sinif_v1.pt")


def get_model_tag() -> str:
    weights_path = get_weights_path()
    base = os.path.basename(weights_path)
    tag, _ = os.path.splitext(base)
    return tag


def build_model(num_classes: int = NUM_CLASSES, backbone_name: str = BACKBONE_NAME) -> torchvision.models.detection.MaskRCNN:
    # Build FPN backbone without pretrained weights to align with custom checkpoints
    backbone = resnet_fpn_backbone(backbone_name, weights=None)
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model


def load_model(device: torch.device) -> torchvision.models.detection.MaskRCNN:
    weights_path = get_weights_path()
    model = build_model(NUM_CLASSES, BACKBONE_NAME)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)  # allow minor key diffs
    model.to(device).eval()
    return model


# ----------------------------
# Search utils
# ----------------------------
def find_video_by_name(filename: str) -> Optional[str]:
    # If an absolute or direct path is provided
    if os.path.isabs(filename) and os.path.isfile(filename):
        return filename

    project_root = os.path.dirname(os.path.abspath(__file__))
    direct_in_project = os.path.join(project_root, filename)
    if os.path.isfile(direct_in_project):
        return direct_in_project
    search_dirs: List[str] = [
        os.path.join(project_root, "Videolar"),
        os.path.join(project_root, "Sonuç Videolar"),
        os.path.join(project_root, "MyData"),
        os.path.join(project_root, "data"),
        project_root,
    ]
    exts = [".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"]

    # Build candidate basenames (support subpaths by matching just the basename)
    candidates: List[str] = []
    has_ext = os.path.splitext(filename)[1] != ""
    base_in = os.path.basename(filename)
    if has_ext:
        candidates.append(base_in)
    else:
        name_wo_ext = base_in
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
def draw_predictions(rgb: np.ndarray, boxes: np.ndarray, masks: np.ndarray, scores: np.ndarray, labels: np.ndarray, thr: float) -> np.ndarray:
    overlay = rgb.copy()
    keep = scores >= thr if scores.size > 0 else np.array([], dtype=bool)
    boxes = boxes[keep] if boxes.size > 0 else boxes
    masks = masks[keep] if masks.size > 0 else masks
    scores = scores[keep] if scores.size > 0 else scores
    labels = labels[keep] if labels.size > 0 else labels

    # Dynamic font size based on frame height
    h, w = rgb.shape[:2]
    font_scale = max(0.7, min(2.5, h / 720.0 * 1.2))
    thickness = max(2, int(round(font_scale * 2)))

    # Farklı sınıflar için farklı renkler
    colors = [
        (0, 0, 255),      # Kırmızı
        (0, 255, 0),      # Yeşil
        (255, 0, 0),      # Mavi
        (0, 255, 255),    # Sarı
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Cyan
        (128, 0, 128),    # Mor
        (255, 165, 0),     # Turuncu
    ]

    # Masks
    for i, m in enumerate(masks):
        m_bin = (m[0] >= 0.5).astype(np.uint8)
        if i < len(labels):
            color = colors[labels[i] % len(colors)]
        else:
            color = (0, 255, 0)  # Varsayılan yeşil
        contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=2)
        overlay[m_bin == 1] = (overlay[m_bin == 1] * 0.4 + np.array(color) * 0.6).astype(np.uint8)

    # Boxes
    for i, ((x1, y1, x2, y2), sc) in enumerate(zip(boxes, scores)):
        if i < len(labels):
            color = colors[labels[i] % len(colors)]
            class_name = CLASS_NAMES[labels[i]] if labels[i] < len(CLASS_NAMES) else f"class_{labels[i]}"
        else:
            color = (0, 0, 255)  # Varsayılan kırmızı
            class_name = "unknown"
            
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Sınıf ismi ve skor
        label = f"{class_name}: {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx, ty = int(x1), max(int(y1) - 8, th + 4)
        
        # Text background for readability
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
    model_tag = get_model_tag()

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
    out_path = os.path.join(project_root, OUTPUT_DIR, f"{stem}_{model_tag}_pred.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    try:
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"Total frames: {total_frames}")
        print(f"FPS: {fps}")
        print(f"Output will be saved to: {out_path}")
        
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
                
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processing frame {frame_count}/{total_frames}")
                
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_t = F.to_tensor(rgb).to(device)

            outputs = model([img_t])
            out = outputs[0]

            scores = out.get("scores", torch.empty(0)).detach().cpu().numpy()
            boxes = out.get("boxes", torch.empty(0, 4)).detach().cpu().numpy()
            masks = out.get("masks", torch.empty(0, 1, rgb.shape[0], rgb.shape[1])).detach().cpu().numpy()
            labels = out.get("labels", torch.empty(0)).detach().cpu().numpy()

            vis = draw_predictions(rgb, boxes, masks, scores, labels, score_thr)
            writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    finally:
        cap.release()
        writer.release()

    print(f"Video processing completed! Saved {frame_count} frames.")
    return out_path


def main() -> None:
    print("Starting video prediction...")
    print(f"Video: {VIDEO_NAME}")
    print(f"Score threshold: {SCORE_THR}")
    print(f"Device: {DEVICE}")
    print(f"Model: {get_model_tag()}")
    print("-" * 50)
    
    try:
        out_path = process_video(VIDEO_NAME, score_thr=SCORE_THR, device_str=DEVICE)
        print(f"Successfully saved annotated video to: {out_path}")
    except Exception as e:
        print(f"Error processing video: {e}")


if __name__ == "__main__":
    main()
