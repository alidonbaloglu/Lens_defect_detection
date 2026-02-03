import os
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN


# ----------------------------
# Config (edit here)
# ----------------------------
VIDEO_NAME = "Videolar/Kabin/cizik-10.mp4"  # only filename; will be searched recursively in project folders
SCORE_THR = 0.5
DEVICE = "cuda"  # or "cpu"
OUTPUT_DIR = os.path.join("results", "MaskRCNN", "video_preds")
INPUT_VIDEOS_DIR = os.path.join("Videolar","İşaretli")
TARGET_FPS = 5.0  # sampling rate for batch processing (frames per second)
CROP_OUTPUT_DIR = os.path.join("results", "MaskRCNN", "crop")
BATCH_MODE = True  # set True to process all videos in INPUT_VIDEOS_DIR and save crops
BACKBONE_NAME = "resnet101"  # choose between "resnet50" and "resnet101" to match your checkpoint

# ----------------------------
# Model utils
# ----------------------------  İLK SORUN İÇİN  
NUM_CLASSES = 3  # background + defect (Lens model)


def get_weights_path() -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(project_root, "Modeller", "MaskRCNN_COCO_v4.pt")


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
def draw_predictions(rgb: np.ndarray, boxes: np.ndarray, masks: np.ndarray, scores: np.ndarray, thr: float) -> np.ndarray:
    overlay = rgb.copy()
    keep = scores >= thr if scores.size > 0 else np.array([], dtype=bool)
    boxes = boxes[keep] if boxes.size > 0 else boxes
    masks = masks[keep] if masks.size > 0 else masks
    scores = scores[keep] if scores.size > 0 else scores

    # Dynamic font size based on frame height
    h, w = rgb.shape[:2]
    font_scale = max(0.7, min(2.5, h / 720.0 * 1.2))
    thickness = max(2, int(round(font_scale * 2)))

    # Masks
    for m in masks:
        m_bin = (m[0] >= 0.5).astype(np.uint8)
        color = (0, 255, 0)
        contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=2)
        overlay[m_bin == 1] = (overlay[m_bin == 1] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

    # Boxes
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        # Text background for readability
        label = f"{sc:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx, ty = int(x1), max(int(y1) - 8, th + 4)
        cv2.rectangle(overlay, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    blended = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)
    return blended


# ----------------------------
# Enhancement utils
# ----------------------------
def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def homomorphic_illumination_correct(gray: np.ndarray, sigma: float = 30) -> np.ndarray:
    gray_log = np.log1p(gray.astype(np.float32))
    blur = cv2.GaussianBlur(gray_log, (0, 0), sigma)
    out = gray_log - blur
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return out


def enhance_for_defects(
    bgr: np.ndarray,
    min_focus_var: float = 120.0,
    denoise_strength: int = 3,
    clahe_clip: float = 1.5,
    clahe_tile: Tuple[int, int] = (8, 8),
    unsharp_sigma: float = 1.0,
    unsharp_amount: float = 0.3,
    bilateral_d: int = 5,
    bilateral_sigma_color: float = 50.0,
    bilateral_sigma_space: float = 50.0,
    scale_factor: float = 1.0,
    min_size: int = 100,
) -> Tuple[np.ndarray, dict]:
    # 1) Upscale image for better resolution
    h, w = bgr.shape[:2]
    if h < min_size or w < min_size:
        # Use INTER_CUBIC for better quality upscaling
        new_h = max(min_size, int(h * scale_factor))
        new_w = max(min_size, int(w * scale_factor))
        upscaled = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        upscaled = bgr.copy()
    
    # 2) Very light denoising to preserve natural look
    denoised = cv2.bilateralFilter(upscaled, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    
    # 3) Simple contrast enhancement without aggressive processing
    # Convert to LAB color space for better color preservation
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply gentle CLAHE only to L channel
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    l_enhanced = clahe.apply(l)
    
    # Merge back to LAB and convert to BGR
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 4) Very gentle sharpening to maintain natural look
    blur = cv2.GaussianBlur(enhanced_bgr, (0, 0), unsharp_sigma)
    final = cv2.addWeighted(enhanced_bgr, 1 + unsharp_amount, blur, -unsharp_amount, 0)

    # 5) Additional upscaling if still too small
    final_h, final_w = final.shape[:2]
    if final_h < min_size or final_w < min_size:
        final = cv2.resize(final, (max(min_size, final_w), max(min_size, final_h)), 
                          interpolation=cv2.INTER_CUBIC)

    # 6) Focus/quality check (convert to grayscale for focus calculation)
    gray_for_focus = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    focus = variance_of_laplacian(gray_for_focus)
    quality_ok = focus >= min_focus_var

    return final, {"focus_var": float(focus), "quality_ok": bool(quality_ok)}


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
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img_t = F.to_tensor(rgb).to(device)

            outputs = model([img_t])
            out = outputs[0]

            scores = out.get("scores", torch.empty(0)).detach().cpu().numpy()
            boxes = out.get("boxes", torch.empty(0, 4)).detach().cpu().numpy()
            masks = out.get("masks", torch.empty(0, 1, rgb.shape[0], rgb.shape[1])).detach().cpu().numpy()

            vis = draw_predictions(rgb, boxes, masks, scores, score_thr)
            writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    finally:
        cap.release()
        writer.release()

    return out_path


@torch.no_grad()
def _infer_on_frame(model: torchvision.models.detection.MaskRCNN, device: torch.device, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_t = F.to_tensor(rgb).to(device)
    outputs = model([img_t])
    out = outputs[0]
    scores = out.get("scores", torch.empty(0)).detach().cpu().numpy()
    boxes = out.get("boxes", torch.empty(0, 4)).detach().cpu().numpy()
    masks = out.get("masks", torch.empty(0, 1, rgb.shape[0], rgb.shape[1])).detach().cpu().numpy()
    return scores, boxes, masks


def _safe_crop(bgr: np.ndarray, box: np.ndarray, target_size: int = 100) -> Optional[np.ndarray]:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Crop the original bounding box
    crop = bgr[y1:y2, x1:x2].copy()
    
    # Calculate center and create 100x100 crop around the center
    crop_h, crop_w = crop.shape[:2]
    center_x = crop_w // 2
    center_y = crop_h // 2
    
    # Calculate the 100x100 region centered on the original crop center
    half_size = target_size // 2
    
    # Calculate the region in the original image coordinates
    orig_center_x = x1 + center_x
    orig_center_y = y1 + center_y
    
    # Calculate bounds for 100x100 region
    new_x1 = max(0, orig_center_x - half_size)
    new_y1 = max(0, orig_center_y - half_size)
    new_x2 = min(w, orig_center_x + half_size)
    new_y2 = min(h, orig_center_y + half_size)
    
    # If the region is smaller than 100x100, pad it
    if new_x2 - new_x1 < target_size or new_y2 - new_y1 < target_size:
        # Calculate padding needed
        pad_x = max(0, target_size - (new_x2 - new_x1))
        pad_y = max(0, target_size - (new_y2 - new_y1))
        
        # Adjust bounds to include padding
        new_x1 = max(0, new_x1 - pad_x // 2)
        new_y1 = max(0, new_y1 - pad_y // 2)
        new_x2 = min(w, new_x2 + pad_x - pad_x // 2)
        new_y2 = min(h, new_y2 + pad_y - pad_y // 2)
    
    # Extract the 100x100 region
    small_crop = bgr[new_y1:new_y2, new_x1:new_x2].copy()
    
    # Resize to exactly 100x100 if needed
    if small_crop.shape[0] != target_size or small_crop.shape[1] != target_size:
        small_crop = cv2.resize(small_crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    
    return small_crop


@torch.no_grad()
def process_video_to_crops(
    video_path: str,
    model: torchvision.models.detection.MaskRCNN,
    device: torch.device,
    score_thr: float,
    target_fps: float,
    crops_root: str,
) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _ = (width, height)  # kept for potential future use

    # Determine frame sampling interval to approximate target_fps
    if orig_fps <= 0:
        orig_fps = 25.0
    interval = max(1, int(round(orig_fps / max(1e-6, target_fps))))

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(crops_root, video_stem)
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    frame_idx = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if frame_idx % interval != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            scores, boxes, _masks = _infer_on_frame(model, device, rgb)

            if scores.size > 0:
                keep = scores >= score_thr
                boxes_keep = boxes[keep]
                scores_keep = scores[keep]
                for det_i, (box, sc) in enumerate(zip(boxes_keep, scores_keep)):
                    crop = _safe_crop(bgr, box)
                    if crop is None or crop.size == 0:
                        continue
                    enhanced_crop, _info = enhance_for_defects(crop)
                    filename = f"{video_stem}_f{frame_idx:06d}_d{det_i:02d}_s{sc:.2f}.jpg"
                    cv2.imwrite(os.path.join(out_dir, filename), enhanced_crop)
                    saved += 1

            frame_idx += 1
    finally:
        cap.release()

    return saved


@torch.no_grad()
def process_directory(
    input_dir: str = INPUT_VIDEOS_DIR,
    target_fps: float = TARGET_FPS,
    score_thr: float = SCORE_THR,
    device_str: str = DEVICE,
    crops_out_dir: str = CROP_OUTPUT_DIR,
) -> int:
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    model = load_model(device)

    project_root = os.path.dirname(os.path.abspath(__file__))
    search_root = os.path.join(project_root, input_dir) if not os.path.isabs(input_dir) else input_dir
    if not os.path.isdir(search_root):
        raise FileNotFoundError(f"Input directory not found: {search_root}")

    os.makedirs(os.path.join(project_root, crops_out_dir), exist_ok=True)
    crops_root = os.path.join(project_root, crops_out_dir)

    video_exts = {".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"}
    total_saved = 0
    for root, _dirs, files in os.walk(search_root):
        for fn in files:
            if os.path.splitext(fn)[1] in video_exts:
                vpath = os.path.join(root, fn)
                total_saved += process_video_to_crops(
                    vpath, model, device, score_thr, target_fps, crops_root
                )

    return total_saved


def main() -> None:
    if BATCH_MODE:
        total = process_directory(
            input_dir=INPUT_VIDEOS_DIR,
            target_fps=TARGET_FPS,
            score_thr=SCORE_THR,
            device_str=DEVICE,
            crops_out_dir=CROP_OUTPUT_DIR,
        )
        print(f"Saved {total} crops to directory: {os.path.join(os.path.dirname(os.path.abspath(__file__)), CROP_OUTPUT_DIR)}")
    else:
        out_path = process_video(VIDEO_NAME, score_thr=SCORE_THR, device_str=DEVICE)
        print(f"Saved annotated video to: {out_path}")


if __name__ == "__main__":
    main()


