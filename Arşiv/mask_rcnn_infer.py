import os
from typing import Tuple, Optional, List

import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
import cv2


NUM_CLASSES = 2  # background + defect

# In-script configuration (edit here)
IMAGE_NAME = "resim.png"  # only filename; search is recursive in project folders
SCORE_THR = 0.5
DEVICE = "cuda"  # or "cpu"


def build_model(num_classes: int = NUM_CLASSES) -> torchvision.models.detection.MaskRCNN:
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layers = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layers, num_classes
    )
    return model


def load_image(image_path: str) -> Tuple[np.ndarray, torch.Tensor]:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(rgb)  # float32 [0,1]
    return rgb, tensor


@torch.no_grad()
def run_inference(
    image_path: str,
    weights_path: str,
    output_path: str,
    score_thr: float = 0.5,
    device_str: str = "cuda"
) -> str:
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    model = build_model(NUM_CLASSES)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])  # saved in training script
    model.eval().to(device)

    rgb, img_tensor = load_image(image_path)
    outputs = model([img_tensor.to(device)])
    out = outputs[0]

    scores = out.get("scores", torch.empty(0)).detach().cpu().numpy()
    boxes = out.get("boxes", torch.empty(0, 4)).detach().cpu().numpy()
    masks = out.get("masks", torch.empty(0, 1, rgb.shape[0], rgb.shape[1])).detach().cpu().numpy()

    keep = scores >= score_thr if scores.size > 0 else np.array([], dtype=bool)
    boxes = boxes[keep] if boxes.size > 0 else boxes
    masks = masks[keep] if masks.size > 0 else masks
    scores = scores[keep] if scores.size > 0 else scores

    overlay = rgb.copy()
    # Draw masks
    for m in masks:
        m_bin = (m[0] >= 0.5).astype(np.uint8)
        color = (0, 255, 0)
        contours, _ = cv2.findContours(m_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=2)
        overlay[m_bin == 1] = (overlay[m_bin == 1] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

    # Dynamic font size based on image height
    h, w = rgb.shape[:2]
    font_scale = max(0.7, min(2.5, h / 720.0 * 1.2))
    thickness = max(2, int(round(font_scale * 2)))

    # Draw boxes + readable labels
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        label = f"{sc:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx, ty = int(x1), max(int(y1) - 8, th + 4)
        cv2.rectangle(overlay, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    blended = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return output_path


def find_image_by_name(filename: str) -> Optional[str]:
    # If user accidentally passed a path, accept it directly
    if os.path.isfile(filename):
        return filename

    project_root = os.path.dirname(os.path.abspath(__file__))
    search_dirs: List[str] = [
        os.path.join(project_root, "MyData", "Lens_data_v2"),
        os.path.join(project_root, "MyData"),
        os.path.join(project_root, "Videolar"),
        os.path.join(project_root, "data"),
        project_root,
    ]
    exts = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

    # If filename has no extension, try all
    candidates: List[str] = []
    has_ext = os.path.splitext(filename)[1] != ""
    if has_ext:
        candidates.append(filename)
    else:
        name_wo_ext = filename
        for e in exts:
            candidates.append(name_wo_ext + e)

    # Search for any candidate within search_dirs (recursive)
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            files_set = set(files)
            for cand in candidates:
                if cand in files_set:
                    return os.path.join(root, cand)
    return None


def infer_by_name(filename: str, score_thr: float = SCORE_THR, device: str = DEVICE) -> str:
    project_root = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(project_root, "Modeller", "MaskRCNN_Lens.pt")

    image_path = find_image_by_name(filename)
    if image_path is None:
        raise FileNotFoundError(f"Image filename not found in project folders: {filename}")

    stem = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(project_root, "results", "MaskRCNN", "single_preds")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{stem}_pred.png")

    out_path = run_inference(
        image_path=image_path,
        weights_path=weights_path,
        output_path=output_path,
        score_thr=score_thr,
        device_str=device,
    )
    return out_path


def main() -> None:
    out_path = infer_by_name(IMAGE_NAME, score_thr=SCORE_THR, device=DEVICE)
    print(f"Saved prediction visualization to: {out_path}")


if __name__ == "__main__":
    main()




