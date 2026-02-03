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
OUTPUT_DIR = os.path.join("results", "MaskRCNN", "sinifli_cnn_sonuc")
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
# Kamera ayarları (DataCollector'dan taşınan güncel ayarlar)
# ----------------------------
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

# Kamera bazlı farklı ayar vermek için (DataCollector'dan taşındı)
CAMERA_SETTINGS_PER_CAMERA: dict[int, dict] = {
    0:{
        'brightness':100,
        'contrast':126,
        'saturation':122,
        'hue':20,
        'focus': 23,
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
        'brightness':128,
        'contrast':128,
        'saturation':128,
        'focus': 30,
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
    white_balance = settings.get('white_balance', 4500)
    brightness = settings.get('brightness', 128)
    contrast = settings.get('contrast', 128)
    saturation = settings.get('saturation', 128)
    hue = settings.get('hue', 0)
    sharpness = settings.get('sharpness', 128)
    gain = settings.get('gain', 0)
    exposure = settings.get('exposure', 0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
    # auto_exposure için 3=auto, 1=manual
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if auto_exposure else 1)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1 if auto_white_balance else 0)
    if not autofocus:
        cap.set(cv2.CAP_PROP_FOCUS, focus)
    if not auto_white_balance:
        try:
            # Sürücülerin beklediği aralığa (2800-6500) kırp
            wb_val = int(max(2800, min(6500, int(white_balance))))
        except Exception:
            wb_val = 4500
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, wb_val)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    cap.set(cv2.CAP_PROP_SATURATION, saturation)
    cap.set(cv2.CAP_PROP_HUE, hue)
    cap.set(cv2.CAP_PROP_SHARPNESS, sharpness)
    cap.set(cv2.CAP_PROP_GAIN, gain)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)


# ----------------------------
# Part (PartX) klasör yönetimi
# ----------------------------
def list_existing_parts(base_dir: str) -> List[Tuple[int, str]]:
    parts: List[Tuple[int, str]] = []
    if not os.path.isdir(base_dir):
        return parts
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if not os.path.isdir(p):
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

def next_part_dir(base_dir: str) -> Tuple[int, str, str, str]:
    os.makedirs(base_dir, exist_ok=True)
    parts = list_existing_parts(base_dir)
    next_num = 1 if not parts else parts[-1][0] + 1
    part_name = f"Part{next_num}"
    part_dir = os.path.join(base_dir, part_name)
    raw_dir = os.path.join(part_dir, "raw")
    ana_dir = os.path.join(part_dir, "analysis")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ana_dir, exist_ok=True)
    return next_num, part_dir, raw_dir, ana_dir


# DataCollector'dan taşındı: Image index bulma
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
            # Örnek adlandırma formatına göre index'i bulmaya çalış
            # cam_X_focus_XX_NNNNN.png
            if len(parts) >= 4 and parts[-2].isdigit():
                candidate = int(parts[-2])
            # cam_X_focus_XX_NNNNN.jpg (MaskRCNN'den kalan)
            elif len(parts) >= 3 and parts[-1].isdigit():
                candidate = int(parts[-1])
            if candidate is not None and candidate > max_idx:
                max_idx = candidate
        except Exception:
            pass
    return max_idx + 1

# DataCollector'dan taşındı: Baseline focus bulma
def get_baseline_focus(cam_idx: int, caps: dict[int, cv2.VideoCapture]) -> int:
    """Her kamera için başlangıç (baseline) focus'u bulur."""
    settings_focus = CAMERA_SETTINGS_PER_CAMERA.get(cam_idx, {}).get('focus')
    if settings_focus is not None:
        base = int(settings_focus)
    else:
        base = 30
        try:
            cap = caps.get(cam_idx)
            if cap is not None:
                current = int(cap.get(cv2.CAP_PROP_FOCUS))
                if current > 0:
                    base = current
        except Exception:
            pass
    # Makul sınırlar içinde tut (UVC sürücüler için 0-255 aralığı yaygın)
    return max(0, min(255, int(base)))


# DataCollector'dan taşındı: Focus taraması ve kaydetme
def sweep_focus_and_capture(
    cam_idx: int,
    cap: cv2.VideoCapture,
    part_dir: str,
    next_image_idx: int,
    caps: dict[int, cv2.VideoCapture],
) -> int:
    """Belirtilen kamera için focus'u ±5 aralığında tarar ve her değerde fotoğraf çeker."""
    base = get_baseline_focus(cam_idx, caps)
    min_f = max(0, base - 5)
    max_f = min(255, base + 5)

    saved = 0
    # Önce -5'ten başlayarak base'e kadar, sonra base'den +5'e kadar
    focus_values = list(range(min_f, base)) + list(range(base, max_f + 1))
    
    # Geçici olarak autofocus kapatılır (apply_camera_settings zaten kapatıyor olmalı)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    for fval in focus_values:
        try:
            cap.set(cv2.CAP_PROP_FOCUS, fval)
            # odaklama motorunun oturması için kısa bekleme (DataCollector'dan taşındı)
            time.sleep(0.120)
        except Exception:
            pass

        # taze kare al
        frame = None
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                break
            time.sleep(0.010)
        if frame is None:
            continue

        # Dosya adı: cam_0_focus_23_00001.png
        fname = f"cam_{cam_idx}_focus_{fval:02d}_{next_image_idx:05d}.png"
        out_path = os.path.join(part_dir, fname)
        
        # PNG formatında, kayıpsız sıkıştırma ile kaydet (DataCollector'dan taşındı)
        if cv2.imwrite(out_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
            saved += 1
            print(f"[OK] Kamera {cam_idx}: Focus {fval} fotoğrafı kaydedildi -> {fname}")
        else:
            print(f"[ERROR] Kamera {cam_idx}: Focus {fval} fotoğrafı kaydedilemedi")

    # Tarama bittikten sonra kamera ayarlarını tekrar uygula
    try:
        settings = get_settings_for(cam_idx)
        apply_camera_settings(cap, settings)
    except Exception as e:
        print(f"[WARNING] Kamera {cam_idx}: Ayarlar tekrar uygulanamadı: {e}")
        
    return saved
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
            # DataCollector'dan taşındı: Kamera başlatma mantığı
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
                # DataCollector'dan taşındı: Buffer size ayarı
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        # Varsayılan sürücü ayarlarıyla birkaç kare okuyup buffer'ı tazele
        for _ in range(5):
            cap.read()
            time.sleep(0.005)
        # Kamera bazlı ayarları uygula (DataCollector ile aynı mantık)
        try:
            settings = get_settings_for(idx)
            # DataCollector'dan taşındı: Auto-exposure/Auto-white-balance'ı önce açıp sonra kapatma mantığı
            
            # 1. Adım: Auto-Exposure'ı aç (DataCollector'daki gibi 3 saniye bekleme yok, anlık ayar yapılıyor)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # 3 = auto mode
            cap.set(cv2.CAP_PROP_AUTO_WB, 1) # 1 = auto mode

            # 2. Adım: Manuel moda geçiş yap ve ayarları uygula
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 1 = manual mode
            time.sleep(0.200) # Bekleme (DataCollector'daki gibi)
            
            apply_camera_settings(cap, settings) # Ayarları uygula
            time.sleep(0.1)
            apply_camera_settings(cap, settings) # İkinci kez uygula (DataCollector'dan taşındı)
            
            print(f"✓ Kamera {idx} açıldı ve ayarlar uygulandı -> {settings}")
        except Exception as e:
            print(f"[WARNING] Kamera {idx}: Ayarlar uygulanamadı: {e}")
        caps[idx] = cap
    return caps


@torch.no_grad()
def process_cameras_cnn_only(camera_indices: List[int] = CAMERA_INDICES, score_thr: float = SCORE_THR, device_str: str = DEVICE) -> None:
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    model = load_model(device)

    caps = setup_cameras(camera_indices)
    if not caps:
        raise RuntimeError("Hiçbir kamera açılamadı")

    # PartX tabanlı klasörler (her analizde yeni Part oluşturulacak)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Başlangıç PartX ve Image Index'i belirle
    current_part_num, current_part_dir, _, _ = next_part_dir(OUTPUT_DIR)
    # MaskRCNN için raw/analysis klasörleri yerine ana Part klasörünü kullan
    next_image_idx = get_image_index_start(current_part_dir)
    print(f"Başlangıç Kayıt Klasörü: Part{current_part_num}, Başlangıç Index: {next_image_idx}")


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
            cv2.putText(info_img, "'f' - Fokus taramasi yap ve kaydet", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(info_img, "'n' - Sonraki Part klasorune gec", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(info_img, "'q' - Cikis", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow(WIN_CTRL, info_img)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'): # Sonraki Part klasörüne geç (DataCollector'dan taşındı)
                _, current_part_dir, _, _ = next_part_dir(OUTPUT_DIR)
                current_part_num = int(os.path.basename(current_part_dir)[4:])
                next_image_idx = get_image_index_start(current_part_dir)
                print(f"Yeni kayıt klasörü: Part{current_part_num}, Yeni Index: {next_image_idx}")
            elif key == ord('f'): # Fokus taraması yap ve kaydet (DataCollector'dan taşındı)
                print("Fokus taraması başlıyor...")
                total_saved = 0
                for cam_idx, cap in caps.items():
                    # Part klasörü güncel olmayabilir, tekrar al
                    _, current_part_dir, _, _ = next_part_dir(OUTPUT_DIR)
                    total_saved += sweep_focus_and_capture(cam_idx, cap, current_part_dir, next_image_idx, caps)

                if total_saved > 0:
                    next_image_idx += 1
                print(f"Fokus taraması bitti. Toplam kaydedilen: {total_saved} görüntü.")
            elif key == ord('a'):
                # Tüm kameralardan anlık analiz yap, alt satırı güncelle
                print("Analiz başlıyor...")
                i = 0
                # Her analiz başında yeni PartX oluştur (Bu mantık korunuyor)
                current_part_num, current_part_dir, out_raw_dir, out_ana_dir = next_part_dir(OUTPUT_DIR)
                print(f"Kayıt klasörü: Part{current_part_num}")
                ts = time.strftime("%Y%m%d_%H%M%S")
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
                    # Kayıt: ham görüntü + analiz çıktısı
                    try:
                        raw_name = f"cam{cam_idx}_{ts}_{frame_count:06d}_raw.jpg"
                        ana_name = f"cam{cam_idx}_{ts}_{frame_count:06d}_analysis.jpg"
                        cv2.imwrite(os.path.join(out_raw_dir, raw_name), frame)
                        cv2.imwrite(os.path.join(out_ana_dir, ana_name), vis_bgr)
                    except Exception as e:
                        print(f"[WARNING] Kayıt hatası (cam {cam_idx}): {e}")
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