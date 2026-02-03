import os
import pandas as pd
import torch
import torchvision
import numpy as np
import cv2
from torchvision.transforms import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from PIL import Image
import torch.nn.functional as F_torch

# --- AYARLAR ---
MASKRCNN_MODEL_PATH = "C:/Users/ali.donbaloglu/Desktop/Lens/results/best_params_run/maskrcnn_resnet101_best_v1.pt"
MASKRCNN_BACKBONE = "resnet101"
DATASET_DIR = 'test_dataset/test-yeni_kabin'
NUM_CLASSES = 7  # Eğitim kodunda olduğu gibi 7 sınıf
CLASS_NAMES = [
    "background",
    "cizik",
    "enjeksiyon_noktasi",
    "kirik",
    "siyah_nokta",
    "siyahlk",
    "yabanci"
]
OUTPUT_EXCEL_PATH = 'maskrcnn_yenikabin_test_sonuclari.xlsx'

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4


# --- YARDIMCI SINIFLAR (Eğitim kodundan alındı) ---
class MultiModalPreprocessor:
    """Görüntüden RGB, Kenar (Edge) ve Gradyan kanallarını çıkarır."""
    def __call__(self, image_np: np.ndarray):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Kenar tespiti (Canny)
        edges = cv2.Canny(gray, 60, 150)
        
        # Gradyan tespiti (Sobel)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        # Normalizasyon
        if gradient.max() > 0:
            gradient = (gradient / gradient.max() * 255).astype(np.uint8)
        
        return {
            'image': image_np,
            'edge': np.expand_dims(edges, axis=-1),
            'gradient': np.expand_dims(gradient, axis=-1)
        }


class RefinedMaskRCNNPredictor(torch.nn.Module):
    """Kontur (sınır) bilgisiyle zenginleştirilmiş maske tahmincisi."""
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__()
        self.conv5_mask = torch.nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        
        # Kontur iyileştirme katmanları
        self.contour_conv1 = torch.nn.Conv2d(dim_reduced, 128, 3, padding=1)
        self.contour_conv2 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.contour_conv3 = torch.nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        x = self.conv5_mask(x)
        x = self.relu(x)
        
        mask_logits = self.mask_fcn_logits(x)
        
        # Kontur tahmini ve ana maskeye eklenmesi
        contour = self.relu(self.contour_conv1(x))
        contour = self.relu(self.contour_conv2(contour))
        contour_logits = self.contour_conv3(contour)
        
        # Ana maske ve kontur bilgisini birleştir
        return mask_logits + 0.3 * contour_logits


class BoundaryAwareRoIHeads(RoIHeads):
    """RoIHeads sınıfı - test için temel kullanım."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# --- YARDIMCI FONKSİYONLAR ---
def get_ground_truth(label_path):
    """Birden fazla ground truth etiketi döndürür: [(class_id, gt_box), ...]"""
    gt_list = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [float(p) for p in line.split()]
                class_id = int(parts[0])
                points = parts[1:]
                x_coords = points[0::2]
                y_coords = points[1::2]
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)
                gt_box = [(x_min + x_max) / 2, (y_min + y_max) / 2,
                          x_max - x_min, y_max - y_min]
                gt_list.append((class_id, gt_box))
        return gt_list
    except (FileNotFoundError, ValueError):
        return []


def calculate_iou(boxA, boxB):
    """İki kutu arasındaki IoU oranını hesaplar. [x_center, y_center, width, height] formatında olmalı."""
    def to_corners(box):
        x_center, y_center, w, h = box
        return [x_center - w / 2, y_center - h / 2,
                x_center + w / 2, y_center + h / 2]

    boxA_corners, boxB_corners = to_corners(boxA), to_corners(boxB)
    xA = max(boxA_corners[0], boxB_corners[0])
    yA = max(boxA_corners[1], boxB_corners[1])
    xB = min(boxA_corners[2], boxB_corners[2])
    yB = min(boxA_corners[3], boxB_corners[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    if boxAArea + boxBArea - interArea == 0:
        return 0
    return interArea / float(boxAArea + boxBArea - interArea)


def create_enhanced_maskrcnn(num_classes):
    """Eğitim kodundan aynı şekilde model oluşturur."""
    backbone = resnet_fpn_backbone(MASKRCNN_BACKBONE, weights=True)
    
    # İlk conv katmanını 5 kanallı girişe uyarla
    original_conv1 = backbone.body.conv1
    new_conv1 = torch.nn.Conv2d(
        5, original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=(original_conv1.bias is not None)
    )
    with torch.no_grad():
        # RGB ağırlıklarını kopyala
        new_conv1.weight[:, :3, :, :] = original_conv1.weight
        # Extra kanalları RGB ağırlıklarının ortalaması olarak başlat
        new_conv1.weight[:, 3:, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)

    backbone.body.conv1 = new_conv1

    # MaskRCNN modelini oluştur
    model = MaskRCNN(backbone, num_classes=num_classes)

    # Kutu tahmincisini ayarla
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Maske tahmincisini değiştir
    try:
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = RefinedMaskRCNNPredictor(in_features_mask, 256, num_classes)
    except Exception:
        pass

    # Modelin transform'ını 5 kanala uyarla
    try:
        from torchvision.models.detection.transform import GeneralizedRCNNTransform
        img_mean = [0.485, 0.456, 0.406, 0.5, 0.5]
        img_std = [0.229, 0.224, 0.225, 0.5, 0.5]
        model.transform = GeneralizedRCNNTransform(800, 1333, image_mean=img_mean, image_std=img_std)
    except Exception:
        pass

    return model


def load_maskrcnn_model(weights_path, num_classes, device):
    """Modeli ağırlıklarla birlikte yükler."""
    model = create_enhanced_maskrcnn(num_classes)
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        # Checkpoint'in state_dict'ini çıkar
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # strict=False kullanarak uyumsuz ağırlıkları yükle
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        print(f"✅ Model başarıyla yüklendi: {weights_path}")
        return model
    except Exception as e:
        print(f"HATA: Mask R-CNN modeli yüklenemedi. Hata: {e}")
        raise


def preprocess_image(image_path, device):
    """Görüntüyü 5 kanallı tensöre dönüştürür."""
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    
    # Multi-modal preprocessing
    preprocessor = MultiModalPreprocessor()
    modalities = preprocessor(img_np)
    
    # Kanalları birleştir: RGB (3) + Edge (1) + Gradient (1) = 5 kanal
    rgb_tensor = torch.from_numpy(modalities['image']).permute(2, 0, 1).float() / 255.0
    edge_tensor = torch.from_numpy(modalities['edge']).permute(2, 0, 1).float() / 255.0
    gradient_tensor = torch.from_numpy(modalities['gradient']).permute(2, 0, 1).float() / 255.0
    
    # 5 kanallı tensor oluştur
    multi_modal_tensor = torch.cat([rgb_tensor, edge_tensor, gradient_tensor], dim=0)
    
    return multi_modal_tensor.to(device), img


# --- ANA SCRIPT ---
@torch.no_grad()
def main_maskrcnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Kullanılan cihaz: {device}")
    
    # Modeli yükle
    model = load_maskrcnn_model(MASKRCNN_MODEL_PATH, NUM_CLASSES, device)

    images_dir = os.path.join(DATASET_DIR, 'images')
    labels_dir = os.path.join(DATASET_DIR, 'labels')
    
    if not os.path.exists(images_dir):
        print(f"❌ Hata: {images_dir} dizini bulunamadı!")
        return
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"📷 {len(image_files)} görüntü bulundu.")

    results_list = []

    for idx, image_name in enumerate(image_files, 1):
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')

        gt_items = get_ground_truth(label_path)

        if not gt_items:
            results_list.append({
                'Fotoğraf Adı': image_name,
                'Gerçek Etiket': 'Etiket Dosyası Yok/Bozuk',
                'Tahmin Edilen Etiket': 'N/A',
                'IoU': 0,
                'Sonuç': 'Belirsiz'
            })
            print(f"[{idx}/{len(image_files)}] {image_name} - Etiket yok")
            continue

        # Görüntüyü yükle ve işle
        img_tensor, original_img = preprocess_image(image_path, device)
        outputs = model([img_tensor])

        # Mask R-CNN tahminlerini filtrele
        scores = outputs[0]['scores']
        keep_indices = scores > CONFIDENCE_THRESHOLD

        if keep_indices.any():
            boxes = outputs[0]['boxes'][keep_indices]
            labels = outputs[0]['labels'][keep_indices]
        else:
            boxes, labels = [], []

        img_w, img_h = original_img.size

        # Her ground truth objesi için kontrol yap
        for gt_class_id_yolo, gt_box in gt_items:
            gt_class_id_mrcnn = gt_class_id_yolo + 1  # COCO formatında background sınıf vardır
            
            # Sınır kontrolü
            if gt_class_id_mrcnn >= len(CLASS_NAMES):
                gt_class_id_mrcnn = len(CLASS_NAMES) - 1
            
            gt_label_str = CLASS_NAMES[gt_class_id_mrcnn] if gt_class_id_mrcnn < len(CLASS_NAMES) else "Bilinmeyen"

            best_iou = 0
            best_match_label = "Tespit Edilemedi"
            found_correct = False

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box.tolist()
                pred_box = [((x1 + x2) / 2) / img_w, ((y1 + y2) / 2) / img_h,
                            (x2 - x1) / img_w, (y2 - y1) / img_h]
                pred_class_id = label.item()

                iou = calculate_iou(gt_box, pred_box)

                if pred_class_id == gt_class_id_mrcnn and iou > IOU_THRESHOLD:
                    found_correct = True
                    best_iou = iou
                    best_match_label = CLASS_NAMES[pred_class_id] if pred_class_id < len(CLASS_NAMES) else "Bilinmeyen"
                    break

                if iou > best_iou:
                    best_iou = iou
                    best_match_label = CLASS_NAMES[pred_class_id] if pred_class_id < len(CLASS_NAMES) else "Bilinmeyen"

            sonuc = "Doğru" if found_correct else ("Yanlış" if best_match_label != "Tespit Edilemedi" else "Tespit Edilemedi")

            results_list.append({
                'Fotoğraf Adı': image_name,
                'Gerçek Etiket': gt_label_str,
                'Tahmin Edilen Etiket': best_match_label,
                'IoU': f"{best_iou:.2f}",
                'Sonuç': sonuc
            })

            print(f"[{idx}/{len(image_files)}] {image_name} | Etiket='{gt_label_str}', Tahmin='{best_match_label}', IoU={best_iou:.2f} -> {sonuc}")

    # Sonuçları Excel'e kaydet
    df = pd.DataFrame(results_list)
    df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    
    # İstatistikleri yazdır
    print(f"\n{'='*60}")
    print(f"✅ İşlem tamamlandı! Sonuçlar '{OUTPUT_EXCEL_PATH}' dosyasına kaydedildi.")
    print(f"{'='*60}")
    
    if results_list:
        correct_count = sum(1 for r in results_list if r['Sonuç'] == 'Doğru')
        wrong_count = sum(1 for r in results_list if r['Sonuç'] == 'Yanlış')
        undetected_count = sum(1 for r in results_list if r['Sonuç'] == 'Tespit Edilemedi')
        
        print(f"📊 İstatistikler:")
        print(f"   ✓ Doğru Tahmin: {correct_count}")
        print(f"   ✗ Yanlış Tahmin: {wrong_count}")
        print(f"   ⚠ Tespit Edilemedi: {undetected_count}")
        print(f"   📈 Başarı Oranı: {100*correct_count/len(results_list):.2f}%")


if __name__ == '__main__':
    main_maskrcnn()
