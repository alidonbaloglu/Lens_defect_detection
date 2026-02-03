"""
RT-DETR Inference Scripti
=========================
Eğitilmiş RT-DETR modeli ile görsel/video tespiti.

Kullanım:
    # Tek görsel
    python rtdetr_infer.py --source image.jpg
    
    # Klasör
    python rtdetr_infer.py --source images/
    
    # Video
    python rtdetr_infer.py --source video.mp4
    
    # Kamera
    python rtdetr_infer.py --source 0
"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import RTDETR
except ImportError:
    print("Ultralytics yüklü değil. Yüklemek için:")
    print("pip install ultralytics")
    exit(1)


# Görselleştirme ayarları
LINE_THICKNESS = 2  # İnce çizgi kalınlığı (piksel)
BOX_PADDING = 8    # Kutuyu genişletme miktarı (piksel)
FONT_SCALE = 1    # Yazı boyutu
FONT_THICKNESS = 2  # Yazı kalınlığı

# Sınıf renkleri (BGR formatında)
CLASS_COLORS = {
    "siyah_nokta": (255, 0, 0),   # Yeşil
    "cizik": (0, 0, 255),          # Kırmızı
    "default": (255, 165, 0),      # Turuncu
}


def draw_custom_boxes(image, boxes, names, padding=BOX_PADDING, thickness=LINE_THICKNESS):
    """
    Özel kutu çizimi: ince çizgiler ve genişletilmiş kutular.
    
    Args:
        image: Görsel (numpy array)
        boxes: Tespit edilen kutular
        names: Sınıf isimleri sözlüğü
        padding: Kutuyu genişletme miktarı
        thickness: Çizgi kalınlığı
    
    Returns:
        Üzerine çizilmiş görsel
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    for box in boxes:
        # Kutu koordinatları
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = names[cls_id]
        
        # Kutuyu genişlet (padding ekle)
        x1_padded = max(0, int(x1 - padding))
        y1_padded = max(0, int(y1 - padding))
        x2_padded = min(w, int(x2 + padding))
        y2_padded = min(h, int(y2 + padding))
        
        # Sınıf rengini al
        color = CLASS_COLORS.get(cls_name, CLASS_COLORS["default"])
        
        # İnce çizgiyle kutu çiz
        cv2.rectangle(img, (x1_padded, y1_padded), (x2_padded, y2_padded), color, thickness)
        
        # Etiket metni
        label = f"{cls_name}: {conf:.0%}"
        
        # Etiket arka planı için boyut hesapla
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
        
        # Etiket konumu (kutunun üstünde)
        label_y = y1_padded - 5 if y1_padded > text_h + 10 else y2_padded + text_h + 5
        
        # Etiket arka planı
        cv2.rectangle(img, 
                      (x1_padded, label_y - text_h - 2), 
                      (x1_padded + text_w + 4, label_y + 2), 
                      color, -1)
        
        # Etiket metni (beyaz)
        cv2.putText(img, label, (x1_padded + 2, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
    
    return img


# Varsayılan model yolu
DEFAULT_MODEL = r"C:/Users/ali.donbaloglu/Desktop/Lens/training_plots/Yeni_modeller/RT-DETR_X_V1/kabin_20251226_112111/weights/best.pt"

def get_output_dir_from_source(source: str) -> str:
    """
    Kaynak yolundan otomatik çıktı klasörü oluştur.
    Çıktı, kaynak klasörün yanında 'predictions' adıyla oluşturulur.
    """
    source_path = Path(source)
    
    # Kaynak bir dosya mı yoksa klasör mü?
    if source_path.is_file():
        # Dosya ise, dosyanın bulunduğu klasörde predictions oluştur
        parent_dir = source_path.parent
    elif source_path.is_dir():
        # Klasör ise, klasörün yanında predictions oluştur
        parent_dir = source_path.parent
    else:
        # Kamera veya geçersiz kaynak için varsayılan
        parent_dir = Path.cwd()
    
    output_dir = parent_dir / "predictions"
    return str(output_dir)


def predict(
    source: str,
    model_path: str = DEFAULT_MODEL,
    conf: float = 0.40,
    iou: float = 0.40,
    save: bool = True,
    show: bool = False,
    output_dir: str = None,
    padding: int = BOX_PADDING,
    thickness: int = LINE_THICKNESS,
):
    """
    RT-DETR ile object detection yap.
    
    Args:
        source: Görsel/video/klasör/kamera yolu
        model_path: Model dosyası yolu
        conf: Confidence threshold
        iou: IoU threshold
        save: Sonuçları kaydet
        show: Sonuçları göster
        output_dir: Çıktı klasörü (None ise otomatik oluşturulur)
        padding: Kutu genişletme miktarı (piksel)
        thickness: Çizgi kalınlığı (piksel)
    """
    # Çıktı klasörü belirtilmemişse otomatik oluştur
    if output_dir is None:
        output_dir = get_output_dir_from_source(source)
        print(f"Çıktı klasörü otomatik ayarlandı: {output_dir}")
    
    if not os.path.exists(model_path):
        print(f"Model bulunamadı: {model_path}")
        return None
    
    print(f"Model yükleniyor: {model_path}")
    model = RTDETR(model_path)
    
    print(f"Kaynak: {source}")
    print(f"Confidence: {conf}")
    print(f"IoU: {iou}")
    print(f"Kutu genişletme: {padding}px, Çizgi kalınlığı: {thickness}px")
    
    # Tahmin yap (save=False, kendi görselleştirmemizi yapacağız)
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        save=False,  # Kendi görselleştirmemizi yapacağız
        show=False,
        verbose=True,
    )
    
    # Çıktı klasörünü oluştur
    save_dir = Path(output_dir) / "predict"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Sonuçları yazdır ve özel görselleştirme ile kaydet
    for i, result in enumerate(results):
        boxes = result.boxes
        orig_img = result.orig_img  # Orijinal görsel (BGR)
        
        # Dosya adını al
        if hasattr(result, 'path') and result.path:
            img_name = Path(result.path).name
        else:
            img_name = f"image_{i+1}.jpg"
        
        if boxes is not None and len(boxes) > 0:
            print(f"\nGörsel {i+1}: {len(boxes)} tespit")
            for j, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                cls_name = result.names[cls_id]
                xyxy = box.xyxy[0].tolist()
                print(f"  [{j+1}] {cls_name}: {conf_val:.2%} - [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
            
            # Özel görselleştirme ile çiz
            annotated_img = draw_custom_boxes(orig_img, boxes, result.names, padding=padding, thickness=thickness)
        else:
            print(f"\nGörsel {i+1}: Tespit yok")
            annotated_img = orig_img.copy()
        
        # Görseli kaydet
        if save:
            save_path = save_dir / img_name
            cv2.imwrite(str(save_path), annotated_img)
        
        # Görseli göster
        if show:
            cv2.imshow(f"RT-DETR: {img_name}", annotated_img)
            cv2.waitKey(0)
    
    if show:
        cv2.destroyAllWindows()
    
    if save:
        print(f"\nSonuçlar kaydedildi: {save_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="RT-DETR Inference")
    parser.add_argument("--source", type=str, required=True, help="Görsel/video/klasör/kamera")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model yolu")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.40, help="IoU threshold")
    parser.add_argument("--save", action="store_true", default=True, help="Sonuçları kaydet")
    parser.add_argument("--show", action="store_true", help="Sonuçları göster")
    parser.add_argument("--output", type=str, default=None, help="Çıktı klasörü (varsayılan: kaynak ile aynı yerde 'predictions')")
    parser.add_argument("--padding", type=int, default=BOX_PADDING, help=f"Kutu genişletme miktarı (varsayılan: {BOX_PADDING}px)")
    parser.add_argument("--thickness", type=int, default=LINE_THICKNESS, help=f"Çizgi kalınlığı (varsayılan: {LINE_THICKNESS}px)")
    
    args = parser.parse_args()
    
    predict(
        source=args.source,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        show=args.show,
        output_dir=args.output,
        padding=args.padding,
        thickness=args.thickness,
    )


if __name__ == "__main__":
    main()
