from pathlib import Path
import cv2
import torch
from ultralytics import YOLO


# Sabit ayarlar: yalnızca dosyayı çalıştırmanız yeterli
INPUT_DIR = Path("C:/Users/ali.donbaloglu/Desktop/Lens/kamera/results/DataCollection/Part16/raw")
OUTPUT_DIR = Path("C:/Users/ali.donbaloglu/Desktop/Lens/kamera/results/DataCollection/Part16/output")
MODEL_PATH = Path("C:/Users/ali.donbaloglu/Desktop/Lens/training_plots/Kabin_merged_YOLO/weights/best.pt")
IMG_SIZE = 1408
CONF_THRESH = 0.3


def run_folder_inference(input_dir: Path, output_dir: Path, model_path: Path, imgsz: int = 1000, conf: float = 0.3) -> None:
    """Run YOLO on every image in the folder and save annotated outputs and box data."""
    model = YOLO(str(model_path))
    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skip unreadable file: {img_path.name}")
            continue

        results = model.predict(
            source=img,
            imgsz=imgsz,
            conf=conf,
            device=device,
            half=use_cuda,
            verbose=False,
        )

        annotated = results[0].plot()
        out_image_path = output_dir / f"{img_path.stem}_annotated.jpg"
        cv2.imwrite(str(out_image_path), annotated)

        boxes = results[0].boxes
        out_txt_path = output_dir / f"{img_path.stem}.txt"
        with out_txt_path.open("w") as f:
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    f.write(f"{cls_id} {conf_score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
            else:
                f.write("no detections\n")

        print(f"Saved {out_image_path.name} and box file")


if __name__ == "__main__":
    run_folder_inference(INPUT_DIR, OUTPUT_DIR, MODEL_PATH, IMG_SIZE, CONF_THRESH)
