import argparse
import os
from pathlib import Path


# Klasör yolunu buraya girin
TARGET_DIR = Path(r"Videolar/Okvideo/Part1_frames")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}


def ensure_unique(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem = dst.stem
    ext = dst.suffix
    parent = dst.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Klasördeki tüm görsellerin adının başına ön ek ekler.")
    parser.add_argument("prefix", nargs="?", default="lens_", help="Eklenecek ön ek (varsayılan: lens_)")
    args = parser.parse_args()

    if not TARGET_DIR.exists() or not TARGET_DIR.is_dir():
        print(f"Hata: Klasör bulunamadı: {TARGET_DIR}")
        return 1

    images = [p for p in TARGET_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    if not images:
        print("Görsel bulunamadı.")
        return 0

    count = 0
    for src in images:
        if src.name.startswith(args.prefix):
            continue
        dst = src.with_name(args.prefix + src.name)
        dst = ensure_unique(dst)
        os.replace(src, dst)
        count += 1

    print(f"Tamamlandı: {count} dosya güncellendi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


