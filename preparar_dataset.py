"""
preparar_dataset.py
-------------------
Procesa dataset_icesi y añade los datos al dataset_chile y dataset_nose existentes.

Clases en dataset_icesi:
  0 = face
  1 = nose
  2 = no_face

Salida dataset_chile (modelo face/no_face):
  0 = face
  1 = no_face
  (las etiquetas de nose se descartan)

Salida dataset_nose (modelo nose):
  0 = nose
  Imagen: máscara negra fuera del bbox de cara (misma resolución, coordenadas YOLO intactas)
  (solo imágenes que tienen AMBAS etiquetas: face + nose)

Split aplicado a los nuevos datos: 70% train / 20% val / 10% test
Los datos existentes en dataset_chile y dataset_nose NO se modifican.
"""

import logging
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────
SRC_DIR         = Path(r"D:\Usuarios\Labneurociencias\Documents\Chile\Chile\Data\dataset_icesi")
DST_CHILE       = Path(r"D:\Usuarios\Labneurociencias\Documents\Chile\Chile\Data\dataset_chile")
DST_NOSE        = Path(r"D:\Usuarios\Labneurociencias\Documents\Chile\Chile\Data\dataset_nose")

# ── Clases origen (dataset_icesi) ─────────────────────────────────────────────
CLS_FACE    = 0
CLS_NOSE    = 1
CLS_NO_FACE = 2

# ── Split ─────────────────────────────────────────────────────────────────────
SPLIT_TRAIN = 0.70
SPLIT_VAL   = 0.20
SPLIT_TEST  = 0.10
SEED        = 42
# ──────────────────────────────────────────────────────────────────────────────


# ── Utilidades ────────────────────────────────────────────────────────────────

def ensure_dataset_dirs(base: Path) -> None:
    for split in ("train", "val", "test"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)


def read_labels(txt_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Lee un archivo YOLO y devuelve lista de (class_id, cx, cy, w, h)."""
    labels = []
    for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            labels.append((int(parts[0]), float(parts[1]), float(parts[2]),
                           float(parts[3]), float(parts[4])))
    return labels


def write_labels(txt_path: Path, labels: List[Tuple]) -> None:
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in labels]
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def split_indices(n: int) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(n))
    random.shuffle(indices)
    n_train = round(n * SPLIT_TRAIN)
    n_val   = round(n * SPLIT_VAL)
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def apply_face_mask(img: np.ndarray, face_label: Tuple) -> np.ndarray:
    """Pone todo en negro excepto la región del bbox de cara."""
    _, cx, cy, w, h = face_label
    H, W = img.shape[:2]
    xmin = max(0, int((cx - w / 2) * W))
    ymin = max(0, int((cy - h / 2) * H))
    xmax = min(W, int((cx + w / 2) * W))
    ymax = min(H, int((cy + h / 2) * H))
    mask = np.zeros_like(img)
    mask[ymin:ymax, xmin:xmax] = img[ymin:ymax, xmin:xmax]
    return mask


# ── Recolección de muestras ───────────────────────────────────────────────────

def collect_samples(src: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Escanea src y separa las muestras válidas para cada dataset.
    Retorna (muestras_chile, muestras_nose).
    """
    samples_chile = []
    samples_nose  = []

    image_files = sorted(src.glob("*.png"))

    for img_path in image_files:
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue  # imagen sin etiquetar → ignorar

        labels = read_labels(txt_path)
        if not labels:
            continue

        classes_presentes = {l[0] for l in labels}

        # ── dataset_chile: necesita face o no_face ──────────────────────────
        face_labels    = [l for l in labels if l[0] == CLS_FACE]
        no_face_labels = [l for l in labels if l[0] == CLS_NO_FACE]

        if face_labels or no_face_labels:
            new_labels = []
            for l in face_labels:
                new_labels.append((0, l[1], l[2], l[3], l[4]))   # face → 0
            for l in no_face_labels:
                new_labels.append((1, l[1], l[2], l[3], l[4]))   # no_face → 1

            samples_chile.append({
                "img_path": img_path,
                "labels":   new_labels,
            })

        # ── dataset_nose: necesita face + nose ─────────────────────────────
        nose_labels = [l for l in labels if l[0] == CLS_NOSE]

        if face_labels and nose_labels:
            # Tomar la cara más grande como máscara
            biggest_face = max(face_labels, key=lambda l: l[3] * l[4])
            new_nose_labels = [(0, l[1], l[2], l[3], l[4]) for l in nose_labels]  # nose → 0

            samples_nose.append({
                "img_path":   img_path,
                "face_label": biggest_face,
                "labels":     new_nose_labels,
            })

    return samples_chile, samples_nose


# ── Escritura de splits ───────────────────────────────────────────────────────

def write_split(
    samples: List[Dict],
    dst: Path,
    apply_mask: bool = False,
) -> Dict[str, int]:
    """
    Divide las muestras en train/val/test y las escribe en dst.
    Si apply_mask=True, aplica la máscara de cara antes de guardar.
    Retorna conteo por split.
    """
    ensure_dataset_dirs(dst)
    idx_train, idx_val, idx_test = split_indices(len(samples))
    split_map = {
        "train": idx_train,
        "val":   idx_val,
        "test":  idx_test,
    }
    counts = {}

    for split_name, indices in split_map.items():
        img_out_dir = dst / "images" / split_name
        lbl_out_dir = dst / "labels" / split_name
        count = 0

        for i in indices:
            s = samples[i]
            img_path: Path = s["img_path"]
            stem = img_path.stem

            # Destino (evitar colisión con archivos existentes)
            out_img  = img_out_dir / img_path.name
            out_lbl  = lbl_out_dir / (stem + ".txt")

            # Si ya existe el nombre, agregar sufijo único
            if out_img.exists() or out_lbl.exists():
                suffix = 1
                while (img_out_dir / f"{stem}_{suffix}.png").exists():
                    suffix += 1
                out_img = img_out_dir / f"{stem}_{suffix}.png"
                out_lbl = lbl_out_dir / f"{stem}_{suffix}.txt"

            if apply_mask:
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning("No se pudo leer: %s", img_path)
                    continue
                masked = apply_face_mask(img, s["face_label"])
                cv2.imwrite(str(out_img), masked)
            else:
                shutil.copy2(img_path, out_img)

            write_labels(out_lbl, s["labels"])
            count += 1

        counts[split_name] = count
        logger.info("  %s/%s → %d muestras", dst.name, split_name, count)

    return counts


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(SEED)

    logger.info("Escaneando dataset_icesi: %s", SRC_DIR)
    samples_chile, samples_nose = collect_samples(SRC_DIR)

    logger.info("Muestras válidas para dataset_chile : %d", len(samples_chile))
    logger.info("Muestras válidas para dataset_nose  : %d", len(samples_nose))

    # ── dataset_chile ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("── Procesando dataset_chile (face / no_face) ──")
    counts_chile = write_split(samples_chile, DST_CHILE, apply_mask=False)
    total_chile = sum(counts_chile.values())
    logger.info("  Total añadido a dataset_chile: %d", total_chile)

    # ── dataset_nose ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("── Procesando dataset_nose (nose con máscara de cara) ──")
    counts_nose = write_split(samples_nose, DST_NOSE, apply_mask=True)
    total_nose = sum(counts_nose.values())
    logger.info("  Total añadido a dataset_nose: %d", total_nose)

    # ── Resumen ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESUMEN")
    logger.info("=" * 60)
    logger.info("dataset_chile  train/val/test: %d / %d / %d",
                counts_chile["train"], counts_chile["val"], counts_chile["test"])
    logger.info("dataset_nose   train/val/test: %d / %d / %d",
                counts_nose["train"], counts_nose["val"], counts_nose["test"])
    logger.info("")
    logger.info("Datos existentes NO modificados.")
    logger.info("Proceso completado.")


if __name__ == "__main__":
    main()
