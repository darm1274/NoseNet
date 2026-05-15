"""
extraer_frames_etiquetado.py
----------------------------
Extrae N frames aleatorios distribuidos proporcionalmente entre todos los MP4
encontrados de forma recursiva en una carpeta madre.

Salida:
  - frames_para_etiquetado/   → imágenes PNG listas para etiquetar
  - frames_para_etiquetado/manifest.csv  → trazabilidad (video origen, frame index)

Uso:
  python extraer_frames_etiquetado.py
  (abre un diálogo para seleccionar la carpeta madre)
"""

import csv
import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Configuración ─────────────────────────────────────────────────────────────
TOTAL_FRAMES   = 1000   # Total de frames a extraer
MIN_POR_VIDEO  = 5      # Mínimo garantizado por video (evita que videos cortos queden sin representación)
SEED           = 42     # Semilla para reproducibilidad (None = aleatorio puro)
OUTPUT_SUBDIR  = "frames_para_etiquetado"
# ──────────────────────────────────────────────────────────────────────────────


def find_mp4s(root: Path) -> List[Path]:
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if fname.lower().endswith(".mp4"):
                found.append(Path(dirpath) / fname)
    return found


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(count, 0)


def build_frame_key(root: Path, video_path: Path) -> str:
    """Nombre único para los frames: GX_PI__videoname"""
    try:
        rel = video_path.relative_to(root)
        parts = list(rel.parts)
        if len(parts) == 1:
            return video_path.stem
        return "__".join(parts[:-1] + [video_path.stem])
    except ValueError:
        return video_path.stem


def distribuir_proporcional(
    frame_counts: List[int],
    total: int,
    minimo: int,
) -> List[int]:
    """
    Distribuye `total` frames entre N videos proporcional a su longitud,
    garantizando al menos `minimo` frames por video.
    """
    n = len(frame_counts)
    # Primero asignar el mínimo a todos
    asignados = [min(minimo, fc) for fc in frame_counts]
    restante = total - sum(asignados)

    if restante <= 0:
        return asignados

    # Distribuir el resto proporcionalmente al excedente disponible
    excedente = [max(fc - minimo, 0) for fc in frame_counts]
    total_excedente = sum(excedente)

    if total_excedente == 0:
        return asignados

    for i in range(n):
        extra = round(restante * excedente[i] / total_excedente)
        asignados[i] += min(extra, excedente[i])

    # Ajuste fino para llegar exactamente a `total`
    diff = total - sum(asignados)
    indices = sorted(range(n), key=lambda i: excedente[i], reverse=True)
    for i in indices:
        if diff == 0:
            break
        puede = frame_counts[i] - asignados[i]
        ajuste = max(-asignados[i], min(diff, puede))
        asignados[i] += ajuste
        diff -= ajuste

    return asignados


def extraer_frames_de_video(
    video_path: Path,
    frame_indices: List[int],
    output_dir: Path,
    key: str,
) -> List[Tuple[str, str, int]]:
    """
    Extrae los frames indicados de un video y los guarda como PNG.
    Retorna lista de (nombre_archivo, ruta_video_relativa, frame_index).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("No se pudo abrir: %s", video_path)
        return []

    indices_set = set(frame_indices)
    guardados = []
    current = 0

    try:
        while cap.isOpened() and indices_set:
            ok, frame = cap.read()
            if not ok:
                break
            if current in indices_set:
                fname = f"{key}_f{current:05d}.png"
                out_path = output_dir / fname
                cv2.imwrite(str(out_path), frame)
                guardados.append((fname, str(video_path), current))
                indices_set.discard(current)
            current += 1
    finally:
        cap.release()

    return guardados


def main() -> None:
    if SEED is not None:
        random.seed(SEED)

    root_tk = tk.Tk()
    root_tk.withdraw()
    carpeta = filedialog.askdirectory(title="Selecciona la carpeta madre con los videos MP4")
    if not carpeta:
        logger.warning("No se seleccionó carpeta. Saliendo.")
        return

    root = Path(carpeta)
    output_dir = root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Buscando videos en: %s", root)
    videos = find_mp4s(root)
    if not videos:
        logger.error("No se encontraron archivos MP4.")
        return
    logger.info("Videos encontrados: %d", len(videos))

    # Contar frames por video
    logger.info("Contando frames por video...")
    frame_counts = []
    for v in tqdm(videos, desc="Contando frames", unit="video"):
        frame_counts.append(get_frame_count(v))

    total_disponible = sum(frame_counts)
    logger.info("Total de frames disponibles: %d", total_disponible)

    n_extraer = min(TOTAL_FRAMES, total_disponible)
    if n_extraer < TOTAL_FRAMES:
        logger.warning("Solo hay %d frames en total — se extraerán todos.", total_disponible)

    # Distribución proporcional
    asignaciones = distribuir_proporcional(frame_counts, n_extraer, MIN_POR_VIDEO)

    logger.info("Distribución calculada:")
    for v, fc, asig in zip(videos, frame_counts, asignaciones):
        logger.info("  %-60s  total=%5d  muestra=%3d", v.name, fc, asig)

    # Extracción
    manifest_rows = []
    total_guardados = 0

    for video_path, fc, n_muestra in tqdm(
        zip(videos, frame_counts, asignaciones),
        total=len(videos),
        desc="Extrayendo frames",
        unit="video",
    ):
        if n_muestra == 0 or fc == 0:
            continue

        indices = sorted(random.sample(range(fc), min(n_muestra, fc)))
        key = build_frame_key(root, video_path)
        guardados = extraer_frames_de_video(video_path, indices, output_dir, key)
        manifest_rows.extend(guardados)
        total_guardados += len(guardados)

    # Guardar manifest CSV
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["archivo_png", "video_origen", "frame_index"])
        writer.writerows(manifest_rows)

    logger.info("=" * 60)
    logger.info("Frames extraídos : %d", total_guardados)
    logger.info("Carpeta de salida: %s", output_dir)
    logger.info("Manifest         : %s", manifest_path)
    logger.info("=" * 60)
    logger.info("Siguiente paso: ejecutar abrir_labelimg.bat para etiquetar.")


if __name__ == "__main__":
    main()
