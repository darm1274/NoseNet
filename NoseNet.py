import gc
import gzip
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm

# Limpieza de memoria
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
VIDEO_EXTS = (".mp4",)  # Puede agregar ".avi", ".mov", etc. si lo requiere

# Tamaño de batch para inferencia GPU. Ajustar según VRAM disponible:
#   4 GB VRAM  → batch_size = 8
#   8 GB VRAM  → batch_size = 16
#   12+ GB     → batch_size = 32
BATCH_SIZE = 64

# Preprocesamiento para imágenes térmicas con paleta de color (ironbow, rainbow, etc.).
# Convierte a escala de grises + CLAHE para acercar la distribución a la del entrenamiento.
# Se puede activar/desactivar desde el diálogo al inicio, o forzar aquí:
#   True  → siempre activo
#   False → siempre desactivo  ← modelos v2 entrenados en color, NO usar preprocesamiento
#   None  → pregunta al usuario cada vez
PREPROCESAR_ESCALA_GRISES: Optional[bool] = False

# -----------------------------------------------------------------------------
# Rutas de pesos (relativas al directorio del script)
# Descarga los pesos desde la sección Releases del repositorio y colócalos aquí:
#   weights/face_no_face/best.pt
#   weights/nose/best.pt
# -----------------------------------------------------------------------------
WEIGHTS_FACE = Path(__file__).parent / "weights" / "face_no_face" / "best.pt"
WEIGHTS_NOSE = Path(__file__).parent / "weights" / "nose" / "best.pt"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_files_by_ext(folder: Path, exts: Tuple[str, ...]) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def find_files_recursive(root_folder: Path, exts: Tuple[str, ...]) -> List[Path]:
    """Busca recursivamente archivos con las extensiones dadas en todas las subcarpetas."""
    found = []
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in sorted(filenames):
            if Path(fname).suffix.lower() in exts:
                found.append(Path(dirpath) / fname)
    return found


def build_key(root_folder: Path, file_path: Path) -> str:
    """Genera una clave única carpeta__archivo preservando la subcarpeta relativa."""
    try:
        rel = file_path.relative_to(root_folder)
        parts = list(rel.parts)
        # Si el archivo está directamente en la raíz, usar solo el stem
        if len(parts) == 1:
            return file_path.stem
        # Unir subcarpetas + stem con doble guión bajo
        return "__".join(parts[:-1] + [file_path.stem])
    except ValueError:
        return file_path.stem


def build_frames_output_dir(output_root_dir: Path, prefix: str, key: str) -> Path:
    out_dir = output_root_dir / f"{prefix}_{key}"
    ensure_dir(out_dir)
    return out_dir


def save_frame_png(output_dir: Path, key: str, frame_index: int, frame_bgr: np.ndarray) -> None:
    out_path = output_dir / f"{key}_frame{frame_index:04d}.png"
    ok = cv2.imwrite(str(out_path), frame_bgr)
    if not ok:
        raise IOError(f"No se pudo escribir el frame: {out_path}")


# -----------------------------------------------------------------------------
# Reporte
# -----------------------------------------------------------------------------
class Reporte:
    """Acumula eventos durante el procesamiento y genera un resumen al final."""

    def __init__(self) -> None:
        self.inicio: datetime = datetime.now()
        self.registros: List[Dict[str, Any]] = []

    def _agregar(self, nivel: str, archivo: str, mensaje: str, detalle: str = "") -> None:
        self.registros.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "nivel": nivel,
            "archivo": archivo,
            "mensaje": mensaje,
            "detalle": detalle,
        })

    def ok(self, archivo: str, mensaje: str, detalle: str = "") -> None:
        self._agregar("OK", archivo, mensaje, detalle)

    def advertencia(self, archivo: str, mensaje: str, detalle: str = "") -> None:
        self._agregar("ADVERTENCIA", archivo, mensaje, detalle)

    def error(self, archivo: str, mensaje: str, detalle: str = "") -> None:
        self._agregar("ERROR", archivo, mensaje, detalle)

    def guardar(self, output_dir: Path) -> Path:
        ensure_dir(output_dir)
        timestamp_str = self.inicio.strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"reporte_{timestamp_str}.txt"

        fin = datetime.now()
        duracion = fin - self.inicio

        totales = {"OK": 0, "ADVERTENCIA": 0, "ERROR": 0}
        for r in self.registros:
            totales[r["nivel"]] = totales.get(r["nivel"], 0) + 1

        lineas = [
            "=" * 80,
            "  REPORTE DE PROCESAMIENTO - NoseNet",
            "=" * 80,
            f"  Inicio    : {self.inicio.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Fin       : {fin.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Duración  : {str(duracion).split('.')[0]}",
            f"  OK        : {totales['OK']}",
            f"  Advertencias: {totales['ADVERTENCIA']}",
            f"  Errores   : {totales['ERROR']}",
            "=" * 80,
            "",
        ]

        for r in self.registros:
            linea = f"[{r['timestamp']}] [{r['nivel']:12s}] {r['archivo']}"
            lineas.append(linea)
            lineas.append(f"    {r['mensaje']}")
            if r["detalle"]:
                lineas.append(f"    Detalle: {r['detalle']}")
            lineas.append("")

        lineas += [
            "=" * 80,
            "  RESUMEN FINAL",
            "=" * 80,
        ]
        for r in self.registros:
            if r["nivel"] in ("ADVERTENCIA", "ERROR"):
                lineas.append(f"  [{r['nivel']:12s}] {r['archivo']} — {r['mensaje']}")

        report_path.write_text("\n".join(lineas), encoding="utf-8")
        logger.info("Reporte guardado en: %s", report_path)
        return report_path


# -----------------------------------------------------------------------------
# 1) GZIP -> Frames PNG
# -----------------------------------------------------------------------------
def procesar_gzip_y_guardar(
    filename: Path,
    output_root_dir: Path,
    key: str,
    max_workers: int = 4,
) -> Tuple[Optional[Path], int]:
    """
    Procesa un archivo .gzip con:
      - Header: 4 bytes (2 valores uint16: ancho y alto).
      - Luego frames consecutivos en 16 bits (2 bytes por píxel).

    Convierte a 8 bits y guarda PNG en:
      data/initial_data/gzip_converted_<key>

    Retorna:
      (output_dir, n_frames)
    """
    try:
        filename = filename.resolve()
        if not filename.exists():
            logger.error("Archivo no encontrado: %s", filename)
            return None, 0

        output_dir = build_frames_output_dir(output_root_dir, "gzip_converted", key)

        logger.info("Procesando gzip: %s", filename)

        with gzip.open(str(filename), "rb") as f:
            data = f.read()

        if len(data) < 4:
            logger.error("Archivo muy pequeño, header no encontrado: %s", filename)
            return output_dir, 0

        header = np.frombuffer(data[:4], dtype=np.uint16)
        if header.size < 2:
            logger.error("Header inválido: %s", filename)
            return output_dir, 0

        w, h = int(header[0]), int(header[1])
        logger.info("Dimensiones detectadas: %dx%d", w, h)

        frame_size = w * h * 2
        frame_data = data[4:]
        n_frames = len(frame_data) // frame_size
        if n_frames <= 0:
            logger.warning("No se encontraron frames en: %s", filename)
            return output_dir, 0

        logger.info("Frames detectados: %d", n_frames)

        frames_array = np.frombuffer(frame_data, dtype=np.uint16, count=n_frames * w * h).reshape(n_frames, h, w)

        scale = 255.0 / 25.0
        frames_8bit = np.clip(
            ((frames_array.astype(np.float32) / 100.0) - 293.15) * scale,
            0,
            255,
        ).astype(np.uint8)

        def guardar_frame(i: int) -> None:
            gray = frames_8bit[i]
            frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            save_frame_png(output_dir, key, i, frame_bgr)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(guardar_frame, range(n_frames)))

        logger.info("Frames guardados en: %s", output_dir)
        return output_dir, n_frames

    except Exception as e:
        logger.exception("Error procesando gzip %s: %s", filename, e)
        return None, 0


# -----------------------------------------------------------------------------
# 1b) MP4 -> Frames PNG
# -----------------------------------------------------------------------------
def procesar_mp4_y_guardar(
    filename: Path,
    output_root_dir: Path,
    key: str,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
) -> Tuple[Optional[Path], int]:
    """
    Extrae frames desde un .mp4 usando OpenCV y los guarda como PNG en:
      data/initial_data/mp4_converted_<key>

    Parámetros:
      frame_stride: guarda 1 de cada N frames (1 = guarda todos).
      max_frames: límite máximo de frames guardados (None = sin límite).

    Retorna:
      (output_dir, n_frames_guardados)
    """
    if frame_stride < 1:
        raise ValueError("frame_stride debe ser >= 1")

    try:
        filename = filename.resolve()
        if not filename.exists():
            logger.error("Video no encontrado: %s", filename)
            return None, 0

        output_dir = build_frames_output_dir(output_root_dir, "mp4_converted", key)

        logger.info("Procesando mp4: %s", filename)

        cap = cv2.VideoCapture(str(filename))
        if not cap.isOpened():
            logger.error("No se pudo abrir el video: %s", filename)
            return output_dir, 0

        saved = 0
        read_index = 0

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                if read_index % frame_stride == 0:
                    save_frame_png(output_dir, key, saved, frame_bgr)
                    saved += 1

                    if max_frames is not None and saved >= max_frames:
                        break

                read_index += 1
        finally:
            cap.release()

        if saved == 0:
            logger.warning("No se guardaron frames del video: %s", filename)

        logger.info("Frames guardados (%d) en: %s", saved, output_dir)
        return output_dir, saved

    except Exception as e:
        logger.exception("Error procesando mp4 %s: %s", filename, e)
        return None, 0


# -----------------------------------------------------------------------------
# Preprocesamiento de dominio
# -----------------------------------------------------------------------------
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocesar_imagen(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convierte una imagen BGR con paleta de color térmica a escala de grises
    y aplica CLAHE para normalizar el contraste.
    Devuelve BGR de 3 canales (requerido por YOLOv5).
    La imagen original en color NO se modifica — esta copia se usa solo para inferencia.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = _clahe.apply(gray)
    return cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)


# -----------------------------------------------------------------------------
# 2) Face / no_face
# -----------------------------------------------------------------------------
def _load_image(path: Path):
    """Carga una imagen con OpenCV; retorna (path, imagen_bgr) o (path, None)."""
    return path, cv2.imread(str(path))


def detectar_face_en_frames(
    input_frames_dir: Path,
    output_face_dir: Path,
    output_no_face_dir: Path,
    model_face,
    batch_size: int = BATCH_SIZE,
    preprocess: bool = False,
) -> Tuple[int, int]:
    """
    Para cada imagen en input_frames_dir (en batches):
      - Si detecta rostro (clase 0), crea máscara dejando solo la región del rostro.
      - Si no detecta, guarda la imagen original en output_no_face_dir.

    preprocess=True: convierte a escala de grises + CLAHE antes de inferir
                     (las imágenes guardadas conservan el color original).

    Retorna:
      (n_con_cara, n_sin_cara)
    """
    ensure_dir(output_face_dir)
    ensure_dir(output_no_face_dir)

    archivos = sorted([p for p in input_frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    n_face = 0
    n_no_face = 0

    with tqdm(total=len(archivos), desc="Detección face/no_face", unit="imagen") as pbar:
        for batch_start in range(0, len(archivos), batch_size):
            batch_paths = archivos[batch_start: batch_start + batch_size]

            with ThreadPoolExecutor(max_workers=min(batch_size, 8)) as pool:
                loaded = list(pool.map(lambda p: _load_image(p), batch_paths))

            valid = [(p, img) for p, img in loaded if img is not None]
            if not valid:
                pbar.update(len(batch_paths))
                continue

            valid_paths, valid_imgs = zip(*valid)

            # Inferencia en batch: con preprocesamiento pasa arrays grises; sin él, rutas
            if preprocess:
                batch_input = [preprocesar_imagen(img) for img in valid_imgs]
            else:
                batch_input = [str(p) for p in valid_paths]
            results = model_face(batch_input)

            for idx, (image_path, image) in enumerate(zip(valid_paths, valid_imgs)):
                original_height, original_width = image.shape[:2]
                detections = results.xyxy[idx].cpu().numpy()
                face_boxes = [d for d in detections if int(d[5]) == 0]

                if not face_boxes:
                    cv2.imwrite(str(output_no_face_dir / image_path.name), image)
                    n_no_face += 1
                    continue

                face_boxes.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
                xmin, ymin, xmax, ymax, conf, class_id = face_boxes[0]
                xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
                xmax, ymax = min(original_width, int(xmax)), min(original_height, int(ymax))

                mask = np.zeros_like(image)
                mask[ymin:ymax, xmin:xmax] = image[ymin:ymax, xmin:xmax]
                cv2.imwrite(str(output_face_dir / image_path.name), mask)
                n_face += 1

            pbar.update(len(batch_paths))

    return n_face, n_no_face


# -----------------------------------------------------------------------------
# 3) Nose
# -----------------------------------------------------------------------------
def detectar_nose_en_face_images(
    face_images_dir: Path,
    initial_frames_dir: Path,
    output_nose_dir: Path,
    results_frames_dir: Path,
    results_txt_path: Path,
    model_nose,
    batch_size: int = BATCH_SIZE,
    preprocess: bool = False,
) -> Tuple[int, int]:
    """
    Para cada imagen en face_images_dir (en batches):
      - Ejecuta detección de nariz (clase 0).
      - Dibuja caja en la imagen filtrada (face) y guarda en output_nose_dir.
      - Dibuja caja en la imagen original correspondiente y guarda en results_frames_dir.
      - Registra coordenadas en results_txt_path.

    preprocess=True: convierte a escala de grises + CLAHE antes de inferir
                     (las imágenes guardadas conservan el color original).

    Retorna:
      (n_con_nariz, n_sin_nariz)
    """
    ensure_dir(output_nose_dir)
    ensure_dir(results_frames_dir)
    ensure_dir(results_txt_path.parent)

    results_txt_path.write_text("Imagen, xmin, ymin, xmax, ymax, confianza\n", encoding="utf-8")

    archivos = sorted([p for p in face_images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    n_nose = 0
    n_no_nose = 0

    with tqdm(total=len(archivos), desc="Detección nose", unit="imagen") as pbar:
        for batch_start in range(0, len(archivos), batch_size):
            batch_paths = archivos[batch_start: batch_start + batch_size]

            # Carga paralela de face images e initial images simultáneamente
            with ThreadPoolExecutor(max_workers=min(batch_size * 2, 16)) as pool:
                loaded_face = list(pool.map(lambda p: _load_image(p), batch_paths))
                initial_paths = [initial_frames_dir / p.name for p in batch_paths]
                loaded_initial = list(pool.map(lambda p: _load_image(p), initial_paths))

            valid_indices = [i for i, (_, img) in enumerate(loaded_face) if img is not None]
            if not valid_indices:
                pbar.update(len(batch_paths))
                continue

            valid_face_paths = [batch_paths[i] for i in valid_indices]

            # Inferencia en batch: con preprocesamiento pasa arrays grises; sin él, rutas
            if preprocess:
                batch_input = [preprocesar_imagen(loaded_face[i][1]) for i in valid_indices]
            else:
                batch_input = [str(batch_paths[i]) for i in valid_indices]
            results = model_nose(batch_input)

            txt_lines = []
            for out_idx, orig_idx in enumerate(valid_indices):
                image_path, image = loaded_face[orig_idx]
                original_height, original_width = image.shape[:2]

                detecciones = results.xyxy[out_idx].cpu().numpy()
                nose_boxes = [d for d in detecciones if int(d[5]) == 0]

                if not nose_boxes:
                    n_no_nose += 1
                    continue

                nose_boxes.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
                xmin, ymin, xmax, ymax, conf, class_id = nose_boxes[0]
                xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
                xmax, ymax = min(original_width, int(xmax)), min(original_height, int(ymax))

                image_with_box = image.copy()
                cv2.rectangle(image_with_box, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.imwrite(str(output_nose_dir / image_path.name), image_with_box)

                _, initial_image = loaded_initial[orig_idx]
                if initial_image is not None:
                    initial_with_box = initial_image.copy()
                    cv2.rectangle(initial_with_box, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                    cv2.imwrite(str(results_frames_dir / image_path.name), initial_with_box)

                txt_lines.append(f"{image_path.name}, {xmin}, {ymin}, {xmax}, {ymax}, {conf:.4f}\n")
                n_nose += 1

            if txt_lines:
                with results_txt_path.open("a", encoding="utf-8") as f:
                    f.writelines(txt_lines)

            pbar.update(len(batch_paths))

    return n_nose, n_no_nose


# -----------------------------------------------------------------------------
# Procesamiento de un archivo individual
# -----------------------------------------------------------------------------
def procesar_archivo(
    file_path: Path,
    root_folder: Path,
    initial_data_base: Path,
    face_no_face_base: Path,
    nose_data_base: Path,
    results_base: Path,
    model_face,
    model_nose,
    reporte: Reporte,
    preprocess: bool = False,
) -> None:
    """Ejecuta el pipeline completo para un único archivo .gzip o .mp4."""
    key = build_key(root_folder, file_path)
    archivo_str = str(file_path.relative_to(root_folder))

    logger.info("--- Procesando: %s (key=%s)", archivo_str, key)

    # Paso 1: extraer frames
    try:
        if file_path.suffix.lower() == ".gzip":
            initial_frames_dir, n_frames = procesar_gzip_y_guardar(file_path, initial_data_base, key)
        else:
            initial_frames_dir, n_frames = procesar_mp4_y_guardar(
                file_path,
                initial_data_base,
                key,
                frame_stride=1,
                max_frames=None,
            )
    except Exception as exc:
        reporte.error(archivo_str, "Error extrayendo frames", str(exc))
        return

    if initial_frames_dir is None or n_frames == 0:
        reporte.advertencia(archivo_str, "No se extrajeron frames del archivo")
        return

    reporte.ok(archivo_str, f"Frames extraídos: {n_frames}", str(initial_frames_dir))

    # Paso 2: face/no_face
    output_face_dir = face_no_face_base / key / "face"
    output_no_face_dir = face_no_face_base / key / "no_face"
    ensure_dir(output_face_dir)
    ensure_dir(output_no_face_dir)

    try:
        n_face, n_no_face = detectar_face_en_frames(
            initial_frames_dir, output_face_dir, output_no_face_dir, model_face,
            preprocess=preprocess,
        )
        reporte.ok(
            archivo_str,
            f"Detección de rostros: {n_face} con cara, {n_no_face} sin cara",
        )
    except Exception as exc:
        reporte.error(archivo_str, "Error en detección de rostros", str(exc))
        return

    if n_face == 0:
        reporte.advertencia(archivo_str, "Ningún frame con rostro detectado — se omite detección de nariz")
        return

    # Paso 3: nose
    output_nose_dir = nose_data_base / key
    results_frames_dir = results_base / key / "frames"
    results_txt_path = results_base / key / "nose_detections.txt"
    ensure_dir(output_nose_dir)
    ensure_dir(results_frames_dir)

    try:
        n_nose, n_no_nose = detectar_nose_en_face_images(
            output_face_dir,
            initial_frames_dir,
            output_nose_dir,
            results_frames_dir,
            results_txt_path,
            model_nose,
            preprocess=preprocess,
        )
        reporte.ok(
            archivo_str,
            f"Detección de nariz: {n_nose} con nariz, {n_no_nose} sin nariz",
            str(results_txt_path),
        )
    except Exception as exc:
        reporte.error(archivo_str, "Error en detección de nariz", str(exc))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    yolo_device = "0" if use_cuda else "cpu"  # YOLOv5 requiere índice numérico, no "cuda"
    logger.info("Usando dispositivo: %s", device)

    # Directorios base de salida (relativos al directorio de NoseNet.py)
    script_dir = Path(__file__).parent
    initial_data_base = script_dir / "data" / "initial_data"
    face_no_face_base = script_dir / "data" / "face_no_face"
    nose_data_base = script_dir / "data" / "nose"
    results_base = script_dir / "results"
    reports_dir = script_dir / "reports"

    for d in [initial_data_base, face_no_face_base, nose_data_base, results_base, reports_dir]:
        ensure_dir(d)

    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU detectada: %s (%.1f GB VRAM)", gpu_name, vram_gb)
        logger.info("Batch size configurado: %d  — ajusta BATCH_SIZE en el código si hay OOM", BATCH_SIZE)

    # Verificar que los pesos existen antes de continuar
    for weights_path in (WEIGHTS_FACE, WEIGHTS_NOSE):
        if not weights_path.exists():
            logger.error(
                "Pesos no encontrados: %s\n"
                "Descarga los pesos desde la sección Releases del repositorio "
                "y colócalos en la carpeta weights/",
                weights_path,
            )
            return

    logger.info("Cargando modelo de face...")
    model_face = torch.hub.load(
        str(script_dir / "Models" / "yolov5"),
        "custom",
        path=str(WEIGHTS_FACE),
        source="local",
        device=yolo_device,
    )
    model_face.conf = 0.5
    if use_cuda:
        model_face.half()  # FP16: ~2x más rápido en GPU moderna

    logger.info("Cargando modelo de nose...")
    model_nose = torch.hub.load(
        str(script_dir / "Models" / "yolov5_nose"),
        "custom",
        path=str(WEIGHTS_NOSE),
        source="local",
        device=yolo_device,
    )
    model_nose.conf = 0.5
    if use_cuda:
        model_nose.half()  # FP16

    # Seleccionar carpeta madre (puede contener subcarpetas con mp4/gzip)
    root = tk.Tk()
    root.withdraw()
    carpeta = filedialog.askdirectory(
        title="Selecciona la carpeta madre con archivos .gzip y/o .mp4 (incluye subcarpetas)"
    )
    if not carpeta:
        logger.warning("No se seleccionó ninguna carpeta. Saliendo...")
        return

    carpeta_path = Path(carpeta)

    # Preguntar al usuario si activar preprocesamiento (a menos que esté forzado en la constante)
    if PREPROCESAR_ESCALA_GRISES is None:
        preprocess = messagebox.askyesno(
            title="Preprocesamiento de imagen",
            message=(
                "¿Activar conversión a escala de grises + CLAHE?\n\n"
                "Recomendado si las imágenes tienen paleta de color térmica\n"
                "(ironbow, rainbow, etc.) y el modelo se entrenó con imágenes en B/N.\n\n"
                "SÍ  → activa preprocesamiento\n"
                "NO → usa las imágenes tal como están"
            ),
        )
    else:
        preprocess = PREPROCESAR_ESCALA_GRISES

    logger.info("Preprocesamiento escala de grises: %s", "ACTIVO" if preprocess else "DESACTIVADO")

    # Búsqueda recursiva de archivos
    gzip_files = find_files_recursive(carpeta_path, (".gzip",))
    mp4_files = find_files_recursive(carpeta_path, VIDEO_EXTS)
    files = gzip_files + mp4_files

    if not files:
        logger.warning("No se encontraron archivos .gzip o .mp4 en: %s (ni en subcarpetas)", carpeta_path)
        return

    logger.info(
        "Archivos a procesar: %d (gzip=%d, mp4=%d) — búsqueda recursiva desde: %s",
        len(files), len(gzip_files), len(mp4_files), carpeta_path,
    )

    reporte = Reporte()
    reporte.ok(
        str(carpeta_path),
        f"Sesión iniciada — {len(files)} archivo(s) encontrado(s) — "
        f"preprocesamiento: {'ACTIVO' if preprocess else 'DESACTIVADO'}",
    )

    for file_path in tqdm(files, desc="Procesando archivos", unit="archivo"):
        procesar_archivo(
            file_path=file_path,
            root_folder=carpeta_path,
            initial_data_base=initial_data_base,
            face_no_face_base=face_no_face_base,
            nose_data_base=nose_data_base,
            results_base=results_base,
            model_face=model_face,
            model_nose=model_nose,
            reporte=reporte,
            preprocess=preprocess,
        )

    reporte.ok(str(carpeta_path), "Sesión finalizada")
    report_path = reporte.guardar(reports_dir)

    logger.info("Proceso completado. Reporte en: %s", report_path)


if __name__ == "__main__":
    main()
