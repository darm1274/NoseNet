import os
import gzip
import gc
import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


gc.collect()  # Libera memoria en CPU
torch.cuda.empty_cache()  # Libera memoria en GPU


IMG_EXTS = (".png", ".jpg", ".jpeg")
INPUT_EXTS = (".gzip", ".mp4")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_files_with_ext(folder: str, exts: tuple[str, ...]) -> list[str]:
    try:
        return sorted(
            [
                f
                for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(exts)
            ]
        )
    except FileNotFoundError:
        return []


###########################################
# Función para procesar archivo gzip y extraer frames PNG
###########################################
def procesar_gzip_y_guardar(filename: str, output_root_dir: str, max_workers: int = 4) -> tuple[str | None, int]:
    try:
        filename = os.path.abspath(filename)
        if not os.path.exists(filename):
            print(f"❌ Archivo no encontrado: {filename}")
            return None, 0

        print(f"📂 Procesando archivo gzip: {filename}")
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_dir = os.path.join(output_root_dir, f"gzip_converted_{base_name}")
        ensure_dir(output_dir)

        with gzip.open(filename, "rb") as f:
            try:
                _framerate = np.frombuffer(f.read(8), dtype=np.float64)[0]
                dims = np.frombuffer(f.read(4), dtype=np.uint16, count=2)
                if dims.size < 2:
                    raise ValueError("Error leyendo las dimensiones")
                w, h = int(dims[0]), int(dims[1])
            except Exception as meta_error:
                print(f"❌ Error leyendo metadatos en {filename}: {meta_error}")
                return None, 0

            framelen = w * h * 2  # Cada pixel ocupa 2 bytes
            record_size = 8 + framelen

            frames_list: list[np.ndarray] = []
            frame_index = 0

            while True:
                try:
                    record = f.read(record_size)
                except Exception as read_error:
                    print(f"❌ Error al leer frame {frame_index} en {filename}: {read_error}")
                    break

                if len(record) < record_size:
                    print(f"⚠️ Registro incompleto en frame {frame_index}. Finalizando extracción.")
                    break

                try:
                    frame_data = record[8:]  # omitir timestamp
                    data_array = np.frombuffer(frame_data, dtype=np.uint8)
                    frame_uint16 = data_array.view(np.uint16).reshape(h, w)

                    # Conversión a 8-bit (misma lógica original)
                    scale = 255.0 / 25.0
                    frame_8bit = np.clip(
                        ((frame_uint16.astype(np.float32) / 100.0) - 293.15) * scale,
                        0,
                        255,
                    ).astype(np.uint8)

                    frames_list.append(frame_8bit)
                except Exception as conversion_error:
                    print(f"❌ Error en conversión del frame {frame_index} en {filename}: {conversion_error}")

                frame_index += 1

        n_frames = len(frames_list)
        if n_frames == 0:
            print(f"⚠️ No se encontraron frames completos en {filename}.")
            return output_dir, 0

        def guardar_frame(i: int, frame_gray: np.ndarray) -> None:
            frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
            output_path = os.path.join(output_dir, f"{base_name}_frame{i:04d}.png")
            cv2.imwrite(output_path, frame_rgb)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(guardar_frame, i, frames_list[i]) for i in range(n_frames)]
            for future in futures:
                future.result()

        print(f"✅ {n_frames} frames guardados en: {output_dir}")
        return output_dir, n_frames

    except Exception as e:
        print(f"❌ Error general procesando gzip {filename}: {e}")
        return None, 0


###########################################
# Función para procesar archivo mp4 y extraer frames PNG
###########################################
def procesar_video_y_guardar(
    filename: str,
    output_root_dir: str,
    max_workers: int = 4,
    frame_step: int = 1,
) -> tuple[str | None, int]:
    """
    Extrae frames de un .mp4 usando OpenCV y los guarda como PNG en RGB.
    frame_step=1 guarda todos los frames.
    frame_step=2 guarda 1 de cada 2 frames, etc.
    """
    try:
        filename = os.path.abspath(filename)
        if not os.path.exists(filename):
            print(f"❌ Archivo no encontrado: {filename}")
            return None, 0

        if frame_step < 1:
            raise ValueError("frame_step debe ser >= 1")

        print(f"📂 Procesando video mp4: {filename}")
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_dir = os.path.join(output_root_dir, f"mp4_converted_{base_name}")
        ensure_dir(output_dir)

        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            print(f"❌ No se pudo abrir el video: {filename}")
            return None, 0

        frames_to_save: list[tuple[int, np.ndarray]] = []
        frame_idx = 0
        saved_idx = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % frame_step == 0:
                # Guardar en RGB como PNG (consistente con el flujo)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames_to_save.append((saved_idx, frame_rgb))
                saved_idx += 1

            frame_idx += 1

        cap.release()

        if saved_idx == 0:
            print(f"⚠️ No se extrajeron frames desde: {filename}")
            return output_dir, 0

        def guardar_frame(i: int, frame_rgb: np.ndarray) -> None:
            output_path = os.path.join(output_dir, f"{base_name}_frame{i:04d}.png")
            # OpenCV escribe en BGR, por eso se convierte de vuelta
            cv2.imwrite(output_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(guardar_frame, i, frame) for i, frame in frames_to_save]
            for future in futures:
                future.result()

        print(f"✅ {saved_idx} frames guardados en: {output_dir}")
        return output_dir, saved_idx

    except Exception as e:
        print(f"❌ Error general procesando mp4 {filename}: {e}")
        return None, 0


def extraer_frames_a_png(
    input_path: str,
    initial_data_base: str,
    max_workers: int = 4,
    frame_step_mp4: int = 1,
) -> tuple[str | None, int, str]:
    """
    Devuelve:
      - initial_frames_dir
      - n_frames
      - base_name (sin extensión)
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".gzip":
        initial_frames_dir, n_frames = procesar_gzip_y_guardar(
            input_path, initial_data_base, max_workers=max_workers
        )
        return initial_frames_dir, n_frames, base_name

    if ext == ".mp4":
        initial_frames_dir, n_frames = procesar_video_y_guardar(
            input_path,
            initial_data_base,
            max_workers=max_workers,
            frame_step=frame_step_mp4,
        )
        return initial_frames_dir, n_frames, base_name

    print(f"⚠️ Extensión no soportada: {input_path}")
    return None, 0, base_name


###########################################
# Función para detectar rostro en cada frame
###########################################
def detectar_face_en_frames(input_frames_dir: str, output_face_dir: str, output_no_face_dir: str, model_face) -> None:
    ensure_dir(output_face_dir)
    ensure_dir(output_no_face_dir)

    archivos = [f for f in os.listdir(input_frames_dir) if f.lower().endswith(IMG_EXTS)]
    for image_file in tqdm(archivos, desc="Detección face/no_face", unit="imagen"):
        image_path = os.path.join(input_frames_dir, image_file)

        image = cv2.imread(image_path)
        if image is None:
            continue
        original_height, original_width = image.shape[:2]

        results = model_face(image_path)
        detections = results.xyxy[0].cpu().numpy()

        face_boxes = [d for d in detections if int(d[5]) == 0]
        if not face_boxes:
            cv2.imwrite(os.path.join(output_no_face_dir, image_file), image)
            continue

        face_boxes.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
        xmin, ymin, xmax, ymax, conf, class_id = face_boxes[0]
        xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
        xmax, ymax = min(original_width, int(xmax)), min(original_height, int(ymax))

        mask = np.zeros_like(image)
        mask[ymin:ymax, xmin:xmax] = image[ymin:ymax, xmin:xmax]

        cv2.imwrite(os.path.join(output_face_dir, image_file), mask)


###########################################
# Función para detectar nariz en imágenes de rostro
###########################################
def detectar_nose_en_face_images(
    face_images_dir: str,
    initial_frames_dir: str,
    output_nose_dir: str,
    results_frames_dir: str,
    results_txt_path: str,
    model_nose,
) -> None:
    ensure_dir(output_nose_dir)
    ensure_dir(results_frames_dir)

    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("Imagen, xmin, ymin, xmax, ymax, confianza\n")

    archivos = [f for f in os.listdir(face_images_dir) if f.lower().endswith(IMG_EXTS)]
    for image_file in tqdm(archivos, desc="Detección nose", unit="imagen"):
        image_path = os.path.join(face_images_dir, image_file)

        image = cv2.imread(image_path)
        if image is None:
            continue
        original_height, original_width = image.shape[:2]

        results = model_nose(image_path)
        detecciones = results.xyxy[0].cpu().numpy()

        nose_boxes = [d for d in detecciones if int(d[5]) == 0]
        if not nose_boxes:
            continue

        nose_boxes.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
        xmin, ymin, xmax, ymax, conf, class_id = nose_boxes[0]
        xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
        xmax, ymax = min(original_width, int(xmax)), min(original_height, int(ymax))

        image_with_box = image.copy()
        cv2.rectangle(image_with_box, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        cv2.imwrite(os.path.join(output_nose_dir, image_file), image_with_box)

        initial_image_path = os.path.join(initial_frames_dir, image_file)
        if os.path.exists(initial_image_path):
            initial_image = cv2.imread(initial_image_path)
            if initial_image is not None:
                initial_with_box = initial_image.copy()
                cv2.rectangle(initial_with_box, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.imwrite(os.path.join(results_frames_dir, image_file), initial_with_box)

        with open(results_txt_path, "a", encoding="utf-8") as f:
            f.write(f"{image_file}, {xmin}, {ymin}, {xmax}, {ymax}, {conf:.4f}\n")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Directorios base (relativos)
    initial_data_base = os.path.join("data", "initial_data")
    face_no_face_base = os.path.join("data", "face_no_face")
    nose_data_base = os.path.join("data", "nose")
    results_base = "results"

    ensure_dir(initial_data_base)
    ensure_dir(face_no_face_base)
    ensure_dir(nose_data_base)
    ensure_dir(results_base)

    print("Cargando modelo de face...")
    model_face = torch.hub.load(
        "./Models/yolov5",
        "custom",
        path="./Models/yolov5/runs/train/face_no_face/weights/best.pt",
        source="local",
    )
    model_face.conf = 0.5

    print("Cargando modelo de nose...")
    model_nose = torch.hub.load(
        "./Models/yolov5_nose",
        "custom",
        path="./Models/yolov5_nose/runs/train/nose/weights/best.pt",
        source="local",
    )
    model_nose.conf = 0.5

    root = tk.Tk()
    root.withdraw()

    carpeta_input = filedialog.askdirectory(title="Selecciona la carpeta con archivos .gzip y/o .mp4")
    if not carpeta_input:
        print("No se seleccionó ninguna carpeta. Saliendo...")
        return

    input_files = list_files_with_ext(carpeta_input, INPUT_EXTS)
    if not input_files:
        print("⚠️ No se encontraron archivos .gzip o .mp4 en la carpeta seleccionada.")
        return

    # Opcional: para mp4 muy largos, subir frame_step_mp4 a 2, 3, 4...
    frame_step_mp4 = 1
    max_workers = 4

    for file in tqdm(input_files, desc="Procesando archivos", unit="archivo"):
        input_path = os.path.join(carpeta_input, file)

        initial_frames_dir, n_frames, base_name = extraer_frames_a_png(
            input_path=input_path,
            initial_data_base=initial_data_base,
            max_workers=max_workers,
            frame_step_mp4=frame_step_mp4,
        )
        if n_frames == 0 or initial_frames_dir is None:
            continue

        # Paso 2: face/no_face
        output_face_dir = os.path.join(face_no_face_base, base_name, "face")
        output_no_face_dir = os.path.join(face_no_face_base, base_name, "no_face")
        ensure_dir(output_face_dir)
        ensure_dir(output_no_face_dir)

        detectar_face_en_frames(initial_frames_dir, output_face_dir, output_no_face_dir, model_face)

        # Paso 3: nose
        output_nose_dir = os.path.join(nose_data_base, base_name)
        results_frames_dir = os.path.join(results_base, base_name, "frames")
        ensure_dir(output_nose_dir)
        ensure_dir(results_frames_dir)

        results_txt_path = os.path.join(results_base, base_name, "nose_detections.txt")

        detectar_nose_en_face_images(
            face_images_dir=output_face_dir,
            initial_frames_dir=initial_frames_dir,
            output_nose_dir=output_nose_dir,
            results_frames_dir=results_frames_dir,
            results_txt_path=results_txt_path,
            model_nose=model_nose,
        )

        # Limpieza ligera por archivo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n🚀 ¡Proceso completado en todos los archivos!")


if __name__ == "__main__":
    main()