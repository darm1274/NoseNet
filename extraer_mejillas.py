# -*- coding: utf-8 -*-
"""
CheekNet — Extraccion de temperatura de mejillas a partir de las coordenadas de nariz.

Para cada video termico (.gzip) y su archivo de coordenadas de nariz, ubica dos
ROIs (mejilla izquierda / derecha) ancladas geometricamente a la nariz y las
refina con informacion termica:

  1. Capa geometrica : ROI a +-k*ancho_nariz del centro de la nariz, a la altura
                       del centro de la nariz.
  2. Refinamiento    : se enmascaran los pixeles fuera del rango de piel y se
                       desliza la ventana (busqueda local) para maximizar la
                       fraccion de pixeles de piel -> corrige rotaciones leves
                       y evita el fondo frio / bordes.
  3. Control de calidad por frame:
        - ROI_FUERA   : muy pocos pixeles de piel dentro de la ROI.
        - ASIMETRIA   : |T_izq - T_der| por encima del umbral.

Salidas (en la misma carpeta results/<clave>/ que la nariz):
  - Cheek coordinates TXT/<archivo>.txt
        frame, Lxmin,Lymin,Lxmax,Lymax, Rxmin,Rymin,Rxmax,Rymax,
        L_mean, R_mean, L_weighted, R_weighted, n_skin_L, n_skin_R, flags
  - Cheek information TXT/<archivo>.txt
        image, L_roi_temp, L_weighted_temp, R_roi_temp, R_weighted_temp
        (mismo formato de matriz que "Nose information": Kelvin, ';' entre filas)

Uso:
    python extraer_mejillas.py
    (procesa results/C4F03 por defecto; editar BASE_RESULTS abajo)
"""
import gzip
import csv
import re
from pathlib import Path

import numpy as np
import cv2

# ----------------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------------
# BASE_RESULTS puede apuntar a:
#   - un caso con subcarpetas 'Thermal GZIP' + 'Nose coordinates TXT' (estilo C4F03), o
#   - una carpeta plana con pares <nombre>.gzip + <nombre>.txt (coordenadas de nariz), o
#   - la raiz 'results/' que contiene varios casos -> se procesan todos.
BASE_RESULTS = Path(__file__).parent / "results"

# Si OUTPUT_ROOT es None, las salidas se escriben dentro de la propia carpeta de
# datos. Si se define una ruta, las salidas van a OUTPUT_ROOT/<nombre_carpeta>/...
# (recomendado cuando los GZIP estan en una carpeta de datos crudos que no se
# quiere modificar).
OUTPUT_ROOT = None

# COORDS_DIR: carpeta EXTERNA con las coordenadas de caja de la nariz
# (formato 'Imagen, xmin, ymin, xmax, ymax, confianza'), emparejadas por nombre
# base del gzip: <stem_gzip>.txt. Util cuando el .txt junto al gzip NO son las
# cajas (p.ej. en ESTUDIO_1_COORDENADAS_NARIZ el .txt es informacion termica).
# Si es None, se busca el .txt de coordenadas junto al gzip o en 'Nose coordinates TXT'.
COORDS_DIR = None

# HOMOLOG_INPLACE: escribe el resultado de mejillas JUNTO a cada archivo de nariz
# (mismo directorio del gzip), como '<stem>_mejillas.txt' (+ '_mejillas_coords.txt'),
# en vez de en subcarpetas 'Cheek ... TXT'. Util cuando se quiere un homologo 1:1
# de la nariz en la misma carpeta. Ignora OUTPUT_ROOT.
HOMOLOG_INPLACE = False

# SKIP_EXISTING: omite grabaciones cuyo homologo ya existe -> permite reanudar
# un lote interrumpido sin reprocesar lo ya hecho.
SKIP_EXISTING = True

# --- Modo de exigencia de la deteccion -------------------------------------
# False -> PERMISIVO: casi toda nariz produce mejilla (umbrales laxos).
# True  -> ESTRICTO : exige ROI casi toda sobre piel, baja tolerancia a
#                     asimetria y marca cabezas giradas (perfil) como dudosas.
MODO_ESTRICTO = False

# --- Geometria de la ROI de mejilla, ANCLADA A LOS BORDES del nose box --------
# (en multiplos del ancho nw / alto nh de la nariz). Se permite un solape leve
# con la nariz (CHEEK_INNER) para mantenerlas centradas en la cara; se ubican a
# la altura del pomulo, justo por encima/debajo del centro nasal (sin los ojos).
CHEEK_INNER = 0.10      # el borde interno entra en la nariz 0.10*nw (solape ~12%, < 15%)
CHEEK_W = 0.80          # ancho de la ROI de mejilla
CHEEK_TOP = -0.05       # borde superior respecto al centro nasal (- = mas ARRIBA)
CHEEK_BOT = 0.65        # borde inferior, por debajo del centro de la nariz

# --- Refinamiento termico ---
SKIN_K_MIN = 303.15     # 30 C  -> limite inferior de piel
SKIN_K_MAX = 310.15     # 37 C  -> limite superior de piel
SEARCH_RADIUS = 3       # px de busqueda VERTICAL local (0 = solo geometrico)
# REFINAR_SIMETRICO: ajuste vertical simetrico (misma altura para ambas mejillas)
# para centrarlas sobre la piel del pomulo. El horizontal queda FIJO a los bordes
# de la nariz (no puede invadir la nariz ni subir hacia los ojos).
REFINAR_SIMETRICO = True

# --- Umbrales de calidad segun el modo -------------------------------------
# Se aplican via aplicar_modo() (llamada en main()), para que sobre-escribir
# MODO_ESTRICTO antes de main() tenga efecto sobre los umbrales.
MIN_SKIN_FRACTION = 0.45
ASYM_THRESHOLD_C = 2.0
PROFILE_RATIO = 0.0


def aplicar_modo():
    """Fija los umbrales de calidad segun MODO_ESTRICTO."""
    global MIN_SKIN_FRACTION, ASYM_THRESHOLD_C, PROFILE_RATIO
    if MODO_ESTRICTO:
        MIN_SKIN_FRACTION = 0.70   # ROI debe estar casi toda sobre piel
        ASYM_THRESHOLD_C = 1.0     # marca asimetrias L/R mas sutiles
        PROFILE_RATIO = 0.55       # min(fracL,fracR)/max < esto -> PERFIL (cabeza girada)
    else:
        MIN_SKIN_FRACTION = 0.45   # permisivo
        ASYM_THRESHOLD_C = 2.0
        PROFILE_RATIO = 0.0        # 0 = desactiva el chequeo de perfil

# --- Frames anotados (revision visual) -------------------------------------
# Si GUARDAR_FRAMES esta activo, por cada video se crea:
#   Cheek frames/<video>/con_mejilla/   -> frames con mejilla detectada (ROIs marcadas)
#   Cheek frames/<video>/sin_mejilla/   -> frames sin nariz, o con mejilla dudosa (flag)
# Para no generar millones de imagenes, solo se exportan frames de las primeras
# N_CASOS_FRAMES grabaciones de cada carpeta. El resto genera solo los TXT.
GUARDAR_FRAMES = True
N_CASOS_FRAMES = 3      # nro de grabaciones por carpeta a las que se exportan frames
FRAME_SCALE = 6         # factor de zoom de las imagenes guardadas
GUARDAR_CADA = 1        # 1 = todos los frames; N>1 = uno de cada N (para reducir volumen)


# ----------------------------------------------------------------------------
def leer_gzip(path):
    with gzip.open(str(path), "rb") as f:
        data = f.read()
    header = np.frombuffer(data[:4], dtype=np.uint16)
    w, h = int(header[0]), int(header[1])
    frame_size = w * h * 2
    frame_data = data[4:]
    n = len(frame_data) // frame_size
    arr = np.frombuffer(frame_data, dtype=np.uint16, count=n * w * h).reshape(n, h, w)
    return arr, w, h


def parse_nose_coords(path):
    """Devuelve dict {frame_idx: (xmin,ymin,xmax,ymax,conf)}."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)  # header
        for line in r:
            if len(line) < 6:
                continue
            name = line[0].strip()
            m = re.search(r"frame(\d+)", name)
            if not m:
                continue
            try:
                idx = int(m.group(1))
                xmin, ymin, xmax, ymax = (int(line[1]), int(line[2]), int(line[3]), int(line[4]))
                conf = float(line[5])
            except ValueError:
                continue
            out[idx] = (xmin, ymin, xmax, ymax, conf, name)
    return out


def weighted_temp(roi_k):
    """Temperatura ponderada (Kelvin), IDENTICA a 'Nose information'.

    Kernel gaussiano con sigma=1, centrado en (filas//2, cols//2), normalizado
    a suma 1 y aplicado sobre TODA la ROI (sin mascara). Replica exactamente
    extract_thermal_information.py para que mejilla y nariz sean comparables.
    """
    if roi_k.size == 0:
        return float("nan")
    sigma = 1.0
    rows = np.arange(roi_k.shape[0]) - roi_k.shape[0] // 2
    cols = np.arange(roi_k.shape[1]) - roi_k.shape[1] // 2
    kernel = np.exp(-((rows[:, None] ** 2 + cols[None, :] ** 2) / (2 * sigma ** 2)))
    kernel /= np.sum(kernel)
    return float(np.sum(roi_k * kernel))


def cheek_pair_base(xmin, ymin, xmax, ymax):
    """Cajas L/R base ancladas a los BORDES de la nariz (sin recortar, bordes float).

    Horizontal: borde interno pegado al lado de la nariz (mas un gap), hacia afuera.
    Vertical:   por debajo del centro nasal (no incluye ojos), misma altura ambas.
    """
    nw = max(1.0, xmax - xmin)
    nh = max(1.0, ymax - ymin)
    cy = (ymin + ymax) / 2.0
    y0 = cy + CHEEK_TOP * nh
    y1 = cy + CHEEK_BOT * nh
    cw = CHEEK_W * nw
    inn = CHEEK_INNER * nw
    Lx1 = xmin + inn      # borde interno (derecho) de la izq: solapa la nariz por dentro
    Lx0 = Lx1 - cw
    Rx0 = xmax - inn      # borde interno (izquierdo) de la der
    Rx1 = Rx0 + cw
    return (Lx0, y0, Lx1, y1), (Rx0, y0, Rx1, y1)


def clip_edges(box, w, h):
    x0, y0, x1, y1 = box
    return (max(0, int(round(x0))), max(0, int(round(y0))),
            min(w, int(round(x1))), min(h, int(round(y1))))


def skin_fraction(frame_k, box):
    x0, y0, x1, y1 = box
    patch = frame_k[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    mask = (patch >= SKIN_K_MIN) & (patch <= SKIN_K_MAX)
    return mask.mean()


def refine_pair(frame_k, xmin, ymin, xmax, ymax, w, h):
    """Ajuste VERTICAL simetrico del par de mejillas. El horizontal queda fijo a
    los bordes de la nariz (cajas ancladas por cheek_pair_base, no invaden la
    nariz ni suben hacia los ojos). Solo desliza ambas en y (mismo dy, sesgo
    hacia abajo) para centrarse sobre la piel del pomulo, maximizando piel."""
    Lb, Rb = cheek_pair_base(xmin, ymin, xmax, ymax)
    if SEARCH_RADIUS <= 0:
        return clip_edges(Lb, w, h), clip_edges(Rb, w, h)

    best = None
    best_score = -1.0
    for dy in range(-1, SEARCH_RADIUS + 1):   # como maximo 1 px hacia arriba
        Lc = clip_edges((Lb[0], Lb[1] + dy, Lb[2], Lb[3] + dy), w, h)
        Rc = clip_edges((Rb[0], Rb[1] + dy, Rb[2], Rb[3] + dy), w, h)
        if Lc[2] - Lc[0] < 2 or Rc[2] - Rc[0] < 2 or Lc[3] - Lc[1] < 2:
            continue
        frac = 0.5 * (skin_fraction(frame_k, Lc) + skin_fraction(frame_k, Rc))
        score = frac - 0.01 * abs(dy)
        if score > best_score:
            best_score = score
            best = (Lc, Rc)
    if best is None:
        best = (clip_edges(Lb, w, h), clip_edges(Rb, w, h))
    return best


def cheek_stats(frame_k, box):
    """Devuelve (roi_K, weighted_K, mean_C, n_skin).

    - weighted_K : temperatura ponderada gaussiana (formula de la nariz, Kelvin).
    - mean_C     : media de los pixeles de piel dentro de la ROI (C).
    - n_skin     : nro de pixeles dentro del rango de piel.
    """
    x0, y0, x1, y1 = box
    patch = frame_k[y0:y1, x0:x1]
    if patch.size == 0:
        return patch, float("nan"), float("nan"), 0
    wt_k = weighted_temp(patch)
    skin = (patch >= SKIN_K_MIN) & (patch <= SKIN_K_MAX)
    n_skin = int(skin.sum())
    mean_c = (patch[skin].mean() - 273.15) if n_skin else float("nan")
    return patch, wt_k, mean_c, n_skin


def matrix_to_str(patch_k):
    """Matriz Kelvin -> '[v v v;v v v]', formato identico a Nose information."""
    return "[" + ";".join(" ".join(map(str, row)) for row in patch_k) + "]"


def procesar_video(gzip_path, coords_path, info_out_file, coords_out_file, frames_root):
    arr, w, h = leer_gzip(gzip_path)
    coords = parse_nose_coords(coords_path)
    stem = gzip_path.stem  # nombre sin .gzip

    coords_lines = ["frame,Lxmin,Lymin,Lxmax,Lymax,Rxmin,Rymin,Rxmax,Rymax,"
                    "L_mean_C,R_mean_C,L_weighted_C,R_weighted_C,n_skin_L,n_skin_R,flags\n"]
    info_lines = ["image,L_roi_temp,L_weighted_temp,R_roi_temp,R_weighted_temp\n"]

    # carpetas de frames clasificados
    con_dir = sin_dir = None
    if GUARDAR_FRAMES and frames_root is not None:
        con_dir = frames_root / stem / "con_mejilla"
        sin_dir = frames_root / stem / "sin_mejilla"
        con_dir.mkdir(parents=True, exist_ok=True)
        sin_dir.mkdir(parents=True, exist_ok=True)

    n_con = 0   # frames con mejilla detectada (OK)
    n_sin_flag = 0   # frames con nariz pero mejilla dudosa
    n_sin_nariz = 0  # frames sin nariz

    for idx in range(arr.shape[0]):
        frame_k = arr[idx].astype(np.float64) / 100.0
        guardar = GUARDAR_FRAMES and frames_root is not None and (idx % GUARDAR_CADA == 0)

        if idx not in coords:
            # --- sin nariz -> sin mejilla posible ---
            n_sin_nariz += 1
            if guardar:
                img = dibujar_frame(arr[idx], w, h, nose=None, boxes=None,
                                    label=f"frame {idx}  SIN NARIZ", color_borde=(60, 60, 60))
                cv2.imwrite(str(sin_dir / f"frame{idx:04d}.png"), img)
            continue

        xmin, ymin, xmax, ymax, conf, name = coords[idx]

        boxes = {}
        stats = {}
        if REFINAR_SIMETRICO and SEARCH_RADIUS > 0:
            boxes["L"], boxes["R"] = refine_pair(frame_k, xmin, ymin, xmax, ymax, w, h)
        else:
            Lb, Rb = cheek_pair_base(xmin, ymin, xmax, ymax)
            boxes["L"] = clip_edges(Lb, w, h)
            boxes["R"] = clip_edges(Rb, w, h)
        stats["L"] = cheek_stats(frame_k, boxes["L"])
        stats["R"] = cheek_stats(frame_k, boxes["R"])

        (Lpatch, Lwt_k, Lmean, Lns) = stats["L"]
        (Rpatch, Rwt_k, Rmean, Rns) = stats["R"]
        Lbox, Rbox = boxes["L"], boxes["R"]

        # flags de calidad
        flags = []
        fracL = skin_fraction(frame_k, Lbox)
        fracR = skin_fraction(frame_k, Rbox)
        if fracL < MIN_SKIN_FRACTION:
            flags.append("ROI_FUERA_L")
        if fracR < MIN_SKIN_FRACTION:
            flags.append("ROI_FUERA_R")
        if not (np.isnan(Lmean) or np.isnan(Rmean)) and abs(Lmean - Rmean) > ASYM_THRESHOLD_C:
            flags.append("ASIMETRIA")
        # perfil: una mejilla mucho menos sobre piel que la otra -> cabeza girada
        if PROFILE_RATIO > 0 and max(fracL, fracR) > 0 and \
                min(fracL, fracR) / max(fracL, fracR) < PROFILE_RATIO:
            flags.append("PERFIL")
        ok = not flags
        flag_str = "OK" if ok else "|".join(flags)

        coords_lines.append(
            f"{idx},{Lbox[0]},{Lbox[1]},{Lbox[2]},{Lbox[3]},"
            f"{Rbox[0]},{Rbox[1]},{Rbox[2]},{Rbox[3]},"
            f"{Lmean:.3f},{Rmean:.3f},{Lwt_k - 273.15:.3f},{Rwt_k - 273.15:.3f},"
            f"{Lns},{Rns},{flag_str}\n"
        )
        info_lines.append(
            f"{name},{matrix_to_str(Lpatch)},{Lwt_k},"
            f"{matrix_to_str(Rpatch)},{Rwt_k}\n"
        )

        # clasificacion: OK -> con_mejilla ; flag -> sin_mejilla (dudosa)
        if ok:
            n_con += 1
        else:
            n_sin_flag += 1

        if guardar:
            label = f"frame {idx}  L:{Lmean:.1f}C R:{Rmean:.1f}C  [{flag_str}]"
            color_borde = (0, 200, 0) if ok else (0, 140, 255)  # verde / naranja
            img = dibujar_frame(arr[idx], w, h, nose=(xmin, ymin, xmax, ymax),
                                boxes=(Lbox, Rbox), label=label, color_borde=color_borde)
            destino = con_dir if ok else sin_dir
            cv2.imwrite(str(destino / f"frame{idx:04d}.png"), img)

    info_out_file.write_text("".join(info_lines), encoding="utf-8")
    coords_out_file.write_text("".join(coords_lines), encoding="utf-8")
    return n_con, n_sin_flag, n_sin_nariz


def dibujar_frame(frame_raw, w, h, nose, boxes, label, color_borde):
    """Frame termico anotado: nariz (rojo), ROIs de mejilla (cyan), borde de estado."""
    k = frame_raw.astype(np.float32) / 100.0
    img8 = np.clip((k - 293.15) * (255.0 / 25.0), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)
    s = FRAME_SCALE
    big = cv2.resize(color, (w * s, h * s), interpolation=cv2.INTER_NEAREST)

    if nose is not None:
        nx0, ny0, nx1, ny1 = nose
        cv2.rectangle(big, (nx0 * s, ny0 * s), (nx1 * s, ny1 * s), (0, 0, 255), 1)
    if boxes is not None:
        for box, tag in zip(boxes, ("L", "R")):
            x0, y0, x1, y1 = box
            cv2.rectangle(big, (x0 * s, y0 * s), (x1 * s, y1 * s), (255, 255, 0), 1)
            cv2.putText(big, tag, (x0 * s + 2, y0 * s + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # borde de color segun estado (verde=ok, naranja=dudosa, gris=sin nariz)
    cv2.rectangle(big, (0, 0), (w * s - 1, h * s - 1), color_borde, 3)
    cv2.putText(big, label, (4, h * s - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return big


def emparejar_archivos(base):
    """Empareja cada .gzip con su .txt de coordenadas de nariz por nombre base.

    Soporta dos disposiciones:
      A) subcarpetas 'Thermal GZIP' + 'Nose coordinates TXT'  (estilo C4F03)
      B) carpeta plana: <nombre>.gzip junto a <nombre>.txt
    """
    if (base / "Thermal GZIP").is_dir():
        gzip_dir = base / "Thermal GZIP"
        coords_dir = base / "Nose coordinates TXT"
    else:
        gzip_dir = coords_dir = base  # carpeta plana
    # una carpeta externa de coordenadas tiene prioridad
    if COORDS_DIR is not None:
        coords_dir = COORDS_DIR

    pares = []
    for gz in sorted(gzip_dir.glob("*.gzip")):
        base_name = gz.stem  # p.ej. 'C1F01_2023-10-19 11.43.17  thermal.80x60.16bit.raw'
        # coincidencia exacta de nombre o con prefijo de clave (p.ej. 'C4F03_...')
        exacto = coords_dir / f"{base_name}.txt"
        if exacto.exists():
            pares.append((gz, exacto))
            continue
        candidatos = [c for c in coords_dir.glob(f"*{base_name}.txt") if c != gz]
        if candidatos:
            pares.append((gz, candidatos[0]))
        else:
            print(f"[AVISO] sin coordenadas para {gz.name}")
    return pares


def procesar_caso(base):
    """Procesa una carpeta de caso. Frames de revision solo para las primeras
    N_CASOS_FRAMES grabaciones de la carpeta.

    Dos disposiciones de salida:
      - HOMOLOG_INPLACE: por cada gzip se escribe, JUNTO al archivo de nariz,
        '<stem>_mejillas.txt' (info, homologo de Nose information) y
        '<stem>_mejillas_coords.txt' (cajas + flags). Frames -> 'Mejillas frames/'.
      - subcarpetas (por defecto): 'Cheek information TXT/' y 'Cheek coordinates TXT/'.
    """
    pares = emparejar_archivos(base)
    modo = "ESTRICTO" if MODO_ESTRICTO else "PERMISIVO"

    if not HOMOLOG_INPLACE:
        out_base = base if OUTPUT_ROOT is None else (OUTPUT_ROOT / base.name)
        out_base.mkdir(parents=True, exist_ok=True)
        out_coords_dir = out_base / "Cheek coordinates TXT"
        out_info_dir = out_base / "Cheek information TXT"
        out_coords_dir.mkdir(exist_ok=True)
        out_info_dir.mkdir(exist_ok=True)
    frames_root = None
    if GUARDAR_FRAMES:
        carpeta_frames = "Mejillas frames" if HOMOLOG_INPLACE else "Cheek frames"
        frames_root = base / carpeta_frames if HOMOLOG_INPLACE else \
            (base if OUTPUT_ROOT is None else (OUTPUT_ROOT / base.name)) / carpeta_frames
        frames_root.mkdir(parents=True, exist_ok=True)

    print(f"[{base.name}] {len(pares)} video(s) | modo {modo} | "
          f"frames para las primeras {N_CASOS_FRAMES} grabaciones")
    for i, (gz, coords) in enumerate(pares):
        fr = frames_root if (GUARDAR_FRAMES and i < N_CASOS_FRAMES) else None
        if HOMOLOG_INPLACE:
            # salida junto al gzip / archivo de nariz, mismo nombre base
            info_file = gz.with_name(f"{gz.stem}_mejillas.txt")
            coords_file = gz.with_name(f"{gz.stem}_mejillas_coords.txt")
        else:
            info_file = out_info_dir / f"{gz.stem}.txt"
            coords_file = out_coords_dir / f"{gz.stem}.txt"
        # reanudable: saltar lo ya procesado
        if SKIP_EXISTING and info_file.exists() and coords_file.exists():
            print(f"    {gz.name}: ya procesado, se omite")
            continue
        # resiliente: un video que falle no detiene el lote
        try:
            n_con, n_flag, n_sin = procesar_video(gz, coords, info_file, coords_file, fr)
            total = n_con + n_flag + n_sin
            marca = "  [con frames]" if fr is not None else ""
            print(f"    {gz.name}: {total} frames -> {n_con} con mejilla, "
                  f"{n_flag} dudosa, {n_sin} sin nariz{marca}")
        except Exception as e:
            print(f"    [ERROR] {gz.name}: {type(e).__name__}: {e}")


def es_caso(path):
    """Un caso es una carpeta con GZIP: subcarpeta 'Thermal GZIP' o .gzip directos."""
    if (path / "Thermal GZIP").is_dir():
        return True
    return any(path.glob("*.gzip"))


def descubrir_casos(path):
    """Si 'path' es un caso lo devuelve; si no, busca casos en sus subcarpetas."""
    if es_caso(path):
        return [path]
    return sorted(d for d in path.iterdir() if d.is_dir() and es_caso(d))


def main():
    aplicar_modo()
    casos = descubrir_casos(BASE_RESULTS)
    if not casos:
        print(f"No se encontraron casos (carpetas con 'Thermal GZIP') en {BASE_RESULTS}")
        return
    modo = "ESTRICTO" if MODO_ESTRICTO else "PERMISIVO"
    print(f"{len(casos)} caso(s) | modo {modo} "
          f"(skin>={MIN_SKIN_FRACTION}, asim<={ASYM_THRESHOLD_C}, perfil={PROFILE_RATIO})\n")
    for base in casos:
        procesar_caso(base)
    print("\nListo.")


if __name__ == "__main__":
    main()
