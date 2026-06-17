# NoseNet

Pipeline de detección de nariz en imágenes térmicas basado en YOLOv5. Desarrollado para estudios de neurociencia que requieren seguimiento de temperatura nasal a partir de secuencias de video térmico (formato `.gzip` de cámara FLIR / Seek o `.mp4`).

Incluye además un módulo de **detección de mejillas** ([extraer_mejillas.py](extraer_mejillas.py)) que extrae la temperatura de ambas mejillas a partir de las coordenadas de nariz ya detectadas, **sin necesidad de un modelo adicional**, usando geometría anclada al recuadro de la nariz + refinamiento térmico. Ver [Detección de mejillas](#detección-de-mejillas).

---

## Descripción general

NoseNet implementa un pipeline de dos etapas:

1. **Detección de rostro** — Un modelo YOLOv5 clasifica cada frame como `face` o `no_face`. Las regiones de rostro detectadas son enmascaradas (píxeles fuera del bounding box se ponen a cero) para enfocar la siguiente etapa.
2. **Detección de nariz** — Un segundo modelo YOLOv5 localiza la nariz dentro del recorte de rostro. Las coordenadas del bounding box y la confianza se guardan en CSV para análisis posterior.

```
Archivo .gzip / .mp4
        │
        ▼
   Extracción de frames (PNG)
        │
        ▼
   Modelo Face/No-Face (YOLOv5)
        │           │
      face        no_face
        │
        ▼
   Enmascarado de región facial
        │
        ▼
   Modelo Nose (YOLOv5)
        │
        ▼
   Coordenadas + imágenes anotadas + reporte
```

---

## Estructura del repositorio

```
NoseNet/
├── NoseNet.py                    # Pipeline principal de inferencia
├── NoseNetbyFilesMarkets.py      # Variante experimental de procesamiento
├── preparar_dataset.py           # Preparación y augmentación de datasets
├── extraer_frames_etiquetado.py  # Extracción balanceada de frames para etiquetado
├── generar_reporte.py            # Generación de reportes Excel multi-hoja
├── classes.txt                   # Definición de clases
│
│   # ── Módulo de mejillas ──────────────────────────────────────────────────
├── extraer_mejillas.py           # Extracción de temperatura de mejillas (sin modelo extra)
├── generar_reporte_temperatura.py# Reporte Excel nariz vs mejilla por video
├── mejillas_procesar_lote.py     # Orquestador por carpeta (aislamiento de memoria + reintentos)
├── mejillas_continuidad.py       # Filtra "buenos" + métricas de continuidad por video
├── mejillas_muestra_qc.py        # Muestra visual de control: frames buenos / malos
├── pull.bat                      # Atajo para actualizar el repo (git pull)
│
├── weights/                      # Pesos entrenados (ver sección de descarga)
│   ├── face_no_face/
│   │   └── best.pt               # Detector cara/no-cara (YOLOv5, ~40 MB)
│   └── nose/
│       └── best.pt               # Detector de nariz (YOLOv5, ~40 MB)
│
├── Models/
│   ├── yolov5/                   # Código fuente YOLOv5 (cara/no-cara)
│   └── yolov5_nose/              # Código fuente YOLOv5 (nariz)
│
├── data/                         # Generado en tiempo de ejecución
│   ├── initial_data/             # Frames extraídos de gzip/mp4
│   ├── face_no_face/             # Frames clasificados por cara
│   └── nose/                     # Frames con detección de nariz
│
├── results/                      # Imágenes anotadas + TXT de coordenadas
├── reports/                      # Reportes de sesión (.txt) y Excel (.xlsx)
└── requirements.txt
```

---

## Requisitos

- Python 3.9 o superior
- GPU NVIDIA con CUDA recomendada (funciona en CPU pero mucho más lento)
- VRAM mínima recomendada: 4 GB

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

> **Nota:** PyTorch con soporte CUDA debe instalarse de forma separada según tu versión de CUDA. Consulta [pytorch.org/get-started](https://pytorch.org/get-started/locally/) y selecciona la combinación correcta.
>
> Ejemplo para CUDA 11.8:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Descarga de pesos

Los archivos de pesos no se incluyen en el repositorio por su tamaño. Descárgalos desde la sección **[Releases](../../releases)** y colócalos en las rutas indicadas:

| Archivo | Ruta destino | Descripción |
|---|---|---|
| `face_no_face_best.pt` | `weights/face_no_face/best.pt` | Detector cara/no-cara — YOLOv5, entrenado en imágenes térmicas |
| `nose_best.pt` | `weights/nose/best.pt` | Detector de nariz — YOLOv5, entrenado en regiones faciales térmicas |

```
NoseNet/
└── weights/
    ├── face_no_face/
    │   └── best.pt   ← aquí
    └── nose/
        └── best.pt   ← aquí
```

---

## Uso

### Pipeline principal

```bash
python NoseNet.py
```

Al ejecutar se abrirá un diálogo para:
1. Seleccionar la **carpeta raíz** que contiene los archivos `.gzip` y/o `.mp4` (la búsqueda es recursiva).
2. Confirmar si activar **preprocesamiento** (escala de grises + CLAHE). Recomendado solo si el modelo fue entrenado con imágenes en blanco y negro. Los modelos `v2` se entrenaron en color — seleccionar **NO**.

Los resultados se escriben en:

```
results/<clave>/
├── frames/               # Imágenes originales con bounding box de nariz
└── nose_detections.txt   # CSV: imagen, xmin, ymin, xmax, ymax, confianza

reports/
└── reporte_YYYYMMDD_HHMMSS.txt   # Resumen de la sesión
```

### Configuración de batch size

Si tienes errores de memoria (OOM), reduce `BATCH_SIZE` al inicio de `NoseNet.py`:

```python
BATCH_SIZE = 64   # 12+ GB VRAM
BATCH_SIZE = 32   # 8 GB VRAM
BATCH_SIZE = 16   # 6 GB VRAM
BATCH_SIZE = 8    # 4 GB VRAM
```

### Generación de reporte Excel

Una vez procesados los datos, ejecuta:

```bash
python generar_reporte.py
```

Genera `reporte_NoseNet.xlsx` con cuatro hojas:
- **Detalle** — estadísticas por caso
- **Por grupo** — agrupación por condición experimental
- **Por segmento** — 6 períodos temporales
- **Por sujeto** — comparativa S1 / S2

### Extracción de frames para etiquetado

Para preparar un conjunto de imágenes para anotación manual con herramientas como LabelImg o CVAT:

```bash
python extraer_frames_etiquetado.py
```

Parámetros editables al inicio del script:
- `TOTAL_FRAMES = 1000` — total de frames a extraer
- `MIN_POR_VIDEO = 5` — mínimo de frames por video
- `SEED = 42` — semilla de aleatoriedad para reproducibilidad

### Preparación de dataset de entrenamiento

```bash
python preparar_dataset.py
```

Integra un dataset externo (formato YOLO con 3 clases: `face`, `nose`, `no_face`) y lo divide en dos datasets especializados:
- `dataset_chile/` — clasificación binaria cara/no-cara (70/20/10 train/val/test)
- `dataset_nose/` — detección de nariz con enmascarado facial

---

## Formato de datos de entrada

### Archivos `.gzip` (cámara térmica)

Formato propietario de captura térmica de 80×60 píxeles:

```
Bytes 0–3   : Header — uint16[0]=ancho, uint16[1]=alto
Bytes 4+    : Frames consecutivos — uint16 por píxel (valor × 100 = temperatura en Kelvin × 100)
```

La conversión a 8 bits aplica calibración de temperatura:
```
pixel_8bit = clip((raw/100 - 293.15) × (255/25), 0, 255)
```
Rango efectivo: 293.15 K (20 °C) a 318.15 K (45 °C).

### Archivos `.mp4`

Video estándar. Se extraen todos los frames con OpenCV (`frame_stride=1`). Para reducir la cantidad de frames, editar el parámetro `frame_stride` en `procesar_archivo()`.

---

## Modelos

Ambos modelos son YOLOv5 entrenados en imágenes térmicas de rostros humanos en condiciones controladas de laboratorio.

| Modelo | Versión | Clases | Umbral confianza |
|---|---|---|---|
| Face / No-Face | v23 | `face` (0), `no_face` (1) | 0.5 |
| Nose | v2 | `nose` (0) | 0.5 |

El umbral de confianza puede modificarse en `NoseNet.py`:

```python
model_face.conf = 0.5
model_nose.conf = 0.5
```

---

## Requisitos de hardware recomendados

| Componente | Mínimo | Recomendado |
|---|---|---|
| CPU | 4 núcleos | 8+ núcleos |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU mode) | NVIDIA 4 GB+ VRAM |
| Almacenamiento | 10 GB libres | 50+ GB (datos) |

---

## Dependencias principales

| Paquete | Versión mínima | Uso |
|---|---|---|
| `torch` | 1.8.0 | Inferencia GPU/CPU |
| `torchvision` | 0.9.0 | Transformaciones de imagen |
| `opencv-python` | 4.1.1 | Carga/escritura de imágenes y video |
| `numpy` | 1.23.5 | Procesamiento de arrays |
| `tqdm` | 4.66.3 | Barras de progreso |
| `openpyxl` | — | Generación de reportes Excel |
| `Pillow` | 10.3.0 | Soporte adicional de formatos |

---

## Estructura de salida detallada

```
data/
├── initial_data/
│   ├── gzip_converted_<clave>/    # Frames extraídos de archivos .gzip
│   └── mp4_converted_<clave>/     # Frames extraídos de archivos .mp4
├── face_no_face/
│   └── <clave>/
│       ├── face/                  # Frames con rostro (región enmascarada)
│       └── no_face/               # Frames sin rostro detectado
└── nose/
    └── <clave>/                   # Frames con bounding box de nariz dibujado

results/
└── <clave>/
    ├── frames/                    # Imágenes originales anotadas con bounding box
    └── nose_detections.txt        # CSV de coordenadas y confianza

reports/
└── reporte_YYYYMMDD_HHMMSS.txt   # Log de sesión con OK/ADVERTENCIA/ERROR
```

La `<clave>` se construye a partir de la ruta relativa del archivo de entrada, usando `__` como separador de directorios (ej.: `caso01__sesion1__video`).

---

## Detección de mejillas

Módulo que extrae la temperatura de **ambas mejillas** reutilizando las coordenadas de nariz ya detectadas — **no entrena ni usa un modelo adicional**. A la resolución térmica (80×63 px) las mejillas no tienen bordes propios fuertes, pero están en posición fija respecto a la nariz, así que se ubican por geometría + termografía.

### Cómo funciona

1. **Geometría anclada a los bordes de la nariz.** Cada ROI de mejilla se coloca a partir del recuadro de la nariz: pegada a su borde lateral (puede solaparlo levemente, ≤15%), con ancho ≈0.8× el de la nariz, a la altura del pómulo (debajo de los ojos). Parámetros: `CHEEK_INNER`, `CHEEK_W`, `CHEEK_TOP`, `CHEEK_BOT`.
2. **Refinamiento térmico simétrico (solo vertical).** Ambas ROIs se deslizan juntas en vertical para centrarse sobre la piel del pómulo, manteniéndose a la misma altura y simétricas (el horizontal queda fijo a la nariz, para no derivar hacia ojos/nariz).
3. **Temperatura.** Se calcula `weighted_temp` con el **mismo kernel gaussiano que la nariz**, de modo que nariz y mejilla son directamente comparables.
4. **Control de calidad por frame** — banderas: `ROI_FUERA` (mejilla con poca piel, por postura), `ASIMETRIA` (|T_izq−T_der| sobre el umbral), `PERFIL` (cabeza girada). Un frame es **bueno** si está sobre piel, alineado y con asimetría ≤ 0.8 °C (la asimetría real mediana entre mejillas es ≈ 0.62 °C).

### Uso

```bash
# 1. Extracción (configura BASE_RESULTS, COORDS_DIR, etc. al inicio del archivo)
python extraer_mejillas.py

# 2. Lote grande con aislamiento de memoria (un proceso por carpeta + reintentos)
python mejillas_procesar_lote.py

# 3. Filtra los frames "buenos" y calcula continuidad por video -> CSV
python mejillas_continuidad.py

# 4. Muestra visual de control (frames buenos vs malos)
python mejillas_muestra_qc.py

# 5. Reporte Excel nariz vs mejilla
python generar_reporte_temperatura.py
```

### Configuración (al inicio de `extraer_mejillas.py`)

| Parámetro | Descripción |
|---|---|
| `BASE_RESULTS` | Carpeta de caso, o raíz con varios casos (se procesan todos) |
| `COORDS_DIR` | Carpeta externa con las coordenadas de caja de nariz (si no están junto al gzip) |
| `OUTPUT_ROOT` | Redirige las salidas fuera de los datos crudos (`None` = junto a los datos) |
| `HOMOLOG_INPLACE` | Escribe `<archivo>_mejillas.txt` junto a cada archivo de nariz |
| `MODO_ESTRICTO` | `False` permisivo / `True` estricto (más exigente con asimetría y postura) |
| `REFINAR_SIMETRICO` | Refinamiento vertical simétrico (recomendado) |
| `SKIP_EXISTING` | Reanuda un lote interrumpido sin reprocesar |
| `GUARDAR_FRAMES`, `N_CASOS_FRAMES` | Exporta imágenes de revisión para los primeros N casos |

### Salidas

```
<stem>_mejillas.txt           # image, L_roi_temp, L_weighted_temp, R_roi_temp, R_weighted_temp
<stem>_mejillas_coords.txt    # cajas L/R + temperaturas (°C) + n_skin + flags
<stem>_mejillas_filtrado.txt  # solo frames "buenos"
continuidad_mejillas.csv      # por video: total, buenas, %continuidad, racha_max, ...
```

> **Formato GZIP.** Estos archivos usan cabecera simple de 4 bytes (ancho, alto) seguida de frames `uint16` consecutivos (sin framerate ni timestamps por frame). El lector de `extraer_mejillas.py` asume ese formato.

---

## Licencia

Este proyecto es parte de una investigación en neurociencias. Contacta a los autores antes de usar o distribuir los datos o modelos entrenados.
