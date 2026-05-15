# Pesos de los modelos

Esta carpeta contiene los pesos entrenados de los modelos YOLOv5 usados por NoseNet.

Los archivos `.pt` **no se incluyen en el repositorio git** por su tamaño (~40 MB c/u).
Descárgalos desde la sección **[Releases](../../releases)** del repositorio y colócalos en las rutas indicadas.

## Estructura esperada

```
weights/
├── face_no_face/
│   └── best.pt      ← Detector cara / no-cara  (~40 MB)
└── nose/
    └── best.pt      ← Detector de nariz         (~40 MB)
```

## Detalles de los modelos

| Modelo | Archivo Release | Arquitectura | Clases | Dataset |
|---|---|---|---|---|
| Face / No-Face v23 | `face_no_face_best.pt` | YOLOv5 | `face`, `no_face` | Imágenes térmicas 80×60 px |
| Nose v2 | `nose_best.pt` | YOLOv5 | `nose` | Recortes faciales térmicos |

## Uso con Git LFS (opcional)

Si deseas versionar los pesos dentro del repositorio usando Git Large File Storage:

```bash
git lfs install
git lfs track "weights/**/*.pt"
git add .gitattributes
git add weights/
git commit -m "Add model weights via Git LFS"
```
