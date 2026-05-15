@echo off
REM ─────────────────────────────────────────────────────────────────────
REM  Entrenamiento modelo FACE / NO_FACE  -  v2
REM  Dataset : dataset_chile  (~10,705 imágenes)
REM  GPU     : RTX 4070 Ti Super (16 GB VRAM)
REM ─────────────────────────────────────────────────────────────────────

python train.py --weights yolov5m.pt --data face_no_face_v2.yaml --hyp hyp_face_v2.yaml --epochs 100 --batch-size 32 --imgsz 640 --device 0 --workers 4 --image-weights --cos-lr --patience 20 --project runs/train --name face_no_face_v2

echo.
echo Entrenamiento face finalizado. Revisa runs/train/face_no_face_v2/
pause
