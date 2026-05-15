@echo off
REM ─────────────────────────────────────────────────────────────────────
REM  Entrenamiento modelo NOSE  -  v2
REM  Dataset : dataset_nose  (~8,470 imágenes)
REM  GPU     : RTX 4070 Ti Super (16 GB VRAM)
REM ─────────────────────────────────────────────────────────────────────

python train.py --weights yolov5m.pt --data nose_v2.yaml --hyp hyp_nose_v2.yaml --epochs 100 --batch-size 32 --imgsz 640 --device 0 --workers 4 --image-weights --cos-lr --patience 20 --project runs/train --name nose_v2

echo.
echo Entrenamiento nose finalizado. Revisa runs/train/nose_v2/
pause
