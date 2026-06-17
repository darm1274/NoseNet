# -*- coding: utf-8 -*-
"""100 frames visuales: 50 BUENOS (estricto OK) + 50 MALOS (descartados:
asimetria / ROI fuera), en subcarpetas bueno/ y malo/."""
import csv
import random
from pathlib import Path

import numpy as np
import cv2
import extraer_mejillas as em

ROOT = Path(r"D:\Usuarios\Labneurociencias\Documents\Chile\ESTUDIO_1_COORDENADAS_NARIZ")
NOSE = Path(r"D:\Usuarios\Labneurociencias\Documents\Chile\Chile\Modelos\NoseNet\COORDENADAS_NARIZ")
OUT = Path(__file__).parent / "muestra_bueno_malo"
(OUT / "bueno").mkdir(parents=True, exist_ok=True)
(OUT / "malo").mkdir(parents=True, exist_ok=True)
random.seed(11)
N_LADO = 50
MAXG = 2
ASYM_OK = 0.8     # |T_L - T_R| max para bueno (mediana real ~0.62 C)
YALIGN = 2        # desfase vertical max de cajas L/R (px)


def bueno_estricto(fl, L, R, Lm, Rm):
    yLc = (L[1] + L[3]) / 2.0
    yRc = (R[1] + R[3]) / 2.0
    return fl == "OK" and abs(Lm - Rm) <= ASYM_OK and abs(yLc - yRc) <= YALIGN


def leer(c):
    out = {}
    with open(c, newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r, None)
        for p in r:
            if len(p) < 16:
                continue
            try:
                idx = int(p[0]); L = tuple(int(p[i]) for i in (1, 2, 3, 4))
                R = tuple(int(p[i]) for i in (5, 6, 7, 8))
                Lm, Rm = float(p[9]), float(p[10]); fl = p[15].strip()
            except ValueError:
                continue
            out[idx] = (fl, L, R, Lm, Rm)
    return out


buenos, malos = [], []
for estr in ROOT.glob("*/*_mejillas_sim_coords.txt"):
    movie = estr.parent.name
    stem = estr.name.replace("_mejillas_sim_coords.txt", "")
    fe = leer(estr)
    ok = [i for i, v in fe.items() if bueno_estricto(v[0], v[1], v[2], v[3], v[4])]
    bad = [i for i, v in fe.items() if not bueno_estricto(v[0], v[1], v[2], v[3], v[4])]
    for i in random.sample(ok, min(MAXG, len(ok))):
        buenos.append((movie, stem, i) + fe[i])
    for i in random.sample(bad, min(MAXG, len(bad))):
        malos.append((movie, stem, i) + fe[i])

random.shuffle(buenos); random.shuffle(malos)
buenos = buenos[:N_LADO]; malos = malos[:N_LADO]


def render(lista, carpeta, es_bueno):
    lista.sort(key=lambda c: (c[0], c[1]))
    cur = None; arr = w = h = nose = None; n = 0
    for movie, stem, idx, fl, L, R, Lm, Rm in lista:
        if (movie, stem) != cur:
            gz = ROOT / movie / (stem + ".gzip")
            if not gz.exists():
                continue
            arr, w, h = em.leer_gzip(gz)
            nf = NOSE / (stem + ".txt")
            nose = em.parse_nose_coords(nf) if nf.exists() else {}
            cur = (movie, stem)
        if idx >= arr.shape[0]:
            continue
        nb = nose.get(idx); nbox = nb[:4] if nb else None
        if es_bueno:
            estado = "OK"
        elif fl != "OK":
            estado = fl                       # ROI_FUERA / ASIMETRIA estricta / PERFIL
        elif abs(Lm - Rm) > ASYM_OK:
            estado = "ASIM_LEVE"              # 0.5 < |L-R| <= 1.0
        else:
            estado = "DESALINEADA"            # cajas L/R a distinta altura
        border = (0, 200, 0) if es_bueno else (0, 140, 255)
        label = f"{movie[:9]} f{idx} | {estado}  L:{Lm:.1f} R:{Rm:.1f}"
        img = em.dibujar_frame(arr[idx], w, h, nose=nbox, boxes=(L, R),
                               label=label, color_borde=border)
        cv2.putText(img, stem[:18], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
        fn = f"{movie}__{stem[:22]}__f{idx:04d}__{estado.replace('|', '+')}.png".replace(" ", "_")
        cv2.imwrite(str(carpeta / fn), img); n += 1
    return n


nb = render(buenos, OUT / "bueno", True)
nm = render(malos, OUT / "malo", False)
print(f"BUENOS: {nb}  MALOS: {nm}  -> {OUT.name}/")
