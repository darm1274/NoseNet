# -*- coding: utf-8 -*-
"""Filtra la deteccion de mejillas descartando frames asimetricos / fuera de piel
(bueno = flag estricto 'OK') y calcula la CONTINUIDAD por video.

Genera, por grabacion y junto al archivo de nariz:
  <stem>_mejillas_filtrado.txt        (info, solo frames buenos)
  <stem>_mejillas_filtrado_coords.txt (coords, solo frames buenos)
Y un CSV global 'continuidad_mejillas.csv' con, por video:
  total_frames, detectadas(nariz), buenas(OK), sin_nariz, asimetria, roi_fuera,
  perfil, pct_continuidad (buenas/total), racha_max (frames buenos consecutivos).

Todo se deriva de los TXT ya generados (no se re-descomprime el gzip).
"""
import csv
import re
from pathlib import Path

ROOT = Path(r"D:\Usuarios\Labneurociencias\Documents\Chile\ESTUDIO_1_COORDENADAS_NARIZ")
OUT_CSV = Path(__file__).parent / "continuidad_mejillas.csv"

# bueno = sobre piel (flag estricto OK) Y simetrico de verdad:
ASYM_OK = 0.8     # |T_L - T_R| maximo en C (mediana real ~0.62 C)
YALIGN = 2        # desfase vertical maximo entre cajas L/R en px


def frame_de(nombre):
    m = re.search(r"frame(\d+)", nombre)
    return int(m.group(1)) if m else None


def total_frames(nose_txt):
    """nro de frames del video = lineas del .txt de nariz - cabecera."""
    with open(nose_txt, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def leer_coords(coords_file):
    """lista de (idx, flag, Lm, Rm, yLc, yRc) en orden de archivo."""
    filas = []
    with open(coords_file, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for p in r:
            if len(p) < 16:
                continue
            try:
                idx = int(p[0])
                yLc = (int(p[2]) + int(p[4])) / 2.0
                yRc = (int(p[6]) + int(p[8])) / 2.0
                Lm, Rm = float(p[9]), float(p[10])
            except ValueError:
                continue
            filas.append((idx, p[15].strip(), Lm, Rm, yLc, yRc))
    return filas


def es_bueno(fl, Lm, Rm, yLc, yRc):
    """bueno = sobre piel (OK) + |L-R|<=ASYM_OK + cajas alineadas."""
    return (fl == "OK" and abs(Lm - Rm) <= ASYM_OK and abs(yLc - yRc) <= YALIGN)


def racha_maxima(indices_buenos):
    """longitud de la racha mas larga de frames buenos CONSECUTIVOS (idx, idx+1...)."""
    if not indices_buenos:
        return 0
    s = sorted(indices_buenos)
    mejor = cur = 1
    for a, b in zip(s, s[1:]):
        cur = cur + 1 if b == a + 1 else 1
        mejor = max(mejor, cur)
    return mejor


def filtrar_info(info_file, buenos_set, dest):
    """copia las filas del info cuyo frame esta en buenos_set."""
    out = []
    with open(info_file, encoding="utf-8") as f:
        out.append(f.readline())  # cabecera
        for line in f:
            nombre = line.split(",", 1)[0]
            idx = frame_de(nombre)
            if idx in buenos_set:
                out.append(line)
    dest.write_text("".join(out), encoding="utf-8")


def main():
    filas_csv = []
    for nose_txt in sorted(ROOT.glob("*/*.txt")):
        if "_mejillas" in nose_txt.name:
            continue
        stem = nose_txt.stem
        movie = nose_txt.parent.name
        estr_coords = nose_txt.with_name(stem + "_mejillas_sim_coords.txt")
        estr_info = nose_txt.with_name(stem + "_mejillas_sim.txt")
        if not estr_coords.exists() or not estr_info.exists():
            continue

        tot = total_frames(nose_txt)
        coords = leer_coords(estr_coords)
        detectadas = len(coords)
        buenos = [idx for idx, fl, Lm, Rm, yL, yR in coords
                  if es_bueno(fl, Lm, Rm, yL, yR)]
        asim = sum(1 for _, fl, *_ in coords if "ASIMETRIA" in fl)
        roif = sum(1 for _, fl, *_ in coords if "ROI_FUERA" in fl)
        perf = sum(1 for _, fl, *_ in coords if "PERFIL" in fl)
        # descartes nuevos dentro de los antes-'OK':
        asim_leve = sum(1 for _, fl, Lm, Rm, yL, yR in coords
                        if fl == "OK" and abs(Lm - Rm) > ASYM_OK)
        desalin = sum(1 for _, fl, Lm, Rm, yL, yR in coords
                      if fl == "OK" and abs(Lm - Rm) <= ASYM_OK and abs(yL - yR) > YALIGN)
        sin_nariz = tot - detectadas
        buenas = len(buenos)
        pct = 100.0 * buenas / tot if tot else 0.0
        racha = racha_maxima(buenos)

        # escribir filtrados (solo buenos) junto a la nariz
        buenos_set = set(buenos)
        filtrar_info(estr_info, buenos_set,
                     nose_txt.with_name(stem + "_mejillas_filtrado.txt"))
        # coords filtrado
        with open(estr_coords, encoding="utf-8") as f:
            head = f.readline()
            rows = [l for l in f if l.split(",", 1)[0].isdigit()
                    and int(l.split(",", 1)[0]) in buenos_set]
        nose_txt.with_name(stem + "_mejillas_filtrado_coords.txt").write_text(
            head + "".join(rows), encoding="utf-8")

        # sujeto = prefijo C#X## del stem
        msub = re.match(r"([Cc]\d[FM]?\w*?)_", stem)
        sujeto = msub.group(1) if msub else stem[:6]
        filas_csv.append(dict(
            pelicula=movie, sujeto=sujeto, archivo=stem,
            total=tot, detectadas=detectadas, buenas=buenas,
            sin_nariz=sin_nariz, asimetria=asim, roi_fuera=roif, perfil=perf,
            asim_leve=asim_leve, desalineada=desalin,
            pct_continuidad=round(pct, 1), racha_max=racha,
            pct_racha=round(100.0 * racha / tot, 1) if tot else 0.0,
        ))

    cols = ["pelicula", "sujeto", "archivo", "total", "detectadas", "buenas",
            "sin_nariz", "asimetria", "roi_fuera", "perfil", "asim_leve",
            "desalineada", "pct_continuidad", "racha_max", "pct_racha"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(filas_csv)

    # resumen consola
    n = len(filas_csv)
    if n:
        prom = sum(r["pct_continuidad"] for r in filas_csv) / n
        peor = sorted(filas_csv, key=lambda r: r["pct_continuidad"])[:5]
        print(f"videos: {n} | continuidad media: {prom:.1f}%")
        print("peores 5 (menor continuidad):")
        for r in peor:
            print(f"  {r['pelicula'][:12]:12s} {r['archivo'][:24]:24s} "
                  f"{r['pct_continuidad']:5.1f}%  (buenas {r['buenas']}/{r['total']})")
    print(f"CSV -> {OUT_CSV}")


if __name__ == "__main__":
    main()
