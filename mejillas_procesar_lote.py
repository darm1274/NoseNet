# -*- coding: utf-8 -*-
"""Reprocesa ESTUDIO_1_COORDENADAS_NARIZ con la geometria nueva (anclada a bordes,
sin solape con la nariz, sin ojos) + refinamiento vertical simetrico, modo estricto.
Un proceso por pelicula (aislamiento de memoria). Mueve a '_mejillas_sim'."""
import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(r"D:\Usuarios\Labneurociencias\Documents\Chile\ESTUDIO_1_COORDENADAS_NARIZ")
COORDS = r"D:\Usuarios\Labneurociencias\Documents\Chile\Chile\Modelos\NoseNet\COORDENADAS_NARIZ"
STAGING = Path(__file__).parent / "_sim_staging"

WORKER = (
    "import extraer_mejillas as em\n"
    "from pathlib import Path\n"
    "em.BASE_RESULTS=Path(r'''{base}''')\n"
    "em.COORDS_DIR=Path(r'''{co}''')\n"
    "em.OUTPUT_ROOT=Path(r'''{stg}''')\n"
    "em.MODO_ESTRICTO=True\n"
    "em.REFINAR_SIMETRICO=True\n"
    "em.HOMOLOG_INPLACE=False\n"
    "em.GUARDAR_FRAMES=False\n"
    "em.SKIP_EXISTING=True\n"
    "em.main()\n"
)

movies = sorted(d for d in ROOT.iterdir() if d.is_dir())
# reintenta cada pelicula hasta completar (el segfault es transitorio)
for m in movies:
    for intento in range(4):
        code = WORKER.format(base=m, co=COORDS, stg=STAGING)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        gz = list(m.glob("*.gzip"))
        done = [g for g in gz if (STAGING / m.name / "Cheek information TXT" / (g.stem + ".txt")).exists()]
        if len(done) == len(gz):
            print(f"[proc] {m.name}: OK {len(done)}/{len(gz)}", flush=True)
            break
        print(f"[proc] {m.name}: intento {intento+1} rc={r.returncode} {len(done)}/{len(gz)}", flush=True)

moved = 0
for m in movies:
    for src, suf in ((STAGING / m.name / "Cheek information TXT", "_mejillas_sim.txt"),
                     (STAGING / m.name / "Cheek coordinates TXT", "_mejillas_sim_coords.txt")):
        if not src.exists():
            continue
        for f in src.glob("*.txt"):
            dest = ROOT / m.name / (f.stem + suf)
            if dest.exists():
                continue
            shutil.move(str(f), str(dest))
            moved += 1
print(f"=== movidos: {moved} ===", flush=True)
print("=== FIN ===", flush=True)
