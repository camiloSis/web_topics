from __future__ import annotations

import argparse
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
import pandas as pd

# Año de referencia para "citas por año de vida". Ajustar si se quiere congelar.
REF_YEAR = datetime.now().year

# ---------------------------------------------------------------------------
# 1) Utilidades de texto
# ---------------------------------------------------------------------------
def strip_accents(s: str) -> str:
    """Quita tildes/diacríticos preservando longitud conceptual."""
    if not isinstance(s, str):
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def norm(s) -> str:
    """minúsculas + sin tildes + espacios colapsados."""
    if s is None:
        return ""
    s = strip_accents(str(s)).lower()
    return " ".join(s.split())


def contains_any(text: str, terms: list[str]) -> int:
    """Cuenta cuántos términos distintos aparecen como subcadena."""
    if not text:
        return 0
    return sum(1 for t in terms if t in text)


# ---------------------------------------------------------------------------
# 2) Diccionarios de términos (ya normalizados: minúsculas y sin tildes)
#    >>> AJUSTA AQUÍ si quieres afinar la sensibilidad temática <<<
# ---------------------------------------------------------------------------
RNI_TERMS = [
    "radiacion no ionizante", "radiaciones no ionizantes",
    "non-ionizing radiation", "non ionizing radiation",
    "radiofrecuencia", "radio frequency", "radiofrequency",
    "microondas", "microwave",
    "wifi", "wi-fi", "5g", "4g",
    "telefonia movil", "mobile phone", "cell phone", "cellular",
    "antena", "antennas", "estacion base", "base station",
    "lineas electricas", "power lines",
    "campos electromagneticos", "electromagnetic fields", "emf",
    "elf", "extremely low frequency", "baja frecuencia",
    "campos magneticos", "magnetic field",
    "pantalla", "screen", "monitor",
    "luz azul", "blue light",
    "telecomunicaciones", "telecommunications",
]

SALUD_TERMS = [
    "salud", "health",
    "efectos biologicos", "biological effects", "biological effect",
    "efectos en salud", "health effects", "health effect",
    "cancer", "oncolog", "oncology", "tumor", "glioma", "leucemia", "leukemia",
    "neurolog", "neurological", "cerebr", "brain", "cognitiv",
    "estres oxidativo", "oxidative stress",
    "reproductiv", "reproductive", "fertilidad", "fertility", "esperma", "sperm",
    "sueno", "sleep", "insomnio", "insomnia",
    "ritmo circadiano", "circadian",
    "ocular", "eye", "visual", "vision", "retina", "miopia", "myopia",
    "sintoma", "symptom", "cefalea", "headache",
    "hipersensibilidad", "hypersensitivity",
    "electrohipersensibilidad", "electromagnetic hypersensitivity",
    "embarazo", "pregnancy", "gestante",
    "nino", "children", "adolescente",
]

OBJETIVO_TERMS: dict[str, list[str]] = {
    "O1": [  # Fuentes de RNI en entorno cotidiano y laboral
        "telefonia movil", "mobile phone", "cell phone", "wifi", "wi-fi", "5g",
        "antena", "base station", "lineas electricas", "power lines",
        "pantalla", "screen", "exposicion cotidiana", "daily exposure",
        "entorno laboral", "workplace", "fuente de rni", "source of nir",
        "telecomunicaciones", "telecommunications",
    ],
    "O2": [  # Efectos biológicos y en salud de RF y ELF (población adulta)
        "efectos biologicos", "biological effects",
        "efectos en salud", "health effects",
        "neurolog", "neurological", "oxidativo", "oxidative",
        "reproductiv", "reproductive", "sueno", "sleep",
        "oncolog", "oncology", "cancer", "glioma", "leucemia", "leukemia",
        "adult",
    ],
    "O3": [  # Dispositivos, pantallas y luz azul
        "pantalla", "screen", "luz azul", "blue light",
        "ocular", "eye", "vision", "visual", "retina",
        "circadiano", "circadian", "ritmo", "myopia", "miopia",
    ],
    "O4": [  # Límites de exposición, normativa y prevención
        "limite", "limit", "icnirp", "oms", "who",
        "normativa", "regulation", "prevencion", "prevention",
        "peru", "peruvian", "estandar", "standard", "guideline", "norma",
    ],
    "O5": [  # Poblaciones vulnerables
        "nino", "children", "child", "adolescente", "adolescent",
        "gestante", "pregnant", "embarazo", "pregnancy",
        "vulnerable", "fetal", "fetus", "feto",
    ],
}

OBJETIVO_DESC = {
    "O1": "Fuentes de RNI en entorno cotidiano y laboral",
    "O2": "Efectos biológicos y en salud de RF y ELF",
    "O3": "Dispositivos, pantallas y luz azul",
    "O4": "Límites de exposición, normativa y prevención",
    "O5": "Poblaciones vulnerables",
}

# Reglas para C3 (rigor del diseño) — se evalúan en orden descendente.
# >>> AJUSTA AQUÍ si tus tipo_documento difieren del vocabulario Scopus habitual <<<
C3_RULES: list[tuple[int, list[str]]] = [
    (5, ["revision sistematica", "systematic review", "metaanalisis",
         "meta-analysis", "meta analysis", "cohort", "cohorte",
         "ensayo aleatorizado", "ensayo controlado", "randomized controlled",
         "randomised controlled", "controlled trial", "rct"]),
    (4, ["revision narrativa", "narrative review", "casos y controles",
         "case-control", "case control", "review", "revision"]),
    (3, ["experimental", "in vitro", "in vivo", "transversal",
         "cross-sectional", "cross sectional", "longitudinal"]),
    (2, ["descriptivo", "descriptive", "medicion", "measurement",
         "estudio de caso", "case study", "case report",
         "observacional", "observational"]),
    (1, ["opinion", "comment", "comentario", "hipotesis", "hypothesis",
         "letter", "editorial", "note"]),
]

# Pesos base (suman 1.0)
WEIGHTS_BASE = {"C1": 0.25, "C2": 0.20, "C3": 0.20, "C4": 0.10,
                "C5": 0.10, "C6": 0.10, "C7": 0.05}

# Pesos para SciELO si --sciel-impacto=redistribuir
# (su 5% de C5 se reparte: +5% a C1 y +5% a C3)
WEIGHTS_SCIELO_REDIST = {"C1": 0.30, "C2": 0.20, "C3": 0.25, "C4": 0.10,
                          "C5": 0.00, "C6": 0.10, "C7": 0.05}


# ---------------------------------------------------------------------------
# 3) Cálculo de cada criterio (devuelve puntaje 1..5)
# ---------------------------------------------------------------------------
def c1_pertinencia(row: pd.Series) -> int:
    """C1 (25%) — Pertinencia temática RNI + Salud."""
    title = norm(row.get("titulo"))
    kw = norm(row.get("palabras_clave"))
    abstract = norm(row.get("resumen"))

    rni_t = contains_any(title, RNI_TERMS) > 0
    sal_t = contains_any(title, SALUD_TERMS) > 0
    rni_k = contains_any(kw, RNI_TERMS) > 0
    sal_k = contains_any(kw, SALUD_TERMS) > 0
    rni_a = contains_any(abstract, RNI_TERMS) > 0
    sal_a = contains_any(abstract, SALUD_TERMS) > 0

    if rni_t and sal_t:
        return 5
    if (rni_t and (sal_k or sal_a)) or (sal_t and (rni_k or rni_a)):
        return 4
    if (rni_k and sal_k) or (rni_a and sal_a) or \
       ((rni_k or rni_a) and (sal_k or sal_a)):
        return 3
    if rni_t or sal_t or rni_k or sal_k or rni_a or sal_a:
        return 2
    return 1


def c2_alineacion(row: pd.Series) -> tuple[int, str]:
    """C2 (20%) — Afinidad temática con cada objetivo + bonus por objetivo_origen."""
    title = norm(row.get("titulo"))
    kw = norm(row.get("palabras_clave"))
    abstract = norm(row.get("resumen"))

    hits: dict[str, int] = {}
    for obj, terms in OBJETIVO_TERMS.items():
        # Título pesa doble
        hits[obj] = (2 * contains_any(title, terms)
                     + contains_any(kw, terms)
                     + contains_any(abstract, terms))

    max_hits = max(hits.values())
    if max_hits == 0:
        best_obj = "O1"  # fallback determinista
    else:
        # Desempate: preferir un objetivo declarado en objetivo_origen
        origen = norm(row.get("objetivo_origen"))
        origen_list = [o.strip().upper() for o in origen.split(";") if o.strip()]
        candidates = [o for o, h in hits.items() if h == max_hits]
        best_obj = next((c for c in candidates if c in origen_list), candidates[0])

    h = hits[best_obj]
    if h == 0:
        base = 1
    elif h == 1:
        base = 2
    elif h == 2:
        base = 3
    else:
        base = 4

    origen = norm(row.get("objetivo_origen"))
    origen_list = [o.strip().upper() for o in origen.split(";") if o.strip()]
    if best_obj in origen_list:
        base = min(5, base + 1)

    return base, best_obj


def c3_rigor(row: pd.Series) -> int:
    """C3 (20%) — Rigor del diseño a partir de tipo_documento."""
    tipo = norm(row.get("tipo_documento"))
    if not tipo:
        return 3  # neutro si no hay dato (p.ej. SciELO)
    for score, keys in C3_RULES:
        for k in keys:
            if k in tipo:
                return score
    return 3  # desconocido → neutro


def c4_actualidad(row: pd.Series) -> int:
    """C4 (10%) — Actualidad por rangos bienales."""
    try:
        y = int(row.get("anio"))
    except (TypeError, ValueError):
        return 1
    if y <= 2017:
        return 1
    if y <= 2019:
        return 2
    if y <= 2021:
        return 3
    if y <= 2023:
        return 4
    return 5


def c5_impacto(cites, anio, is_scielo: bool, sciel_mode: str,
               quartiles: dict | None) -> int:
    """C5 (10%) — Citas por año de vida en cuartiles. 0 citas → 1."""
    if is_scielo:
        return 3 if sciel_mode == "neutro" else 0  # redistribuir: peso=0
    if cites is None:
        return 3  # sin dato → neutro
    if cites == 0:
        return 1
    if quartiles is None:
        return 3  # sin cuartiles calculables → neutro
    try:
        y = int(anio)
    except (TypeError, ValueError):
        return 3
    years = max(1, REF_YEAR - y)
    cpy = cites / years
    if cpy <= quartiles["q1"]:
        return 2
    if cpy <= quartiles["q2"]:
        return 3
    if cpy <= quartiles["q3"]:
        return 4
    return 5


def c6_calidad_fuente(row: pd.Series, cuartiles_map: dict) -> int:
    """C6 (10%) — Q1/Q2=5, Q3/Q4=3, sin cuartil=3 (neutro)."""
    revista = norm(row.get("revista"))
    if not revista or not cuartiles_map:
        return 3
    cuartil = cuartiles_map.get(revista)
    if cuartil is None:
        # Búsqueda parcial por si el nombre viene truncado
        for k, v in cuartiles_map.items():
            if k and (k in revista or revista in k):
                cuartil = v
                break
    if cuartil is None:
        return 3
    cuartil = cuartil.upper().strip()
    if cuartil in ("Q1", "Q2"):
        return 5
    if cuartil in ("Q3", "Q4"):
        return 3
    return 3


def c7_verificabilidad(row: pd.Series) -> int:
    """C7 (5%) — DOI + OA=5; DOI solo=4; enlace+OA=3; enlace solo=1."""
    doi = str(row.get("doi") or "").strip()
    url = str(row.get("url") or "").strip()
    is_scielo = norm(row.get("base_origen")) == "scielo"
    oa = bool(row.get("_open_access", False)) or is_scielo

    has_doi, has_url = bool(doi), bool(url)

    if has_doi and oa:
        return 5
    if has_doi:
        return 4
    if has_url and oa:
        return 3
    if has_url:
        return 1
    return 1


# ---------------------------------------------------------------------------
# 4) Carga de fuentes auxiliares
# ---------------------------------------------------------------------------
def load_scopus(scopus_dir: str | None) -> dict[str, dict]:
    """Devuelve {EID: {'citas': int, 'open_access': bool}}."""
    records: dict[str, dict] = {}
    if not scopus_dir:
        return records
    path = Path(scopus_dir)
    if not path.is_dir():
        print(f"[warn] scopus-dir no existe: {path}")
        return records

    for csv_file in sorted(path.glob("Scopus_O*.csv")):
        try:
            df = pd.read_csv(csv_file, encoding="utf-8-sig", dtype=str)
        except Exception as e:
            print(f"[warn] no se pudo leer {csv_file.name}: {e}")
            continue

        # Limpieza de encabezados (BOM, espacios, comillas sobrantes)
        df.columns = [c.strip().strip('"').strip() for c in df.columns]

        def find_col(*substrings: str) -> str | None:
            for c in df.columns:
                lc = c.lower()
                if all(s in lc for s in substrings):
                    return c
            return None

        eid_col = next((c for c in df.columns if c.lower() == "eid"), None)
        cite_col = find_col("cited", "by") or find_col("citedby")
        oa_col = find_col("open", "access")

        if not eid_col:
            continue

        for _, r in df.iterrows():
            eid = str(r.get(eid_col, "")).strip().strip('"').strip()
            if not eid:
                continue

            cites = 0
            if cite_col is not None and pd.notna(r.get(cite_col)):
                raw = str(r[cite_col]).strip().strip('"').strip()
                try:
                    cites = int(float(raw))
                except ValueError:
                    cites = 0

            oa = False
            if oa_col is not None and pd.notna(r.get(oa_col)):
                v = str(r[oa_col]).strip().strip('"').strip().lower()
                oa = bool(v) and v not in ("no", "none", "false", "closed")

            records[eid] = {"citas": cites, "open_access": oa}

    return records


def load_cuartiles(cuartiles_path: str | None) -> dict[str, str]:
    """Devuelve {revista_normalizada: 'Q1'..'Q4'}."""
    out: dict[str, str] = {}
    if not cuartiles_path:
        return out
    p = Path(cuartiles_path)
    if not p.is_file():
        print(f"[warn] cuartiles no encontrado: {p}")
        return out
    try:
        df = pd.read_csv(p, encoding="utf-8-sig", dtype=str).fillna("")
    except Exception as e:
        print(f"[warn] no se pudo leer cuartiles: {e}")
        return out
    for _, r in df.iterrows():
        rev = norm(r.get("revista", ""))
        cu = str(r.get("cuartil", "")).strip().upper()
        if rev and cu:
            out[rev] = cu
    return out


# ---------------------------------------------------------------------------
# 5) Cuartiles de "citas por año de vida" (para C5)
# ---------------------------------------------------------------------------
def compute_c5_quartiles(cpy_series: pd.Series) -> dict | None:
    valid = cpy_series.dropna()
    valid = valid[valid > 0]
    if len(valid) < 4:
        return None
    return {
        "q1": float(valid.quantile(0.25)),
        "q2": float(valid.quantile(0.50)),
        "q3": float(valid.quantile(0.75)),
    }


# ---------------------------------------------------------------------------
# 6) Selección de los 20 (mínimo 4 por objetivo)
# ---------------------------------------------------------------------------
def select_top20(df_sorted: pd.DataFrame) -> tuple[list[tuple[int, str]], dict[str, int]]:
    """Asume df_sorted ya ordenado por puntaje desc. Devuelve [(index, motivo), ...]."""
    selected: list[tuple[int, str]] = []
    selected_idx: set[int] = set()
    counts = {f"O{i}": 0 for i in range(1, 6)}

    # Pase 1: cuota mínima de 4 por objetivo
    for obj in ["O1", "O2", "O3", "O4", "O5"]:
        if len(selected) >= 20:
            break
        for idx, row in df_sorted.iterrows():
            if len(selected) >= 20 or counts[obj] >= 4:
                break
            if idx in selected_idx:
                continue
            if row["objetivo_principal"] != obj:
                continue
            selected.append((idx, f"Cupo mínimo {obj} (puesto {counts[obj] + 1}/4)"))
            selected_idx.add(idx)
            counts[obj] += 1

    # Pase 2: completar con los mejores puntajes restantes
    for idx, row in df_sorted.iterrows():
        if len(selected) >= 20:
            break
        if idx in selected_idx:
            continue
        obj = row["objetivo_principal"]
        selected.append((idx, f"Mejor puntaje restante ({obj})"))
        selected_idx.add(idx)
        counts[obj] = counts.get(obj, 0) + 1

    return selected, counts


# ---------------------------------------------------------------------------
# 7) Pipeline principal
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Selecciona los 20 mejores artículos RNI-salud por matriz ponderada.",
    )
    ap.add_argument("--depurados", required=True,
                    help="CSV de artículos depurados (utf-8-sig).")
    ap.add_argument("--scopus-dir", default=None,
                    help="Carpeta con Scopus_O1..O5.csv (opcional).")
    ap.add_argument("--cuartiles", default=None,
                    help="CSV revista,cuartil (opcional).")
    ap.add_argument("--sciel-impacto", choices=["neutro", "redistribuir"],
                    default="neutro",
                    help="Cómo tratar C5 en SciELO (default: neutro).")
    ap.add_argument("--salida", default=".",
                    help="Carpeta donde escribir los CSV de salida.")
    args = ap.parse_args()

    outdir = Path(args.salida)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Carga principal ----
    dep_path = Path(args.depurados)
    if not dep_path.is_file():
        print(f"[error] no existe --depurados: {dep_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(dep_path, encoding="utf-8-sig", dtype=str).fillna("")
    print(f"Artículos cargados: {len(df)}")

    # ---- Fuentes auxiliares ----
    scopus = load_scopus(args.scopus_dir)
    print(f"Registros Scopus cargados: {len(scopus)}")
    cuartiles_map = load_cuartiles(args.cuartiles)
    print(f"Revistas con cuartil: {len(cuartiles_map)}")

    # ---- Enriquecer con citas y OA ----
    def get_citas(row):
        if norm(row.get("base_origen")) == "scielo":
            return None
        rec = scopus.get(str(row.get("id_original", "")).strip())
        return rec["citas"] if rec else None

    def get_oa(row):
        if norm(row.get("base_origen")) == "scielo":
            return True
        rec = scopus.get(str(row.get("id_original", "")).strip())
        return bool(rec and rec["open_access"])

    df["_citas"] = df.apply(get_citas, axis=1)
    df["_open_access"] = df.apply(get_oa, axis=1)

    # ---- Cuartiles C5 (citas por año) ----
    cpy_list = []
    for _, r in df.iterrows():
        if norm(r.get("base_origen")) == "scielo":
            continue
        c = r["_citas"]
        if c is None or c == 0:
            continue
        try:
            y = int(r.get("anio"))
        except (TypeError, ValueError):
            continue
        cpy_list.append(c / max(1, REF_YEAR - y))
    c5_quarts = compute_c5_quartiles(pd.Series(cpy_list, dtype=float))
    if c5_quarts:
        print(f"C5 cuartiles (cpy): {c5_quarts}")
    else:
        print("[warn] no se pudieron calcular cuartiles C5; se usará neutro=3.")

    # ---- Scoring ----
    scored_rows = []
    for _, row in df.iterrows():
        is_scielo = norm(row.get("base_origen")) == "scielo"

        c1 = c1_pertinencia(row)
        c2, obj = c2_alineacion(row)
        c3 = c3_rigor(row)
        c4 = c4_actualidad(row)
        c5 = c5_impacto(row["_citas"], row.get("anio"),
                        is_scielo, args.sciel_impacto, c5_quarts)
        c6 = c6_calidad_fuente(row, cuartiles_map)
        c7 = c7_verificabilidad(row)

        if is_scielo and args.sciel_impacto == "redistribuir":
            w = WEIGHTS_SCIELO_REDIST
        else:
            w = WEIGHTS_BASE

        total = (w["C1"] * c1 + w["C2"] * c2 + w["C3"] * c3 +
                 w["C4"] * c4 + w["C5"] * c5 + w["C6"] * c6 +
                 w["C7"] * c7) / 5.0 * 100.0

        d = row.to_dict()
        d.update({
            "C1": c1, "C2": c2, "C3": c3, "C4": c4,
            "C5": c5, "C6": c6, "C7": c7,
            "puntaje_total": round(total, 2),
            "objetivo_principal": obj,
            "citas": row["_citas"] if row["_citas"] is not None else 0,
        })
        scored_rows.append(d)

    result = pd.DataFrame(scored_rows)

    # ---- Ranking + selección ----
    result["_anio_num"] = pd.to_numeric(result["anio"], errors="coerce").fillna(0)
    result = result.sort_values(
        by=["puntaje_total", "_anio_num", "citas"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    result["ranking_general"] = range(1, len(result) + 1)

    selected, counts = select_top20(result)

    result["seleccionado"] = False
    result["motivo_seleccion"] = "No seleccionado (fuera de top 20)"
    for idx, motivo in selected:
        result.at[idx, "seleccionado"] = True
        result.at[idx, "motivo_seleccion"] = motivo

    # ---- Salidas ----
    base_cols = [c for c in df.columns if not c.startswith("_")]
    new_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7",
                "puntaje_total", "objetivo_principal", "citas",
                "ranking_general", "seleccionado", "motivo_seleccion"]
    out_cols = base_cols + [c for c in new_cols if c not in base_cols]

    ranking_path = outdir / "ranking_completo.csv"
    result[out_cols].to_csv(ranking_path, index=False, encoding="utf-8-sig")

    top20 = result[result["seleccionado"]].copy().reset_index(drop=True)
    top20["ranking_seleccion"] = range(1, len(top20) + 1)
    top20_path = outdir / "top20_seleccionados.csv"
    top20[["ranking_seleccion"] + out_cols].to_csv(
        top20_path, index=False, encoding="utf-8-sig")

    # Resumen por objetivo
    resumen_rows = []
    for obj in ["O1", "O2", "O3", "O4", "O5"]:
        disp = result[result["objetivo_principal"] == obj]
        sel = disp[disp["seleccionado"]]
        resumen_rows.append({
            "objetivo": obj,
            "descripcion": OBJETIVO_DESC[obj],
            "n_disponibles": len(disp),
            "n_seleccionados": len(sel),
            "puntaje_min": round(sel["puntaje_total"].min(), 2) if len(sel) else None,
            "puntaje_max": round(sel["puntaje_total"].max(), 2) if len(sel) else None,
            "puntaje_promedio": round(sel["puntaje_total"].mean(), 2) if len(sel) else None,
        })
    resumen_path = outdir / "resumen_seleccion.csv"
    pd.DataFrame(resumen_rows).to_csv(resumen_path, index=False, encoding="utf-8-sig")

    # ---- Consola ----
    print("\n=== RESUMEN ===")
    print(f"Total artículos: {len(result)}")
    print(f"Seleccionados  : {len(top20)}")
    print("\nDistribución por objetivo_principal (disponibles / seleccionados):")
    for r in resumen_rows:
        print(f"  {r['objetivo']} {OBJETIVO_DESC[r['objetivo']]:<45} "
              f"{r['n_disponibles']:>4} / {r['n_seleccionados']:>2}")
    print("\nTop 5 por puntaje total:")
    for _, r in top20.head(5).iterrows():
        print(f"  {r['ranking_seleccion']:>2}. [{r['puntaje_total']:>5.2f}] "
              f"{r['objetivo_principal']} | {r['anio']} | {str(r['titulo'])[:70]}")
    print(f"\nArchivos escritos en: {outdir.resolve()}")
    print(f"  - {ranking_path.name}")
    print(f"  - {top20_path.name}")
    print(f"  - {resumen_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())