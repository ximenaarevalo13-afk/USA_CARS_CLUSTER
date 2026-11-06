"""
segmentador.py
Lógica del modelo de segmentación estratégica de inventario (US AutoMarket).

Reproduce el pipeline del notebook `USA_cars_Segmentacion_Estrategica.ipynb`:
reglas de calidad -> regla de gobernanza (salvage) -> winsorización p1-p99 ->
RobustScaler -> PCA (>= 90% de varianza) -> K-Means K=5 (SEED = 50) -> nombres por perfil.

El módulo no depende de Streamlit: lo usan tanto `entrenar_modelo.py` (genera el .pkl)
como `app.py` (interfaz).
"""
from __future__ import annotations

import datetime as _dt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

# ----------------------------------------------------------------------------- constantes
SEED = 50
ANIO_REF = 2020                     # año máximo del dataset: antigüedad = 2020 - year
K_FINAL = 5
FEATURES = ["price", "antiguedad", "mileage"]
DBSCAN_EPS, DBSCAN_MIN_SAMPLES = 0.28, 10
COLUMNAS_MINIMAS = ["price", "brand", "year", "title_status", "mileage"]

REVISION = "Revisión de datos"
SALVAGE = "Mercado Salvage"
ORDEN_SEG = ["Gama Alta", "Seminuevo Premium", "Seminuevo de Volumen", "Uso Medio", "Alto Recorrido", SALVAGE]
COLORES = {"Gama Alta": "#6250d6", "Seminuevo Premium": "#2a78d6", "Seminuevo de Volumen": "#1baf7a",
           "Uso Medio": "#eda100", "Alto Recorrido": "#eb6834", SALVAGE: "#e87ba4", REVISION: "#9a9893"}

PALANCAS = {
    "Gama Alta": ("Pricing firme",
                  "Tasación individual por comparables y descuento máximo autorizado de 2%; los listados "
                  "marcados como anómalos pasan por revisión manual de precio antes de publicarse.",
                  "Descuento realizado vs publicado; días hasta primera oferta"),
    "Seminuevo Premium": ("Priorización de anuncios",
                          "Posición destacada en búsquedas y retargeting; precio alineado a mercado sin "
                          "descuentos automáticos.",
                          "Leads por anuncio destacado vs no destacado"),
    "Seminuevo de Volumen": ("Promoción de alcance",
                             "Campañas de performance de alto alcance con precio competitivo revisado "
                             "semanalmente contra el benchmark del modelo.",
                             "Costo por lead y tasa de conversión"),
    "Uso Medio": ("Confianza",
                  "Inspección certificada + reporte de historial del VIN en el anuncio para sostener el "
                  "precio pese al kilometraje.",
                  "Precio realizado vs publicado; conversión anuncio → contacto"),
    "Alto Recorrido": ("Rotación",
                       "Escalera de rebajas programada (−3% cada 15 días sin oferta) con precio piso; sin "
                       "inversión en anuncios destacados.",
                       "Días en inventario; % vendido antes de 45 días"),
    SALVAGE: ("Gestión del riesgo",
              "Canal B2B / subasta para reconstructores, divulgación obligatoria del título y exclusión "
              "de campañas retail.",
              "% vendido en canal B2B; reclamos y devoluciones"),
    REVISION: ("Calidad de datos",
               "No se segmenta: corregir el dato (precio, kilometraje, duplicado o vehículo fuera de "
               "alcance) y volver a asignar.",
               "% de listados en revisión por lote"),
}

UMBRALES_MONITOREO = {          # (verde <, rojo >) ; entre ambos = ámbar
    "PSI vs referencia": (0.10, 0.25),
    "% clean fuera del rango de entrenamiento": (0.02, 0.05),
    "% revisión de datos": (0.03, 0.06),
}


# ----------------------------------------------------------------------------- preparación
def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza texto, tipos y crea la antigüedad. No elimina registros."""
    faltan = [c for c in COLUMNAS_MINIMAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas obligatorias: {', '.join(faltan)}")
    out = df.copy()
    for c in ["brand", "title_status", "vin", "model", "state", "color"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip().str.lower()
    for c in ["price", "year", "mileage"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["antiguedad"] = ANIO_REF - out["year"]
    return out


def motivo_revision(df: pd.DataFrame) -> pd.Series:
    """Primera regla de calidad que cumple cada registro ('' si ninguna)."""
    reglas = [
        ("dato_faltante_o_no_numerico", df[["price", "year", "mileage"]].isna().any(axis=1)),
        ("precio_cero", df["price"] <= 0),
        ("mileage_centinela", df["mileage"] == 999999),
        ("fuera_alcance_peterbilt", df["brand"] == "peterbilt"),
        ("vin_duplicado", df["vin"].duplicated(keep="first") if "vin" in df.columns else pd.Series(False, index=df.index)),
        ("mileage_cero_implausible", (df["mileage"] == 0) & (df["antiguedad"] >= 2)),
    ]
    motivo = pd.Series("", index=df.index, dtype=object)
    for nombre, mask in reglas:
        motivo[(motivo == "") & mask.fillna(False)] = nombre
    return motivo


def mercado(df: pd.DataFrame, motivo: pd.Series) -> pd.Series:
    return pd.Series(np.select([motivo != "", df["title_status"] == "salvage insurance"],
                               ["Revisión", "Salvage"], default="Clean"), index=df.index)


def _winsorizar(X: pd.DataFrame, caps: dict) -> np.ndarray:
    W = X[FEATURES].astype(float).copy()
    for v, (lo, hi) in caps.items():
        W[v] = W[v].clip(lo, hi)
    return W.to_numpy()


def _nombrar(clean: pd.DataFrame, labels: np.ndarray) -> dict:
    """Mismo criterio del notebook: nombres por perfil, no por número de cluster."""
    pc = clean.assign(c=labels).groupby("c").agg(pm=("price", "median"), km=("mileage", "mean"),
                                                 ag=("antiguedad", "mean"))
    c_rec = pc["km"].idxmax(); r = pc.drop(c_rec)
    c_uso = r["ag"].idxmax(); r = r.drop(c_uso).sort_values("pm", ascending=False)
    nombres = {int(r.index[0]): "Gama Alta", int(r.index[1]): "Seminuevo Premium",
               int(r.index[2]): "Seminuevo de Volumen", int(c_uso): "Uso Medio", int(c_rec): "Alto Recorrido"}
    assert len(nombres) == K_FINAL
    return nombres


def psi(esperado, actual, eps: float = 1e-4) -> float:
    """Population Stability Index con piso epsilon para evitar ln(0) y divisiones por cero."""
    e = np.clip(np.asarray(esperado, float), eps, None); a = np.clip(np.asarray(actual, float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def semaforo(indicador: str, valor: float) -> str:
    verde, rojo = UMBRALES_MONITOREO[indicador]
    return "verde" if valor < verde else ("rojo" if valor > rojo else "ámbar")


# ----------------------------------------------------------------------------- entrenamiento
def entrenar(df_raw: pd.DataFrame) -> dict:
    """Entrena el pipeline completo y devuelve un diccionario serializable (solo numpy, dicts y sklearn)."""
    df = normalizar(df_raw)
    motivo = motivo_revision(df)
    merc = mercado(df, motivo)
    clean = df[merc == "Clean"].copy()

    caps = {v: (float(clean[v].quantile(.01)), float(clean[v].quantile(.99))) for v in ["antiguedad", "mileage"]}
    caps = {"price": (float(clean.price.min()), float(clean.price.max())), **caps}
    caps = {v: caps[v] for v in FEATURES}
    Xw = _winsorizar(clean, caps)
    scaler = RobustScaler().fit(Xw)
    Xs = scaler.transform(Xw)
    pca_full = PCA(random_state=SEED).fit(Xs)
    n_pc = int(np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.90) + 1)
    pca = PCA(n_components=n_pc, random_state=SEED).fit(Xs)
    Z = pca.transform(Xs)
    km = KMeans(n_clusters=K_FINAL, n_init=10, random_state=SEED).fit(Z)
    nombres = _nombrar(clean, km.labels_)

    # DBSCAN: para un vehículo nuevo, es "anómalo" si queda a más de eps de todo punto núcleo
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(Z)
    nucleos = Z[db.core_sample_indices_]

    seg_clean = np.array([nombres[int(c)] for c in km.labels_])
    seg_all = pd.Series(REVISION, index=df.index, dtype=object)
    seg_all[merc == "Salvage"] = SALVAGE
    seg_all[clean.index] = seg_clean
    seg_df = df.assign(segmento=seg_all)[seg_all != REVISION]
    anom = pd.Series(db.labels_ == -1, index=clean.index)
    perfil = []
    for s in ORDEN_SEG:
        g = seg_df[seg_df.segmento == s]
        perfil.append({
            "segmento": s, "vehiculos": int(len(g)), "participacion": len(g) / len(seg_df),
            "precio_mediano": float(g.price.median()), "precio_p25": float(g.price.quantile(.25)),
            "precio_p75": float(g.price.quantile(.75)), "antiguedad_promedio": float(g.antiguedad.mean()),
            "km_mediano": float(g.mileage.median()), "valor_inventario": float(g.price.sum()),
            "pct_anomalia_dbscan": float(anom.reindex(g.index).mean()) if s != SALVAGE else None,
        })
    total_valor = sum(p["valor_inventario"] for p in perfil)
    for p in perfil:
        p["participacion_valor"] = p["valor_inventario"] / total_valor

    return {
        "version_modelo": "1.0", "fecha_entrenamiento": _dt.date.today().isoformat(),
        "seed": SEED, "anio_ref": ANIO_REF, "k": K_FINAL, "n_componentes": n_pc,
        "varianza_explicada": [float(v) for v in pca.explained_variance_ratio_],
        "silueta": float(silhouette_score(Z, km.labels_)),
        "sklearn_version": sklearn.__version__,
        "features": FEATURES, "caps": caps,
        "rango_ref": {v: (float(clean[v].min()), float(clean[v].max())) for v in FEATURES},
        "scaler": scaler, "pca": pca, "kmeans": km, "nombres": nombres,
        "dbscan_eps": DBSCAN_EPS, "dbscan_nucleos": nucleos,
        "orden_segmentos": ORDEN_SEG, "colores": COLORES, "palancas": PALANCAS,
        "perfil": perfil,
        "dist_ref": {p["segmento"]: p["participacion"] for p in perfil},
        "mapa_ref": {"pc1": Z[:, 0].round(4), "pc2": Z[:, 1].round(4) if n_pc > 1 else np.zeros(len(Z)),
                     "segmento": seg_clean, "price": clean.price.to_numpy(),
                     "antiguedad": clean.antiguedad.to_numpy(), "mileage": clean.mileage.to_numpy()},
        "centroides": km.cluster_centers_,
        "marcas": sorted(df["brand"].dropna().unique().tolist()),
        "n_entrenamiento": {"total": int(len(df)), "clean": int(len(clean)),
                            "salvage": int((merc == "Salvage").sum()), "revision": int((merc == "Revisión").sum())},
    }


# ----------------------------------------------------------------------------- asignación
def asignar(df_raw: pd.DataFrame, modelo: dict) -> pd.DataFrame:
    """Asigna segmento a registros nuevos SIN reentrenar (centroide K-Means más cercano)."""
    df = normalizar(df_raw)
    motivo = motivo_revision(df)
    merc = mercado(df, motivo)
    out = df.copy()
    out["motivo_revision"] = motivo
    out["segmento"] = np.where(merc == "Salvage", SALVAGE, REVISION).astype(object)
    out["fuera_rango_entrenamiento"] = False
    out["valor_extremo_winsorizado"] = False
    out["anomalia_precio"] = False
    out["pc1"], out["pc2"] = np.nan, np.nan

    m = merc == "Clean"
    if m.any():
        clean = df[m]
        Zn = modelo["pca"].transform(modelo["scaler"].transform(_winsorizar(clean, modelo["caps"])))
        labels = modelo["kmeans"].predict(Zn)
        out.loc[m, "segmento"] = [modelo["nombres"][int(c)] for c in labels]
        out.loc[m, "pc1"] = Zn[:, 0]
        out.loc[m, "pc2"] = Zn[:, 1] if Zn.shape[1] > 1 else 0.0
        fuera = np.zeros(m.sum(), dtype=bool); extremo = np.zeros(m.sum(), dtype=bool)
        for v in FEATURES:
            lo, hi = modelo["rango_ref"][v]; clo, chi = modelo["caps"][v]
            fuera |= ((clean[v] < lo) | (clean[v] > hi)).to_numpy()
            extremo |= ((clean[v] < clo) | (clean[v] > chi)).to_numpy()
        out.loc[m, "fuera_rango_entrenamiento"] = fuera
        out.loc[m, "valor_extremo_winsorizado"] = extremo
        dist = NearestNeighbors(n_neighbors=1).fit(modelo["dbscan_nucleos"]).kneighbors(Zn)[0][:, 0]
        out.loc[m, "anomalia_precio"] = dist > modelo["dbscan_eps"]
    out["palanca"] = out["segmento"].map(lambda s: modelo["palancas"][s][0])
    return out


def monitoreo(asignado: pd.DataFrame, modelo: dict) -> dict:
    """Indicadores de estabilidad del lote frente a la población de entrenamiento."""
    seg = asignado.loc[asignado.segmento != REVISION, "segmento"]
    dist = seg.value_counts(normalize=True).reindex(ORDEN_SEG, fill_value=0)
    ref = pd.Series(modelo["dist_ref"]).reindex(ORDEN_SEG)
    clean = asignado[~asignado.segmento.isin([REVISION, SALVAGE])]
    ind = {
        "PSI vs referencia": psi(ref, dist) if len(seg) else float("nan"),
        "% clean fuera del rango de entrenamiento": float(clean.fuera_rango_entrenamiento.mean()) if len(clean) else 0.0,
        "% revisión de datos": float((asignado.segmento == REVISION).mean()),
    }
    return {"indicadores": ind, "semaforo": {k: semaforo(k, v) for k, v in ind.items() if np.isfinite(v)},
            "distribucion": pd.DataFrame({"referencia": ref, "lote": dist})}
