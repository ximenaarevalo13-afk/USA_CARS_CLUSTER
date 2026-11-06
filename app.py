"""
app.py — Segmentación estratégica de inventario · US AutoMarket
Ejecutar localmente:  streamlit run app.py
"""
import inspect
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

from segmentador import (ANIO_REF, COLUMNAS_MINIMAS, ORDEN_SEG, REVISION, SALVAGE, UMBRALES_MONITOREO,
                         asignar, entrenar, monitoreo)

BASE = Path(__file__).parent
RUTA_MODELO = BASE / "modelo_segmentacion.pkl"
RUTA_DATOS = BASE / "data" / "USA_cars_datasets.csv"

st.set_page_config(page_title="Segmentación de inventario · US AutoMarket", page_icon="🚗", layout="wide")

def ancho_completo(funcion):
    """Ocupa todo el ancho con cualquier versión de Streamlit (use_container_width fue reemplazado por width)."""
    parametros = inspect.signature(funcion).parameters
    return {"use_container_width": True} if "use_container_width" in parametros else {"width": "stretch"}


MOTIVOS = {
    "dato_faltante_o_no_numerico": "Precio, año o kilometraje vacío o no numérico.",
    "precio_cero": "Precio igual a 0: anuncio sin precio publicado; requiere tasación.",
    "mileage_centinela": "Kilometraje 999.999: valor de relleno del odómetro.",
    "fuera_alcance_peterbilt": "Tractocamión de carga pesada: fuera del alcance de autos usados.",
    "vin_duplicado": "VIN repetido en el lote: se conserva solo la primera aparición.",
    "mileage_cero_implausible": "0 millas en un vehículo con 2 años o más.",
}
SEMAFORO = {"verde": "🟢 Verde: mantener", "ámbar": "🟡 Ámbar: monitorear", "rojo": "🔴 Rojo: reentrenar"}


# ----------------------------------------------------------------------------- modelo
@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_modelo():
    """Carga el .pkl; si falta o fue creado con otra versión de scikit-learn, reentrena con el CSV del repo."""
    try:
        modelo = joblib.load(RUTA_MODELO)
        if modelo.get("sklearn_version") == sklearn.__version__:
            return modelo, "Archivo modelo_segmentacion.pkl"
        motivo = f"el .pkl se creó con scikit-learn {modelo.get('sklearn_version')} y el servidor usa {sklearn.__version__}"
    except Exception as e:  # archivo ausente o incompatible
        motivo = f"no se pudo leer el .pkl ({type(e).__name__})"
    df = pd.read_csv(RUTA_DATOS)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    return entrenar(df), f"Reentrenado al iniciar: {motivo}"


modelo, origen_modelo = cargar_modelo()
PERFIL = pd.DataFrame(modelo["perfil"]).set_index("segmento")
COLORES = modelo["colores"]
ESCALA = alt.Scale(domain=ORDEN_SEG[:-1], range=[COLORES[s] for s in ORDEN_SEG[:-1]])


def usd(v):
    return f"USD {v:,.0f}".replace(",", ".")


def num(v, d=0):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def chip(segmento):
    c = COLORES.get(segmento, "#9a9893")
    return (f"<span style='background:{c};color:white;padding:6px 14px;border-radius:16px;"
            f"font-weight:600;font-size:1.05rem'>{segmento}</span>")


def mapa_base():
    ref = pd.DataFrame(modelo["mapa_ref"])
    return alt.Chart(ref).mark_circle(size=22, opacity=0.45).encode(
        x=alt.X("pc1:Q", title="PC1 · desgaste (millas y antigüedad)"),
        y=alt.Y("pc2:Q", title="PC2 · nivel de precio"),
        color=alt.Color("segmento:N", scale=ESCALA, title="Segmento", sort=ORDEN_SEG[:-1]),
        tooltip=[alt.Tooltip("segmento:N", title="Segmento"), alt.Tooltip("price:Q", title="Precio", format=",.0f"),
                 alt.Tooltip("antiguedad:Q", title="Antigüedad"), alt.Tooltip("mileage:Q", title="Millas", format=",.0f")],
    )


# ----------------------------------------------------------------------------- encabezado
st.title("🚗 Segmentación estratégica de inventario")
st.caption("US AutoMarket · K-Means (K = 5) sobre PCA de precio, antigüedad y kilometraje + regla de gobernanza "
           "para títulos salvage. Asigna cada vehículo a un segmento con su palanca comercial.")

tab_resumen, tab_uno, tab_lote, tab_modelo = st.tabs(
    ["📊 Resumen de segmentos", "🔎 Clasificar un vehículo", "📁 Clasificar un lote (CSV)", "⚙️ Acerca del modelo"])

# ----------------------------------------------------------------------------- 1. resumen
with tab_resumen:
    n = modelo["n_entrenamiento"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Listados analizados", num(n["total"]))
    c2.metric("Segmentos operativos", "6", help="5 segmentos retail (K-Means) + Mercado Salvage (regla)")
    c3.metric("Valor publicado segmentado", usd(PERFIL.valor_inventario.sum()))
    c4.metric("En revisión de datos", num(n["revision"]), help="Listados que no se segmentan por problemas de calidad")

    tabla = pd.DataFrame({
        "Vehículos": PERFIL.vehiculos,
        "Participación": (PERFIL.participacion * 100).round(1).astype(str) + "%",
        "Precio mediano": PERFIL.precio_mediano.map(usd),
        "Antigüedad promedio": PERFIL.antiguedad_promedio.map(lambda v: f"{num(v, 1)} años"),
        "Millas medianas": PERFIL.km_mediano.map(num),
        "% del valor": (PERFIL.participacion_valor * 100).round(1).astype(str) + "%",
        "Palanca": [modelo["palancas"][s][0] for s in PERFIL.index],
    })
    st.subheader("Perfil de los segmentos")
    st.dataframe(tabla, **ancho_completo(st.dataframe))

    izq, der = st.columns([3, 2])
    with izq:
        st.subheader("Mapa del mercado clean")
        st.altair_chart(mapa_base().properties(height=420).interactive(), **ancho_completo(st.altair_chart))
    with der:
        st.subheader("Precio mediano por segmento")
        barras = PERFIL.reset_index()[["segmento", "precio_mediano"]]
        st.altair_chart(alt.Chart(barras).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("segmento:N", sort=ORDEN_SEG, title=None),
            x=alt.X("precio_mediano:Q", title="USD"),
            color=alt.Color("segmento:N", scale=alt.Scale(domain=ORDEN_SEG, range=[COLORES[s] for s in ORDEN_SEG]),
                            legend=None),
            tooltip=[alt.Tooltip("segmento:N"), alt.Tooltip("precio_mediano:Q", format=",.0f", title="Precio mediano")],
        ).properties(height=420), **ancho_completo(st.altair_chart))

    st.subheader("Palancas de acción")
    cols = st.columns(3)
    for i, s in enumerate(ORDEN_SEG):
        nombre, accion, kpi = modelo["palancas"][s]
        with cols[i % 3].container(border=True):
            st.markdown(chip(s), unsafe_allow_html=True)
            st.markdown(f"**{nombre}:** {accion}")
            st.caption(f"KPI: {kpi}")

# ----------------------------------------------------------------------------- 2. un vehículo
with tab_uno:
    st.subheader("Datos del vehículo")
    with st.form("form_vehiculo"):
        a, b, c = st.columns(3)
        precio = a.number_input("Precio publicado (USD)", min_value=0, max_value=500_000, value=18_000, step=500)
        anio = b.number_input("Año del modelo", min_value=1950, max_value=2030, value=2018, step=1)
        millas = c.number_input("Kilometraje (millas)", min_value=0, max_value=2_000_000, value=40_000, step=1_000)
        d, e = st.columns(2)
        titulo = d.selectbox("Estado del título", ["clean vehicle", "salvage insurance"],
                             format_func=lambda t: "Limpio (clean vehicle)" if t == "clean vehicle" else "Siniestro (salvage insurance)")
        marcas = modelo.get("marcas", ["ford"])
        marca = e.selectbox("Marca", marcas, index=marcas.index("ford") if "ford" in marcas else 0)
        enviar = st.form_submit_button("Asignar segmento", type="primary")

    if enviar:
        r = asignar(pd.DataFrame([{"price": precio, "brand": marca, "year": anio,
                                   "title_status": titulo, "mileage": millas}]), modelo).iloc[0]
        seg = r["segmento"]
        st.markdown(f"### Segmento asignado: {chip(seg)}", unsafe_allow_html=True)

        if seg == REVISION:
            st.error(f"**No se segmenta.** {MOTIVOS.get(r['motivo_revision'], r['motivo_revision'])}")
        else:
            p = PERFIL.loc[seg]
            m1, m2, m3 = st.columns(3)
            m1.metric("Precio vs mediana del segmento", usd(precio),
                      delta=f"{(precio / p.precio_mediano - 1) * 100:+.0f}% vs {usd(p.precio_mediano)}", delta_color="off")
            edad = ANIO_REF - anio
            m2.metric("Antigüedad", "posterior a 2020" if edad < 0 else f"{edad} año{'s' if edad != 1 else ''}", delta=f"promedio del segmento: {num(p.antiguedad_promedio, 1)}",
                      delta_color="off")
            m3.metric("Kilometraje", num(millas), delta=f"mediana del segmento: {num(p.km_mediano)}", delta_color="off")
            if not (p.precio_p25 <= precio <= p.precio_p75):
                st.info(f"El precio está fuera del rango central del segmento ({usd(p.precio_p25)} – {usd(p.precio_p75)}, "
                        "percentiles 25-75): conviene revisarlo frente a comparables.")

        nombre, accion, kpi = modelo["palancas"][seg]
        st.markdown(f"**Palanca recomendada — {nombre}:** {accion}")
        st.caption(f"KPI de seguimiento: {kpi}")

        if r["anomalia_precio"]:
            st.warning("⚠️ **Listado atípico:** queda lejos de las zonas densas del inventario (criterio DBSCAN). "
                       "Revisar el precio manualmente antes de publicar.")
        if r["fuera_rango_entrenamiento"]:
            st.warning("⚠️ Algún valor está fuera del rango observado al entrenar el modelo; la asignación es menos confiable.")
        elif r["valor_extremo_winsorizado"]:
            st.info("Kilometraje o antigüedad por encima del percentil 99 del entrenamiento: se recortó al umbral para asignarlo.")

        if seg not in (REVISION, SALVAGE):
            punto = pd.DataFrame({"pc1": [r["pc1"]], "pc2": [r["pc2"]]})
            capa = alt.Chart(punto).mark_point(shape="diamond", size=320, filled=True, color="black",
                                               stroke="white", strokeWidth=2).encode(x="pc1:Q", y="pc2:Q")
            st.altair_chart((mapa_base() + capa).properties(height=400, title="Ubicación del vehículo (rombo negro)"),
                            **ancho_completo(st.altair_chart))

# ----------------------------------------------------------------------------- 3. lote
with tab_lote:
    st.subheader("Asignar un lote de vehículos")
    st.markdown(f"El archivo CSV debe tener al menos las columnas: `{'`, `'.join(COLUMNAS_MINIMAS)}`. "
                "Si incluye `vin`, los VIN repetidos se envían a revisión. Las demás columnas se conservan.")
    ejemplo = pd.read_csv(RUTA_DATOS)
    ejemplo = ejemplo.drop(columns=[c for c in ejemplo.columns if c.startswith("Unnamed")]).sample(30, random_state=7)
    st.download_button("Descargar CSV de ejemplo (30 vehículos)", ejemplo.to_csv(index=False).encode("utf-8"),
                       "lote_ejemplo.csv", "text/csv")

    archivo = st.file_uploader("Cargar CSV", type=["csv"])
    if archivo is not None:
        try:
            lote = pd.read_csv(archivo)
            lote = lote.drop(columns=[c for c in lote.columns if c.startswith("Unnamed")])
            res = asignar(lote, modelo)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        mon = monitoreo(res, modelo)
        st.markdown(f"**{num(len(res))} registros procesados.**")
        cols = st.columns(3)
        formatos = {"PSI vs referencia": lambda v: num(v, 4)}
        for col, (ind, val) in zip(cols, mon["indicadores"].items()):
            texto = formatos.get(ind, lambda v: f"{num(v * 100, 1)}%")(val) if np.isfinite(val) else "n.d."
            col.metric(ind, texto)
            if ind in mon["semaforo"]:
                verde, rojo = UMBRALES_MONITOREO[ind]
                col.caption(f"{SEMAFORO[mon['semaforo'][ind]]} · umbrales {num(verde, 2)} / {num(rojo, 2)}")
        if len(res) < 200:
            st.caption("Con menos de 200 registros el PSI es poco estable: interprételo como referencia.")
        if "rojo" in mon["semaforo"].values():
            st.error("Al menos un indicador está en rojo: el lote difiere de la población de entrenamiento. "
                     "Revise la fuente y evalúe reentrenar el modelo (ver README).")

        dist = mon["distribucion"].reset_index(names="segmento").melt("segmento", var_name="población", value_name="pct")
        st.altair_chart(alt.Chart(dist).mark_bar().encode(
            x=alt.X("segmento:N", sort=ORDEN_SEG, title=None, axis=alt.Axis(labelAngle=-20)),
            xOffset=alt.XOffset("población:N", sort=["referencia", "lote"]),
            y=alt.Y("pct:Q", title="% de vehículos", axis=alt.Axis(format="%")),
            color=alt.Color("población:N", scale=alt.Scale(domain=["referencia", "lote"], range=["#9a9893", "#2a78d6"]),
                            title=None),
            tooltip=["segmento", "población", alt.Tooltip("pct:Q", format=".1%")],
        ).properties(height=330, title="Distribución por segmento: lote vs entrenamiento"), **ancho_completo(st.altair_chart))

        resumen = res.segmento.value_counts().reindex(ORDEN_SEG + [REVISION], fill_value=0)
        st.dataframe(resumen.rename("vehículos").to_frame().T, **ancho_completo(st.dataframe))

        visibles = [c for c in ["vin", "brand", "model", "year", "price", "mileage", "title_status", "segmento", "palanca",
                                "motivo_revision", "anomalia_precio", "fuera_rango_entrenamiento"] if c in res.columns]
        st.dataframe(res[visibles], **ancho_completo(st.dataframe), height=360)
        st.download_button("Descargar resultados (CSV)",
                           res.drop(columns=["pc1", "pc2"]).to_csv(index=False).encode("utf-8"),
                           "lote_segmentado.csv", "text/csv", type="primary")

# ----------------------------------------------------------------------------- 4. modelo
with tab_modelo:
    st.subheader("Cómo funciona")
    st.markdown(f"""
1. **Reglas de calidad:** precio 0, kilometraje 999.999, Peterbilt, VIN duplicado y 0 millas con 2+ años → *Revisión de datos*.
2. **Regla de gobernanza:** `title_status = salvage insurance` → *Mercado Salvage* (no entra al clustering).
3. **Variables:** precio, antigüedad (`{ANIO_REF} − year`) y kilometraje; antigüedad y kilometraje winsorizados en p1-p99.
4. **Escalado y reducción:** RobustScaler + PCA con {modelo['n_componentes']} componentes ({sum(modelo['varianza_explicada']):.1%} de varianza).
5. **Asignación:** centroide K-Means más cercano (K = {modelo['k']}); el nombre del segmento se asigna por perfil.
6. **Anomalías:** un listado es atípico si queda a más de eps = {modelo['dbscan_eps']} de todo punto núcleo de DBSCAN.
""")
    st.subheader("Ficha técnica")
    st.table(pd.DataFrame({"Valor": [
        modelo["version_modelo"], modelo["fecha_entrenamiento"], str(modelo["seed"]), f"{modelo['silueta']:.3f}",
        num(modelo["n_entrenamiento"]["clean"]), modelo["sklearn_version"], origen_modelo]},
        index=["Versión del modelo", "Fecha de entrenamiento", "Semilla", "Silueta (K-Means)",
               "Vehículos clean de entrenamiento", "scikit-learn del .pkl", "Origen del modelo cargado"]))
    st.subheader("Criterios de monitoreo y reentrenamiento")
    st.table(pd.DataFrame({
        "Verde (mantener)": ["< 0,10", "< 2%", "< 3%"],
        "Ámbar (monitorear)": ["0,10 – 0,25", "2% – 5%", "3% – 6%"],
        "Rojo (reentrenar)": ["> 0,25", "> 5%", "> 6% (revisar la fuente)"],
    }, index=list(UMBRALES_MONITOREO)))
    st.caption("Limitaciones: el dataset no tiene fechas, márgenes ni ventas; los segmentos describen el inventario "
               "publicado, no a los compradores. El modelo no distingue tipo de vehículo (pickup, sedán, SUV).")
