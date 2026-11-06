# 🚗 Segmentación estratégica de inventario — US AutoMarket

Aplicación en **Streamlit** que despliega el modelo de segmentación desarrollado en la actividad
*Segmentación Estratégica de Inventario (USA_cars_datasets)*. Asigna cada vehículo usado a un segmento
comercial y recomienda una palanca de acción (pricing, promoción, rotación o gestión del riesgo).

**Autora:** Yury Ximena Coronel Arevalo

---

## ¿Qué hace la app?

| Pestaña | Función |
|---|---|
| 📊 Resumen de segmentos | Perfil de los 6 segmentos, mapa PCA del mercado y palancas de acción |
| 🔎 Clasificar un vehículo | Ingresa precio, año, kilometraje, título y marca → segmento, palanca y alertas |
| 📁 Clasificar un lote (CSV) | Asigna un archivo completo, calcula indicadores de estabilidad (PSI) y permite descargar el resultado |
| ⚙️ Acerca del modelo | Pipeline, ficha técnica y criterios de reentrenamiento |

## Segmentos

| Segmento | Vehículos | Precio mediano | Palanca |
|---|---|---|---|
| Gama Alta | 209 | USD 45.000 | Pricing firme |
| Seminuevo Premium | 749 | USD 25.500 | Priorización de anuncios |
| Seminuevo de Volumen | 825 | USD 13.800 | Promoción de alcance |
| Uso Medio | 389 | USD 13.000 | Confianza (inspección certificada) |
| Alto Recorrido | 159 | USD 6.500 | Rotación (rebajas programadas) |
| Mercado Salvage | 116 | USD 1.900 | Gestión del riesgo (canal B2B) |

Además, 52 listados quedan en **Revisión de datos** (precio 0, kilometraje 999.999, Peterbilt, VIN duplicado
o 0 millas en vehículos de 2 años o más) y no se segmentan.

## Cómo funciona el modelo

1. **Reglas de calidad** → los registros con datos no confiables van a *Revisión de datos*.
2. **Regla de gobernanza** → `title_status = salvage insurance` forma el segmento *Mercado Salvage*.
3. **Variables del clustering**: precio, antigüedad (`2020 − year`) y kilometraje; antigüedad y kilometraje
   winsorizados en los percentiles 1 y 99.
4. **RobustScaler + PCA** con 2 componentes (91,9% de la varianza).
5. **K-Means con K = 5** (`random_state = 50`, `n_init = 10`); los nombres se asignan según el perfil de cada cluster.
6. **Detección de anomalías**: un listado es atípico si queda a más de `eps = 0,28` de los puntos núcleo de DBSCAN.

Los vehículos nuevos se asignan al **centroide más cercano, sin reentrenar**.

## Estructura del repositorio

```
├── app.py                      # Interfaz Streamlit
├── segmentador.py              # Lógica del modelo (reglas, entrenamiento, asignación, PSI)
├── entrenar_modelo.py          # Genera modelo_segmentacion.pkl
├── modelo_segmentacion.pkl     # Modelo entrenado (scaler, PCA, K-Means, perfiles)
├── data/
│   └── USA_cars_datasets.csv   # Datos de entrenamiento
├── requirements.txt
├── .streamlit/config.toml      # Tema visual
├── .gitignore
└── README.md
```

## Ejecutar localmente

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La app se abre en `http://localhost:8501`.

### Regenerar el modelo (.pkl)

```bash
python entrenar_modelo.py
```

Vuelva a generarlo si reentrena con datos nuevos o si cambia la versión de scikit-learn. Un `.pkl` solo es
compatible con la versión de scikit-learn que lo creó (aquí, **1.9.1**). Si el servidor usa otra versión,
la app reentrena automáticamente con `data/USA_cars_datasets.csv` al iniciar; lo indica en la pestaña
*Acerca del modelo*.

## Publicar en GitHub y Streamlit Community Cloud

1. **Crear el repositorio** en GitHub (por ejemplo, `segmentacion-inventario-usa-cars`), público o privado.
2. **Subir los archivos**:
   - Desde la web: *Add file → Upload files*, arrastrar todo el contenido de la carpeta (incluida la carpeta `data/`
     y `.streamlit/`) y confirmar con *Commit changes*.
   - O desde la terminal:
     ```bash
     git init
     git add .
     git commit -m "App de segmentación de inventario"
     git branch -M main
     git remote add origin https://github.com/<su-usuario>/segmentacion-inventario-usa-cars.git
     git push -u origin main
     ```
3. **Desplegar**: entrar a [share.streamlit.io](https://share.streamlit.io) con la cuenta de GitHub →
   *Create app* → elegir el repositorio, la rama `main` y el archivo principal `app.py`.
4. En **Advanced settings**, seleccionar **Python 3.12** (o 3.13) y pulsar *Deploy*.
5. La app queda publicada en una URL del tipo `https://<nombre>.streamlit.app`. Cada `git push` la actualiza.

> Si la instalación falla por la versión de scikit-learn, cambie en `requirements.txt` la línea
> `scikit-learn==1.9.1` por `scikit-learn>=1.5`: la app reentrenará el modelo al iniciar.

## Formato del CSV para clasificar lotes

Columnas obligatorias: `price`, `brand`, `year`, `title_status`, `mileage`.
Opcionales: `vin` (detecta duplicados), `model`, `state`, `color`, `lot`, `condition` (se conservan en el resultado).
La pestaña de lotes incluye un CSV de ejemplo para descargar.

## Monitoreo y reentrenamiento

| Indicador | Verde (mantener) | Ámbar (monitorear) | Rojo (reentrenar) |
|---|---|---|---|
| PSI de segmentos vs entrenamiento | < 0,10 | 0,10 – 0,25 | > 0,25 |
| % clean fuera del rango de entrenamiento | < 2% | 2% – 5% | > 5% |
| % en revisión de datos | < 3% | 3% – 6% | > 6% (revisar la fuente) |

Se recomienda reentrenar si algún indicador está en rojo, si hay ámbar en dos lotes seguidos o, en cualquier caso,
de forma trimestral con el histórico acumulado.

## Limitaciones

- El dataset no tiene fechas de publicación, márgenes, costos ni ventas: los segmentos describen el inventario
  publicado, no a los compradores, y el impacto económico de las palancas debe validarse con pilotos.
- El modelo no distingue el tipo de vehículo (pickup, sedán, SUV), porque la columna `model` no es confiable.
- La antigüedad se calcula respecto a 2020 (año máximo del dataset); vehículos posteriores se marcan como fuera de rango.
