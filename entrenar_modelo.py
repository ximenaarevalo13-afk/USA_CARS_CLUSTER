"""
entrenar_modelo.py
Genera `modelo_segmentacion.pkl` a partir de `data/USA_cars_datasets.csv`.

Uso:
    python entrenar_modelo.py
    python entrenar_modelo.py --csv ruta/otro_archivo.csv --salida modelo_segmentacion.pkl

Volver a ejecutarlo cuando:
  * se reentrena con un histórico nuevo (ver criterios de reentrenamiento en el README), o
  * cambia la versión de scikit-learn (un .pkl solo es compatible con la versión que lo creó).
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from segmentador import entrenar

BASE = Path(__file__).parent


def main() -> None:
    p = argparse.ArgumentParser(description="Entrena el modelo de segmentación y lo guarda en .pkl")
    p.add_argument("--csv", default=str(BASE / "data" / "USA_cars_datasets.csv"))
    p.add_argument("--salida", default=str(BASE / "modelo_segmentacion.pkl"))
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    modelo = entrenar(df)
    joblib.dump(modelo, a.salida, compress=3)

    n = modelo["n_entrenamiento"]
    print(f"Modelo guardado en: {a.salida}")
    print(f"  Registros: {n['total']:,} | clean: {n['clean']:,} | salvage: {n['salvage']} | revisión: {n['revision']}")
    print(f"  PCA: {modelo['n_componentes']} componentes ({sum(modelo['varianza_explicada']):.1%} de varianza)")
    print(f"  K-Means K={modelo['k']} (semilla {modelo['seed']}) | silueta = {modelo['silueta']:.3f}")
    for s in modelo["perfil"]:
        print(f"  - {s['segmento']:<22} {s['vehiculos']:>4} vehículos | mediana USD {s['precio_mediano']:>9,.0f}")


if __name__ == "__main__":
    main()
