from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ========= CARGAR MODELO Y DATOS HISTÓRICOS SOLO UNA VEZ =========
MODEL_PATH = "lgb_cantidad_model.pkl"
DATA_PATH = "atl_monthly_ready.h5"

print("Cargando modelo y dataset...")
data_model = joblib.load(MODEL_PATH)
model = data_model["model"]
features = data_model["features"]
poblaciones = data_model["poblaciones"]

# Dataset histórico
df = pd.read_hdf(DATA_PATH, key="data")
df["MUNICIPIO"] = df["MUNICIPIO"].str.upper()
df = df.sort_values(["MUNICIPIO", "YEAR_MONTH"])

# Calcular porcentaje de aumento histórico por municipio
df["AUMENTO_PORC"] = df.groupby("MUNICIPIO")["CANTIDAD"].pct_change() * 100
avg_increase = df.groupby("MUNICIPIO")["AUMENTO_PORC"].mean().to_dict()
min_increase = df.groupby("MUNICIPIO")["AUMENTO_PORC"].min().to_dict()
max_increase = df.groupby("MUNICIPIO")["AUMENTO_PORC"].max().to_dict()

# Rango histórico de cantidades
MIN_CANT = df["CANTIDAD"].min()
MAX_CANT = df["CANTIDAD"].max()

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionInput(BaseModel):
    MUNICIPIO: str
    YEAR_MONTH: str
    CANTIDAD: int


@app.post("/api/predict")  # Cambié de /predict a /api/predict
def predict_next(data: PredictionInput):
    try:
        municipio = data.MUNICIPIO.strip().upper()
        fecha = pd.to_datetime(data.YEAR_MONTH)
        cant_actual = int(data.CANTIDAD)
    except:
        raise HTTPException(
            status_code=400,
            detail="Formato inválido. Se espera MUNICIPIO(str), YEAR_MONTH('YYYY-MM-DD'), CANTIDAD(int).",
        )

    poblacion = poblaciones[municipio]

    # ---------- Predicción según modelo o extrapolación ----------
    if MIN_CANT <= cant_actual <= MAX_CANT:
        X_input = pd.DataFrame([{"CANTIDAD": cant_actual, "POBLACION": poblacion}])
        cant_futura_model = model.predict(X_input, num_iteration=model.best_iteration)[
            0
        ]
        aumento_model = (cant_futura_model - cant_actual) / cant_actual * 100
    else:
        # Extrapolación usando promedio histórico del municipio
        promedio_hist = avg_increase.get(municipio, 0)
        aumento_model = promedio_hist
        cant_futura_model = cant_actual * (1 + promedio_hist / 100)

    # Validar porcentaje de aumento dentro de límites históricos
    min_hist = min_increase.get(municipio, aumento_model)
    max_hist = max_increase.get(municipio, aumento_model)
    porcentaje_aumento = max(min(aumento_model, max_hist), min_hist)

    # Ajustar cantidad futura según porcentaje validado
    cant_futura = cant_actual * (1 + porcentaje_aumento / 100)

    # Calcular tasa por 100k
    tasa_futura = cant_futura / poblacion * 100000

    return {
        "MUNICIPIO": municipio,
        "FECHA_INPUT": str(fecha.date()),
        "CANTIDAD_FUTURA_ESTIMADA": int(cant_futura),
        "PREDICCION_PROXIMO_MES_TASA_100k": round(tasa_futura, 2),
        "PORCENTAJE_AUMENTO": round(porcentaje_aumento, 2),
    }
