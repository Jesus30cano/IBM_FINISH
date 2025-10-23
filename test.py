import requests

url = "http://127.0.0.1:8000/predict"

# Valores de prueba para Soledad
cantidades = [2000, 2500, 3000, 4000, 5000]

for cant in cantidades:
    data = {"MUNICIPIO": "SOLEDAD", "YEAR_MONTH": "2024-08-01", "CANTIDAD": cant}
    response = requests.post(url, json=data)

    if response.status_code == 200:
        res = response.json()
        print(
            f"CANTIDAD_ACTUAL: {cant} | CANTIDAD_FUTURA_ESTIMADA: {res['CANTIDAD_FUTURA_ESTIMADA']} | TASA_100k: {res['PREDICCION_PROXIMO_MES_TASA_100k']:.2f}"
        )
    else:
        print(f"Error {response.status_code}: {response.text}")
