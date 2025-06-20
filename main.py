import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('weather_data.csv')

# Convertir fecha a formato numérico
df['Date_Time'] = pd.to_datetime(df['Date_Time'])
df['Year'] = df['Date_Time'].dt.year
df['Month'] = df['Date_Time'].dt.month
df['Day'] = df['Date_Time'].dt.day

# Guardar el mapeo de localidades a códigos
df['Location'] = df['Location'].astype('category')
location_map = dict(enumerate(df['Location'].cat.categories))
location_map_inv = {v: k for k, v in location_map.items()}

# Convertir Location a códigos numéricos
df['Location'] = df['Location'].cat.codes

X = df[['Location', 'Year', 'Month', 'Day']]
y = df[['Temperature_C', 'Humidity_pct', 'Precipitation_mm', 'Wind_Speed_kmh']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
print("Error cuadrático medio:", round(mean_squared_error(y_test, y_pred), 2))

def predecir_clima(localidad, fecha):
    fecha = pd.to_datetime(fecha)
    if localidad not in location_map_inv:
        raise ValueError(f"La localidad '{localidad}' no está en el dataset.")
    localidad_cod = location_map_inv[localidad]
    entrada = [[localidad_cod, fecha.year, fecha.month, fecha.day]]
    pred = modelo.predict(entrada)[0]
    return {
        'Temperature_C': round(float(pred[0]), 2),
        'Humidity_pct': round(float(pred[1]), 2),
        'Precipitation_mm': round(float(pred[2]), 2),
        'Wind_Speed_kmh': round(float(pred[3]), 2)
    }

# Ejemplo de uso (elige una localidad que sí esté en tu CSV)
resultado = predecir_clima('New York', '2025-06-15')
print("Predicción del clima:")
print(f"Temperatura (°C): {resultado['Temperature_C']}")
print(f"Humedad (%): {resultado['Humidity_pct']}")
print(f"Precipitación (mm): {resultado['Precipitation_mm']}")
print(f"Viento (km/h): {resultado['Wind_Speed_kmh']}")