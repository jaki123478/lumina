"""Meteo corrente e previsione breve per la chat Lumina (recupero web, elaborazione locale)."""
import re
import requests

def answer(query: str):
    place = re.sub(r".*?\b(?:a|ad|in|per)\s+", "", query, flags=re.I).strip(" ?.!,") or "Roma"
    session = requests.Session(); session.trust_env = False
    geo = session.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": place, "count": 1, "language": "it", "format": "json"}, timeout=(8, 15)).json()
    row = (geo.get("results") or [None])[0]
    if not row: return "Non trovo la località richiesta.", {"engine":"meteo Open-Meteo", "external_retrieval":True, "sources":[]}
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    data = session.get(forecast_url, params={"latitude":row["latitude"], "longitude":row["longitude"], "current":"temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m", "timezone":"auto"}, timeout=(8, 15)).json()
    cur = data.get("current") or {}; unit = data.get("current_units") or {}
    text = (f"Meteo attuale a {row.get('name', place)}: {cur.get('temperature_2m')} {unit.get('temperature_2m','°C')}, "
            f"umidità {cur.get('relative_humidity_2m')}%, vento {cur.get('wind_speed_10m')} {unit.get('wind_speed_10m','km/h')}. "
            f"Precipitazioni: {cur.get('precipitation', 0)} {unit.get('precipitation','mm')}.")
    return text, {"engine":"meteo Open-Meteo, lettura locale", "external_retrieval":True, "audio_processing":"local", "sources":[forecast_url], "location":row.get("name", place)}
