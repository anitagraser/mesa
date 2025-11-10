#!/usr/bin/env python3
import json, math, os
import numpy as np
import pandas as pd

CSV_IN  = "synthetic_vessel_tracks_with_anomalies_20250517.csv"
CSV_OUT = "data_with_speed_knots.csv"

# --- Cherche la config pour récupérer la bbox et les dimensions du repère ---
def load_bounds_and_dims():
    # chemins possibles
    for p in ("config/config.json", "config.json"):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            gb = cfg.get("geospatial_bounds", {})
            if all(k in gb for k in ("min_lat","max_lat","min_lon","max_lon")):
                width  = float(cfg.get("width", 444.0))
                height = float(cfg.get("height", 200.0))
                return (
                    float(gb["min_lon"]), float(gb["max_lon"]),
                    float(gb["min_lat"]), float(gb["max_lat"]),
                    width, height
                )
    # fallback : tes constantes connues pour la rade (Brest)
    LON_MIN, LON_MAX = -4.6, -4.2666666667
    LAT_MIN, LAT_MAX = 48.2666666667, 48.4166666667
    X_MAX,  Y_MAX    = 444.4444444, 200.0
    return LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, X_MAX, Y_MAX

# --- Haversine (mètres) ---
R_EARTH_M = 6_371_000.0
DEG2RAD = math.pi / 180.0
def haversine_m(lat1, lon1, lat2, lon2):
    lat1r = np.radians(lat1); lat2r = np.radians(lat2)
    dlat  = (lat2 - lat1) * DEG2RAD
    dlon  = (lon2 - lon1) * DEG2RAD
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R_EARTH_M * c

def main():
    df = pd.read_csv(CSV_IN)
    # Colonne d'identifiant (AgentID par défaut ; sinon MMSI ; sinon pas d'id)
    ID_COL = "AgentID" if "AgentID" in df.columns else ("MMSI" if "MMSI" in df.columns else None)

    # Assure la présence de x/y et t
    for col in ("x","y","t"):
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans {CSV_IN}: {col}")

    # 1) t -> datetime et tri
    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    by = [ID_COL, "t"] if ID_COL else ["t"]
    df = df.sort_values(by)

    # 2) Récupère bbox + dimensions du repère
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, X_MAX, Y_MAX = load_bounds_and_dims()

    # 3) Reconstruit lon/lat depuis x/y
    df["lon"] = LON_MIN + (df["x"] / float(X_MAX)) * (LON_MAX - LON_MIN)
    df["lat"] = LAT_MIN + (df["y"] / float(Y_MAX)) * (LAT_MAX - LAT_MIN)

    # 4) Δt, points précédents par trajectoire
    if ID_COL:
        grp = df.groupby(ID_COL, group_keys=False)
        df["lon_prev"] = grp["lon"].shift(1)
        df["lat_prev"] = grp["lat"].shift(1)
        df["t_prev"]   = grp["t"].shift(1)
    else:
        df["lon_prev"] = df["lon"].shift(1)
        df["lat_prev"] = df["lat"].shift(1)
        df["t_prev"]   = df["t"].shift(1)

    dt = (df["t"] - df["t_prev"]).dt.total_seconds()

    # 5) Distance géodésique (m) entre points consécutifs valides
    mask = df[["lon_prev","lat_prev","lon","lat"]].notna().all(axis=1) & dt.notna() & (dt > 0)
    dist_m = np.full(len(df), np.nan, dtype=float)
    dist_m[mask] = haversine_m(
        df.loc[mask, "lat_prev"].to_numpy(),
        df.loc[mask, "lon_prev"].to_numpy(),
        df.loc[mask, "lat"].to_numpy(),
        df.loc[mask, "lon"].to_numpy(),
    )

    # 6) Vitesses
    df["speed_mps"]   = dist_m / dt
    df["speed_knots"] = df["speed_mps"] * 1.943844  # 1 m/s = 1.943844 kn

    # 7) Sauvegarde
    df.to_csv(CSV_OUT, index=False)

    # Petit récap
    n_speed = int(df["speed_knots"].notna().sum())
    vmax    = float(np.nanmax(df["speed_knots"])) if n_speed else float("nan")
    print(f"OK -> {CSV_OUT}  |  lignes avec vitesse: {n_speed}  |  Vmax: {vmax:.2f} kn")

if __name__ == "__main__":
    main()

