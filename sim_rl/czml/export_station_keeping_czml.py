"""Export a station-keeping PPO rollout to a CesiumJS CZML file.

This utility reads the final rollout of a trained CR3BP station-keeping model
(e.g. ``final_rollout_after_manual_stop.csv``) and converts it into a CZML file
that can be visualized in CesiumJS.

The script assumes:

- Scenario name: ``earth-moon-L1-3D`` (customizable),
- a project structure compatible with :mod:`sim_rl.training.train_poc`,
  i.e. ``runs/<scenario_name>/run_YYYYMMDD_HHMMSS/rollouts/...``.
- The final rollout CSV contains position and Δv columns (``x0,x1,x2,dv0,dv1,dv2``).
- A GLTF spacecraft model exists in the same directory as this script,
  otherwise a Cesium sample model is used as fallback.

Usage
-----

From project root:

.. code-block:: bash

    python -m sim_rl.visualization.export_station_keeping_czml
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Starting Station-Keeping to CZML export...")

# --------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------

SCENARIO_NAME = "earth-moon-L1-3D"

# Base run directory relative to this script (mirrors train_poc logic)
BASE_RUN_DIR = Path(__file__).parent / "runs"

CZML_FILENAME = "station_keeping_mission.czml"

# Local GLTF model (fallback if missing)
MODEL_FILENAME = "Gateway_Core.glb"

# Number of steps to export from the rollout
MAX_STEPS = 4000

# CR3BP physics constants
MU_EM = 0.0121505856
DT = 0.01  # CR3BP integration timestep
SIDEREAL_MONTH_SEC = 27.321661 * 24 * 3600
SCALE_FACTOR = 384_400_000.0  # normalize CR3BP units to meters

# Cesium clock start time
START_TIME = datetime.datetime(
    2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
)

# --------------------------------------------------------------------
# 2. Resolve GLTF model path / fallback URL
# --------------------------------------------------------------------

FALLBACK_MODEL_URL = (
    "https://raw.githubusercontent.com/CesiumGS/cesium/main/"
    "Apps/SampleData/models/CesiumAir/Cesium_Air.glb"
)

if os.path.exists(MODEL_FILENAME):
    model_uri = MODEL_FILENAME
    print(f"Local GLTF model found: {MODEL_FILENAME}")
else:
    model_uri = FALLBACK_MODEL_URL
    print(f"{MODEL_FILENAME} not found, using placeholder model: {model_uri}")

# --------------------------------------------------------------------
# 3. Locate run directory & rollout CSV
# --------------------------------------------------------------------

scenario_root = BASE_RUN_DIR / SCENARIO_NAME

latest_txt = scenario_root / "latest_run.txt"

if latest_txt.exists():
    run_dir = Path(latest_txt.read_text(encoding="utf-8").strip())
    print(f"Using run from latest_run.txt: {run_dir}")
else:
    candidates = [
        d for d in scenario_root.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    if not candidates:
        raise FileNotFoundError(f"No run_* directories found in {scenario_root}")
    run_dir = sorted(candidates)[-1]
    print(f"Using latest run_* directory: {run_dir}")

csv_path = run_dir / "rollouts" / "final_rollout_after_manual_stop.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"Rollout CSV not found: {csv_path}")

print(f"Loading rollout CSV: {csv_path}")

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Clip to desired step count
if len(df) > MAX_STEPS:
    df = df.iloc[:MAX_STEPS].reset_index(drop=True)
    print(f"Using first {MAX_STEPS} steps from rollout.")
else:
    print(f"Using all {len(df)} steps.")

# --------------------------------------------------------------------
# 4. Extract positions & Δv data
# --------------------------------------------------------------------

required_pos_cols = {"x0", "x1", "x2"}
if not required_pos_cols.issubset(df.columns):
    raise KeyError("CSV must contain columns x0, x1, x2.")

x_rel = df["x0"].to_numpy()
y_rel = df["x1"].to_numpy()
z_rel = df["x2"].to_numpy()

# Δv visualisation strength → path color
if {"dv0", "dv1", "dv2"}.issubset(df.columns):
    dv_vecs = df[["dv0", "dv1", "dv2"]].to_numpy()
    dv_norms = np.linalg.norm(dv_vecs, axis=1)
    max_thrust = np.max(dv_norms) if np.max(dv_norms) > 0 else 1.0
    norm_dv = dv_norms / max_thrust
    print(
        f"Delta-v detected: min={dv_norms.min():.3e}, "
        f"max={dv_norms.max():.3e}"
    )
else:
    norm_dv = np.zeros(len(df))
    print("No dv0/dv1/dv2 columns found, using constant path color.")

# --------------------------------------------------------------------
# 5. Rotating → inertial transform & time construction
# --------------------------------------------------------------------

seconds_per_step = DT * (SIDEREAL_MONTH_SEC / (2 * np.pi))

cmap = plt.get_cmap("coolwarm")
rgba_time_list = []
moon_pos_data = []
probe_pos_data = []

# Earth & Moon in CR3BP rotating frame
pos_earth_rot = np.array([-MU_EM, 0.0, 0.0])
pos_moon_rot = np.array([1.0 - MU_EM, 0.0, 0.0])

for i in range(len(df)):
    step_idx = i
    t_rot = step_idx * DT

    c, s = np.cos(t_rot), np.sin(t_rot)

    current_time = START_TIME + datetime.timedelta(
        seconds=float(step_idx * seconds_per_step)
    )
    iso_time = current_time.isoformat().replace("+00:00", "Z")

    # Δv → path color
    rgba = cmap(norm_dv[i])
    alpha = int(150 + 105 * norm_dv[i])
    rgba_time_list.extend([
        iso_time,
        int(rgba[0] * 255),
        int(rgba[1] * 255),
        int(rgba[2] * 255),
        alpha,
    ])

    # rotation from rotating → inertial
    def rot(vec):
        return np.array([
            vec[0] * c - vec[1] * s,
            vec[0] * s + vec[1] * c,
            vec[2],
        ])

    earth_in = rot(pos_earth_rot)
    moon_in = rot(pos_moon_rot)
    probe_in = rot(np.array([x_rel[i], y_rel[i], z_rel[i]]))

    moon_final = (moon_in - earth_in) * SCALE_FACTOR
    probe_final = (probe_in - earth_in) * SCALE_FACTOR

    moon_pos_data.extend([
        iso_time,
        float(moon_final[0]),
        float(moon_final[1]),
        float(moon_final[2]),
    ])
    probe_pos_data.extend([
        iso_time,
        float(probe_final[0]),
        float(probe_final[1]),
        float(probe_final[2]),
    ])

end_time = START_TIME + datetime.timedelta(
    seconds=float((len(df) - 1) * seconds_per_step)
)

interval_str = (
    f"{START_TIME.isoformat().replace('+00:00','Z')}/"
    f"{end_time.isoformat().replace('+00:00','Z')}"
)

# --------------------------------------------------------------------
# 6. Build CZML document
# --------------------------------------------------------------------

czml = [
    {
        "id": "document",
        "name": "Station Keeping Mission",
        "version": "1.0",
        "clock": {
            "interval": interval_str,
            "currentTime": START_TIME.isoformat().replace("+00:00", "Z"),
            "multiplier": 3600 * 6,  # 6 hours per real second
            "range": "LOOP_STOP",
            "step": "SYSTEM_CLOCK_MULTIPLIER",
        },
    },
    {
        "id": "Moon",
        "name": "Moon",
        "availability": interval_str,
        "position": {
            "epoch": START_TIME.isoformat().replace("+00:00", "Z"),
            "cartesian": moon_pos_data,
        },
        "ellipsoid": {
            "radii": {"cartesian": [1_737_400.0, 1_737_400.0, 1_737_400.0]},
            "material": {
                "image": {
                    "uri": (
                        "https://raw.githubusercontent.com/CesiumGS/cesium/main/"
                        "Apps/Sandcastle/images/moonSmall.jpg"
                    )
                }
            },
        },
        "label": {
            "text": "Moon",
            "font": "12pt monospace",
            "style": "FILL_AND_OUTLINE",
            "pixelOffset": {"cartesian2": [0, 30]},
            "showBackground": True,
        },
    },
    {
        "id": "StationKeepingProbe",
        "name": "Station-Keeping Probe",
        "availability": interval_str,
        "position": {
            "epoch": START_TIME.isoformat().replace("+00:00", "Z"),
            "cartesian": probe_pos_data,
        },
        "model": {
            "gltf": model_uri,
            "scale": 2000.0,
            "minimumPixelSize": 128,
            "show": True,
        },
        "orientation": {
            "velocityReference": "#StationKeepingProbe"
        },
        "path": {
            "material": {
                "polylineOutline": {
                    "color": {
                        "epoch": START_TIME.isoformat().replace("+00:00", "Z"),
                        "rgba": rgba_time_list,
                    },
                    "outlineColor": {"rgba": [255, 255, 255, 255]},
                    "outlineWidth": 1,
                }
            },
            "width": 4,
            "leadTime": 0,
            "trailTime": 10_000_000,
        },
    },
]

# --------------------------------------------------------------------
# 7. Write CZML
# --------------------------------------------------------------------

with open(CZML_FILENAME, "w", encoding="utf-8") as f:
    json.dump(czml, f)

print("------------------------------------------------------------")
print(f"CZML file written: {CZML_FILENAME}")
print("Open with CesiumJS (see index.html).")
print("------------------------------------------------------------")
