# sim_rl/czml/export_station_keeping_czml.py

"""
Export station-keeping PPO rollouts to a CesiumJS CZML file.

This utility reads one or more rollouts of a trained CR3BP
station-keeping model and converts them into a CZML file that can be
visualized in CesiumJS.

Assumptions
-----------

- Scenario name: ``earth-moon-L1-3D`` (default).
- Run directory structure compatible with the robust training script, i.e.
  ``sim_rl/training/runs_robust/<scenario_name>/run_YYYYMMDD_HHMMSS/rollouts/...``.
- Rollout CSVs contain absolute position in the rotating frame
  (``pos_rot_abs_0, pos_rot_abs_1, pos_rot_abs_2``) and delta-v columns
  (``dv_0, dv_1, dv_2``).
- A GLTF spacecraft model is expected at
  ``sim_rl/czml/Gateway_Core.glb``; if missing, a Cesium sample
  model is used as fallback.
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------------

SCENARIO_NAME = "earth-moon-L1-3D"

HERE = Path(__file__).resolve().parent

BASE_RUN_DIR = HERE.parents[1] / "training" / "runs_robust"

RUN_DIR = Path(
    "sim_rl/training/runs_robust/earth-moon-L1-3D/run_20251209_114916"
)

# Main (late, "perfect") rollout and secondary (early) rollout
PRIMARY_ROLLOUT = RUN_DIR / "rollouts" / "sim_2900_steps_5939200.csv"
SECONDARY_ROLLOUT = RUN_DIR / "rollouts" / "sim_0008_steps_16384.csv"

# Uncontrolled free-drift rollout (already exported from Jupyter)
FREE_ROLLOUT = HERE / "free_drift_uncontrolled.csv"

CZML_FILENAME = HERE / "station_keeping_mission.czml"
MODEL_FILENAME = HERE / "Gateway_Core.glb"

MAX_STEPS = 4000

MU_EM = 0.0121505856
DT = 0.01
SIDEREAL_MONTH_SEC = 27.321661 * 24 * 3600
SCALE_FACTOR = 384_400_000.0

START_TIME = datetime.datetime(
    2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
)

# --------------------------------------------------------------------
# 2. Model URI resolution
# --------------------------------------------------------------------

FALLBACK_MODEL_URL = (
    "https://raw.githubusercontent.com/CesiumGS/cesium/main/"
    "Apps/SampleData/models/CesiumAir/Cesium_Air.glb"
)

if MODEL_FILENAME.exists():
    model_uri = MODEL_FILENAME.name
    print(f"[INFO] Local GLTF model found: {MODEL_FILENAME}")
else:
    model_uri = FALLBACK_MODEL_URL
    print(
        f"[WARN] {MODEL_FILENAME} not found. "
        "Using Cesium sample model as fallback."
    )

# --------------------------------------------------------------------
# 3. Helper functions
# --------------------------------------------------------------------


def load_and_crop_rollout(csv_path: Path) -> pd.DataFrame:
    """
    Load a rollout CSV and crop it to MAX_STEPS if necessary.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Rollout CSV not found: {csv_path}")

    print(f"[INFO] Loading rollout CSV: {csv_path}")
    df_local = pd.read_csv(csv_path)
    df_local.columns = df_local.columns.str.strip()

    if len(df_local) > MAX_STEPS:
        df_local = df_local.iloc[:MAX_STEPS].reset_index(drop=True)
        print(f"[INFO] Using first {MAX_STEPS} steps from rollout.")
    else:
        print(f"[INFO] Using all {len(df_local)} steps from rollout (N={len(df_local)}).")

    return df_local


def extract_probe_data(
    df_local: pd.DataFrame,
    seconds_per_step: float,
    include_moon: bool = False,
):
    """
    Convert a rollout DataFrame in rotating CR3BP coordinates into
    inertial positions and time-colored path data for Cesium.

    If include_moon is True, Moon positions are generated as well.
    """
    required_pos_cols = {"pos_rot_abs_0", "pos_rot_abs_1", "pos_rot_abs_2"}
    if not required_pos_cols.issubset(df_local.columns):
        missing = required_pos_cols.difference(df_local.columns)
        raise KeyError(
            f"CSV must contain columns {sorted(required_pos_cols)}, "
            f"missing: {sorted(missing)}"
        )

    x_rel = df_local["pos_rot_abs_0"].to_numpy()
    y_rel = df_local["pos_rot_abs_1"].to_numpy()
    z_rel = df_local["pos_rot_abs_2"].to_numpy()

    if {"dv_0", "dv_1", "dv_2"}.issubset(df_local.columns):
        dv_vecs = df_local[["dv_0", "dv_1", "dv_2"]].to_numpy()
        dv_norms = np.linalg.norm(dv_vecs, axis=1)
        max_thrust = np.max(dv_norms) if np.max(dv_norms) > 0 else 1.0
        norm_dv = dv_norms / max_thrust
        print(
            f"[INFO] Delta-v statistics: "
            f"min={dv_norms.min():.3e}, max={dv_norms.max():.3e}"
        )
    else:
        norm_dv = np.zeros(len(df_local))
        print("[INFO] No dv_0/dv_1/dv_2 columns found. Using constant path color.")

    cmap = plt.get_cmap("coolwarm")
    rgba_time_list = []
    probe_pos_data = []
    moon_pos_data = []

    pos_earth_rot = np.array([-MU_EM, 0.0, 0.0])
    pos_moon_rot = np.array([1.0 - MU_EM, 0.0, 0.0])

    for i in range(len(df_local)):
        t_rot = i * DT
        c, s = np.cos(t_rot), np.sin(t_rot)

        current_time = START_TIME + datetime.timedelta(
            seconds=float(i * seconds_per_step)
        )
        iso_time = current_time.isoformat().replace("+00:00", "Z")

        rgba = cmap(norm_dv[i])
        alpha = int(150 + 105 * norm_dv[i])
        rgba_time_list.extend(
            [
                iso_time,
                int(rgba[0] * 255),
                int(rgba[1] * 255),
                int(rgba[2] * 255),
                alpha,
            ]
        )

        def rot(vec: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    vec[0] * c - vec[1] * s,
                    vec[0] * s + vec[1] * c,
                    vec[2],
                ]
            )

        earth_in = rot(pos_earth_rot)
        moon_in = rot(pos_moon_rot)
        probe_in = rot(np.array([x_rel[i], y_rel[i], z_rel[i]]))

        moon_final = (moon_in - earth_in) * SCALE_FACTOR
        probe_final = (probe_in - earth_in) * SCALE_FACTOR

        if include_moon:
            moon_pos_data.extend([iso_time, *map(float, moon_final)])
        probe_pos_data.extend([iso_time, *map(float, probe_final)])

    end_time = START_TIME + datetime.timedelta(
        seconds=float((len(df_local) - 1) * seconds_per_step)
    )

    return probe_pos_data, rgba_time_list, moon_pos_data, end_time


# --------------------------------------------------------------------
# 4. Build CZML content for primary, secondary and free-drift runs
# --------------------------------------------------------------------

if not RUN_DIR.exists():
    raise FileNotFoundError(f"Configured RUN_DIR does not exist: {RUN_DIR}")

print(f"[INFO] Using fixed robust run directory: {RUN_DIR}")

seconds_per_step = DT * (SIDEREAL_MONTH_SEC / (2 * np.pi))

# Primary (robust) run
df_primary = load_and_crop_rollout(PRIMARY_ROLLOUT)
probe_pos_primary, rgba_primary, moon_pos_data, end_time_primary = extract_probe_data(
    df_primary,
    seconds_per_step,
    include_moon=True,
)

# Secondary (early) run
probe_pos_secondary = []
rgba_secondary = []
end_time_secondary = START_TIME

if SECONDARY_ROLLOUT.exists():
    df_secondary = load_and_crop_rollout(SECONDARY_ROLLOUT)
    (
        probe_pos_secondary,
        rgba_secondary,
        _,
        end_time_secondary,
    ) = extract_probe_data(
        df_secondary,
        seconds_per_step,
        include_moon=False,
    )
else:
    print(f"[WARN] Secondary rollout not found: {SECONDARY_ROLLOUT}")

# Free-drift (uncontrolled) run
probe_pos_free = []
rgba_free = []
end_time_free = START_TIME

if FREE_ROLLOUT.exists():
    df_free = load_and_crop_rollout(FREE_ROLLOUT)
    (
        probe_pos_free,
        rgba_free,
        _,
        end_time_free,
    ) = extract_probe_data(
        df_free,
        seconds_per_step,
        include_moon=False,
    )
else:
    print(f"[WARN] Free-drift rollout not found: {FREE_ROLLOUT}")

# Global interval (max over all available runs)
end_time = max(end_time_primary, end_time_secondary, end_time_free)

interval_str = (
    f"{START_TIME.isoformat().replace('+00:00','Z')}/"
    f"{end_time.isoformat().replace('+00:00','Z')}"
)

# --------------------------------------------------------------------
# 5. Build CZML
# --------------------------------------------------------------------

czml = [
    {
        "id": "document",
        "name": "Station Keeping Mission",
        "version": "1.0",
        "clock": {
            "interval": interval_str,
            "currentTime": START_TIME.isoformat().replace("+00:00", "Z"),
            "multiplier": 3600 * 6,
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
            "radii": {"cartesian": [1_737_400, 1_737_400, 1_737_400]},
            "material": {
                "image": {
                    "uri": (
                        "https://raw.githubusercontent.com/CesiumGS/cesium/main/"
                        "Apps/Sandcastle/images/moonSmall.jpg"
                    )
                }
            },
        },
    },
    {
        "id": "StationKeepingProbe",
        "name": "Station-Keeping Probe (sim_2900)",
        "availability": interval_str,
        "position": {
            "epoch": START_TIME.isoformat().replace("+00:00", "Z"),
            "cartesian": probe_pos_primary,
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
                        "rgba": rgba_primary,
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

if probe_pos_secondary:
    czml.append(
        {
            "id": "StationKeepingProbe_Run0008",
            "name": "Station-Keeping Probe (sim_0008)",
            "availability": interval_str,
            "position": {
                "epoch": START_TIME.isoformat().replace("+00:00", "Z"),
                "cartesian": probe_pos_secondary,
            },
            "model": {
                "gltf": model_uri,
                "scale": 2000.0,
                "minimumPixelSize": 128,
                "show": True,
            },
            "orientation": {
                "velocityReference": "#StationKeepingProbe_Run0008"
            },
            "path": {
                "material": {
                    "polylineOutline": {
                        "color": {
                            "epoch": START_TIME.isoformat().replace("+00:00", "Z"),
                            "rgba": rgba_secondary,
                        },
                        "outlineColor": {"rgba": [255, 255, 255, 255]},
                        "outlineWidth": 1,
                    }
                },
                "width": 4,
                "leadTime": 0,
                "trailTime": 10_000_000,
            },
        }
    )

if probe_pos_free:
    czml.append(
        {
            "id": "StationKeepingProbe_FreeDrift",
            "name": "Station-Keeping Probe (free drift)",
            "availability": interval_str,
            "position": {
                "epoch": START_TIME.isoformat().replace("+00:00", "Z"),
                "cartesian": probe_pos_free,
            },
            "model": {
                "gltf": model_uri,
                "scale": 2000.0,
                "minimumPixelSize": 128,
                "show": True,
            },
            "orientation": {
                "velocityReference": "#StationKeepingProbe_FreeDrift"
            },
            "path": {
                "material": {
                    "polylineOutline": {
                        "color": {
                            "epoch": START_TIME.isoformat().replace("+00:00", "Z"),
                            "rgba": rgba_free,
                        },
                        "outlineColor": {"rgba": [255, 255, 255, 255]},
                        "outlineWidth": 1,
                    }
                },
                "width": 4,
                "leadTime": 0,
                "trailTime": 10_000_000,
            },
        }
    )

# --------------------------------------------------------------------
# 6. Write CZML
# --------------------------------------------------------------------

with CZML_FILENAME.open("w", encoding="utf-8") as f:
    json.dump(czml, f)

print("------------------------------------------------------------")
print(f"[INFO] CZML file written: {CZML_FILENAME}")
print("Open this file with CesiumJS (see index.html).")
print("------------------------------------------------------------")
