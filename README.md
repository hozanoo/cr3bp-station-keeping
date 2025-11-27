The project expects a GLTF/GLB spacecraft model at:

``sim_rl/czml/Gateway_Core.glb``

Due to file size it is not included in the Git repository.  
A reference model can be downloaded from:

https://science.nasa.gov/3d-resources/gateway-lunar-space-station/


# CR3BP Station-Keeping Project (Earth–Moon L1)

This repository implements a complete simulation, data-engineering, and machine-learning pipeline for spacecraft station-keeping in the Circular Restricted Three-Body Problem (CR3BP).

**Key Technologies:**
- **Physics:** High-fidelity CR3BP integrator (DOP853/RK45) in Rotating Frame.
- **RL:** Gymnasium-compatible environment for Station-Keeping.
- **Data Engineering:** Airflow & PostgreSQL pipeline for batch simulation data.
- **Deep Learning:** Hamiltonian Neural Networks (HNN) for learning orbital dynamics (Physics-Informed).
- **Visualization:** CesiumJS with CZML export.

> **⚠️ Current Status:** The **HNN (Hamiltonian Neural Network)** module (`hnn_models/`) is currently under active development and refactoring. The Simulation and Data Pipeline are stable.

---

## 1. Project Structure

The project is divided into three main domains:

```text
cr3bp_project_3d/
│
├── sim_rl/                  # [Domain A] Simulation & Reinforcement Learning
│   ├── cr3bp/               # Core Physics (N-Body, CR3BP Equations, Gym Env)
│   └── czml/                # Visualization logic (Cesium/CZML exports)
│
├── data_pipeline/           # [Domain B] ETL Pipeline & Airflow
│   ├── extract/             # Scripts to generate batch simulations (CSV)
│   ├── load/                # Loader scripts for PostgreSQL
│   └── dags/                # Airflow DAGs (Scheduled Pipelines)
│
├── hnn_models/              # [Domain C] Hamiltonian Neural Networks (WIP)
│   ├── dataloader/          # PyTorch Datasets
│   ├── model/               # HNN Architecture (hnn.py)
│   └── training/            # Training loops
│
├── deploy/                  # Infrastructure (Docker)
│   ├── docker-compose.yml   # Orchestration of DB, Airflow, Cesium
│   └── docker/              # Dockerfiles and Configs
│
└── notebooks/               # Experimental Notebooks & Analysis
```

---

## 2. Setup & Installation

### 2.1 Prerequisites
* **Docker & Docker Compose** (v2.0+)
* **Python 3.10+** (for local development)

### 2.2 Local Python Environment
To run the simulation or training scripts locally (outside Docker), create a virtual environment:

```bash
# Using Conda
conda create -n cr3bp_env python=3.10 -y
conda activate cr3bp_env

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Running the Infrastructure (Docker)

We use Docker to orchestrate the Database, Airflow, and the Visualization Server.

1.  **Navigate to the deploy folder:**
    ```bash
    cd deploy
    ```

2.  **Start the stack:**
    ```bash
    docker compose up -d --build
    ```

**Services Overview:**

| Service | Address | Description |
| :--- | :--- | :--- |
| **Airflow** | `http://localhost:8080` | Pipeline orchestration. Default login defined in `.env`. |
| **PgAdmin** | `http://localhost:5050` | Database GUI. |
| **Cesium** | `http://localhost:8000` | 3D Visualization Server. |
| **Postgres** | `localhost:5432` | Main Data Warehouse (`cr3bp_db`). |

---

## 4. Usage Guide

### A. Generating Data (The Pipeline)
You can generate simulation data either manually or via Airflow.

**Option 1: Via Airflow (Recommended)**
1.  Go to `http://localhost:8080`.
2.  Trigger the DAG **`cr3bp_daily_pipeline`**.
3.  This will:
    * Run batch simulations (`extract`).
    * Save CSVs to `data_pipeline/raw_exports/`.
    * Load data into Postgres (`load`).

**Option 2: Manual Generation (Local)**
```bash
python data_pipeline/extract/cr3bp_batch_exporter.py
```

### B. Reinforcement Learning (Station Keeping)
The project provides a custom Gym Environment for RL agents.

```python
import gymnasium as gym
from sim_rl.cr3bp.scenarios import ScenarioConfig
from sim_rl.cr3bp.env_cr3bp_station_keeping import Cr3bpStationKeepingEnv

# Configure the scenario
config = ScenarioConfig(system="earth_moon", lagrange_point="L1", dim=3)

# Initialize Environment
env = Cr3bpStationKeepingEnv(scenario=config)

obs, info = env.reset()
# action = env.action_space.sample()  # Replace with RL Agent action
# obs, reward, terminated, truncated, info = env.step(action)
```

### C. Hamiltonian Neural Networks (WIP)
*Note: This module is currently being refactored.*

The HNN model attempts to learn the Hamiltonian H(q,p) directly from the trajectory data.

To run the training loop:
```bash
cd hnn_models/training
python train_hnn.py
```

*Make sure the database is populated via the Airflow pipeline before training.*

### D. Visualization (Cesium)
1.  Ensure the Docker stack is running (`deploy/docker-compose.yml`).
2.  Generate a CZML file from a rollout:
    ```bash
    python sim_rl/czml/export_station_keeping_czml.py
    ```
    *This will create `station_keeping_mission.czml`.*

3.  Open your browser at `http://localhost:8000`.
4.  Cesium will load the generated `.czml` file and the 3D model.

**Note on 3D Models:**
The script looks for `Gateway_Core.glb` in `sim_rl/czml/`. If not found, it falls back to a default Cesium model.

---

## 5. Database Schema
The Postgres database (`cr3bp_db`) is initialized via `deploy/docker/config/postgres_init.sql`.

* `cr3bp_simulation_run`: Metadata for each batch/episode.
* `cr3bp_trajectory_sample`: Time-series data (Position/Velocity).
* **`hnn_training_view`**: A joined view optimized for ML training, aligning states (q, p) with calculated accelerations (dp/dt).

---

## 6. License
Internal Team Project.