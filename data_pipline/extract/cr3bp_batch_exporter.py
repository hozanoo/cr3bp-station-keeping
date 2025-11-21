"""
Batch export of CR3BP simulations for HNN training.

This module is responsible for generating batches of CR3BP trajectories
(e.g. in the rotating frame around Earth–Moon L1 in 3D) and writing them
to disk as intermediate files (CSV or Parquet) in ``data/raw``.

The actual numerical integration should be delegated to the core
simulation code in :mod:`sim_rl.cr3bp`, for example by calling a small
helper that wraps your NBodySystem + Simulator with the DOP853 solver.

The typical flow is:

1. Configure number of simulations, number of steps, time step ``dt``,
   scenario name etc.
2. For each simulation:
   - sample a random initial condition around the scenario (e.g. L1 halo),
   - run the CR3BP integrator with DOP853,
   - write the trajectory to a file in ``data/raw``.
3. Return a list of file paths for downstream loading.

At this stage, the functions below are only skeletons. They define
interfaces and responsibilities but do not yet implement the full logic.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class CR3BPSimulationConfig:
    """
    Configuration for a batch of CR3BP simulations.

    Parameters
    ----------
    scenario_name:
        Name of the CR3BP scenario, e.g. ``"earth-moon-L1-3D"``.
    n_simulations:
        Number of independent simulations to generate in one batch.
    n_steps:
        Number of integration steps per simulation.
    dt:
        Time step of the integrator in normalized CR3BP units.
    output_dir:
        Directory where raw output files will be written (e.g. ``data/raw``).
    """

    scenario_name: str = "earth-moon-L1-3D"
    n_simulations: int = 20
    n_steps: int = 4000
    dt: float = 0.01
    output_dir: Path = Path("data") / "raw"


def generate_cr3bp_batch(config: CR3BPSimulationConfig) -> List[Path]:
    """
    Generate a batch of CR3BP simulations and write them to disk.

    This function is intended to be called from an Airflow task.
    It should:

    - ensure that ``config.output_dir`` exists,
    - run ``config.n_simulations`` CR3BP trajectories with the chosen
      scenario and integrator (e.g. DOP853),
    - save each trajectory as a separate file (CSV or Parquet),
    - return the list of created file paths.

    Parameters
    ----------
    config:
        Configuration describing how many simulations to run and how
        they should be parameterised.

    Returns
    -------
    list of pathlib.Path
        Paths to the generated raw data files.

    Notes
    -----
    The concrete file format (CSV vs. Parquet) and column schema are
    deliberately left open here. A good default is:

    - columns: ``t, x, y, z, vx, vy, vz``
    - one file per simulation.

    The loader module in :mod:`data_pipeline.load.loader_postgres`
    will assume a consistent schema across all files.
    """
    # TODO: Implement integration loop using sim_rl.cr3bp utilities.
    # Example sketch (non-functional placeholder):
    #
    # from sim_rl.cr3bp.env_cr3bp_station_keeping import Cr3bpStationKeepingEnv
    # from sim_rl.cr3bp.scenarios import SCENARIOS
    #
    # ...
    #
    raise NotImplementedError("CR3BP batch export is not implemented yet.")


def discover_raw_files(output_dir: Path) -> Iterable[Path]:
    """
    Discover raw simulation files in the given directory.

    This helper is useful for Airflow tasks that need to find all
    newly-generated files before loading them into the database.

    Parameters
    ----------
    output_dir:
        Directory where raw simulation files are stored.

    Returns
    -------
    iterable of pathlib.Path
        All files that match the chosen raw-data pattern.
    """
    # TODO: Decide on filename pattern (e.g. *.parquet or *.csv)
    # and implement discovery logic accordingly.
    raise NotImplementedError("Raw file discovery is not implemented yet.")
