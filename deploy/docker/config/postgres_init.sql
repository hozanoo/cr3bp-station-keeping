-- -------------------------------------------------------------------------
-- CR3BP schema initialization
--
-- This script defines the core tables used by the CR3BP project:
--   - cr3bp_system
--   - cr3bp_lagrange_point
--   - cr3bp_simulation_run
--   - cr3bp_trajectory_sample
--
-- The definitions are aligned with data_pipeline.load.loader_postgres.ensure_schema.
-- -------------------------------------------------------------------------

-- =========================
-- System and configuration
-- =========================

CREATE TABLE IF NOT EXISTS cr3bp_system (
    system_id     SERIAL PRIMARY KEY,
    name          TEXT UNIQUE NOT NULL,
    frame_default TEXT NOT NULL CHECK (
        frame_default IN ('rotating', 'inertial', 'barycentric')
    )
);

CREATE TABLE IF NOT EXISTS cr3bp_lagrange_point (
    lagrange_point_id SERIAL PRIMARY KEY,
    system_id         INT  NOT NULL REFERENCES cr3bp_system(system_id),
    name              TEXT NOT NULL,
    UNIQUE (system_id, name)
);

-- =========================
-- Simulation runs
-- =========================

CREATE TABLE IF NOT EXISTS cr3bp_simulation_run (
    run_id           UUID PRIMARY KEY,
    system_id        INT  NOT NULL REFERENCES cr3bp_system(system_id),
    lagrange_point_id INT NOT NULL REFERENCES cr3bp_lagrange_point(lagrange_point_id),
    scenario_name    TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL,
    source_file      TEXT NOT NULL
);

-- =========================
-- Trajectory samples
-- =========================

CREATE TABLE IF NOT EXISTS cr3bp_trajectory_sample (
    run_id UUID NOT NULL REFERENCES cr3bp_simulation_run(run_id) ON DELETE CASCADE,
    step   INT  NOT NULL,

    t  DOUBLE PRECISION NOT NULL,

    x  DOUBLE PRECISION NOT NULL,
    y  DOUBLE PRECISION NOT NULL,
    z  DOUBLE PRECISION NOT NULL,

    vx DOUBLE PRECISION NOT NULL,
    vy DOUBLE PRECISION NOT NULL,
    vz DOUBLE PRECISION NOT NULL,

    ax DOUBLE PRECISION NOT NULL,
    ay DOUBLE PRECISION NOT NULL,
    az DOUBLE PRECISION NOT NULL,

    PRIMARY KEY (run_id, step)
);
