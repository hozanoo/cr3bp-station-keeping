-- -----------------------------------------------------------
-- PostgreSQL initialization aligned with .env configuration:
--   DB_NAME=cr3bp_db
--   DB_USER=cr3bp_user
--   DB_PASSWORD=cr3bp_password
-- -----------------------------------------------------------

-- Create user
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_catalog.pg_roles WHERE rolname = 'cr3bp_user'
    ) THEN
        CREATE ROLE cr3bp_user LOGIN PASSWORD 'cr3bp_password';
    END IF;
END;
$$;

-- Create database
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_database WHERE datname = 'cr3bp_db'
    ) THEN
        CREATE DATABASE cr3bp_db OWNER cr3bp_user;
    END IF;
END;
$$;

-- Switch to target database
\connect cr3bp_db;

-- ===================================================================
-- Core CR3BP tables (aligned with loader_postgres.ensure_schema)
-- ===================================================================

CREATE TABLE IF NOT EXISTS cr3bp_system (
    system_id     SERIAL PRIMARY KEY,
    name          TEXT UNIQUE NOT NULL,
    frame_default TEXT NOT NULL CHECK (
        frame_default IN ('rotating', 'inertial', 'barycentric')
    )
);

CREATE TABLE IF NOT EXISTS cr3bp_lagrange_point (
    lagrange_point_id SERIAL PRIMARY KEY,
    system_id         INT NOT NULL REFERENCES cr3bp_system(system_id),
    name              TEXT NOT NULL,
    UNIQUE (system_id, name)
);

CREATE TABLE IF NOT EXISTS cr3bp_simulation_run (
    run_id            UUID PRIMARY KEY,
    system_id         INT NOT NULL REFERENCES cr3bp_system(system_id),
    lagrange_point_id INT NOT NULL REFERENCES cr3bp_lagrange_point(lagrange_point_id),
    scenario_name     TEXT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL,
    source_file       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cr3bp_trajectory_sample (
    run_id UUID NOT NULL REFERENCES cr3bp_simulation_run(run_id) ON DELETE CASCADE,
    step   INT NOT NULL,

    t      DOUBLE PRECISION NOT NULL,

    x      DOUBLE PRECISION NOT NULL,
    y      DOUBLE PRECISION NOT NULL,
    z      DOUBLE PRECISION NOT NULL,

    vx     DOUBLE PRECISION NOT NULL,
    vy     DOUBLE PRECISION NOT NULL,
    vz     DOUBLE PRECISION NOT NULL,

    ax     DOUBLE PRECISION NOT NULL,
    ay     DOUBLE PRECISION NOT NULL,
    az     DOUBLE PRECISION NOT NULL,

    PRIMARY KEY (run_id, step)
);
