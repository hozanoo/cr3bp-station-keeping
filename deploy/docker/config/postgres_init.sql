-- ============================================================
-- CR3BP Database Initialization
-- This setup creates:
--   • User: cr3bp_user
--   • Database: cr3bp_db
--   • Schema: cr3bp
-- All tables remain strictly inside the cr3bp schema.
-- ============================================================

---------------------------------------------------------------
-- Create user if not already existing
---------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_catalog.pg_roles WHERE rolname = 'cr3bp_user'
    ) THEN
        CREATE ROLE cr3bp_user LOGIN PASSWORD 'cr3bp_password';
    END IF;
END;
$$;

---------------------------------------------------------------
-- Create database if not already existing
---------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_database WHERE datname = 'cr3bp_db'
    ) THEN
        CREATE DATABASE cr3bp_db OWNER cr3bp_user;
    END IF;
END;
$$;

---------------------------------------------------------------
-- Connect to the target database and set active schema
---------------------------------------------------------------
\connect cr3bp_db;
SET search_path TO cr3bp, public;


-- ===================================================================
-- Table: cr3bp_system
-- Represents a two-body primary system (e.g., Earth–Moon).
-- ===================================================================

CREATE TABLE IF NOT EXISTS cr3bp_system (
    system_id     SERIAL PRIMARY KEY,
    name          TEXT UNIQUE NOT NULL,
    frame_default TEXT NOT NULL CHECK (
        frame_default IN ('rotating', 'inertial', 'barycentric')
    )
);


-- ===================================================================
-- Table: cr3bp_lagrange_point
-- Represents a Lagrange point belonging to a CR3BP system.
-- ===================================================================

CREATE TABLE IF NOT EXISTS cr3bp_lagrange_point (
    lagrange_point_id SERIAL PRIMARY KEY,
    system_id         INT NOT NULL REFERENCES cr3bp_system(system_id),
    name              TEXT NOT NULL,
    UNIQUE (system_id, name)
);


-- ===================================================================
-- Table: cr3bp_simulation_run
--
-- Stores metadata about a single exported trajectory batch.
-- A run corresponds to one CSV trajectory file.
--
-- Fields:
--   run_id                 UUID               Unique identifier.
--   system_id              INT                Foreign key.
--   lagrange_point_id      INT                Foreign key.
--   scenario_name          TEXT               High-level scenario label.
--   dataset_tag            TEXT               Optional experiment tag
--                                             for training phases.
--   created_at             TIMESTAMPTZ        Timestamp of creation.
--   source_file            TEXT               Absolute CSV source path.
--   integrator             TEXT               Numerical integrator used.
--   step_size              DOUBLE PRECISION   Integrator time step.
--   terminated             BOOLEAN            Indicates explicit termination.
--   termination_reason     TEXT               Encodes why the run ended.
--   initial_condition_type TEXT               IC classification label.
-- ===================================================================

CREATE TABLE IF NOT EXISTS cr3bp_simulation_run (
    run_id                 UUID PRIMARY KEY,
    system_id              INT NOT NULL REFERENCES cr3bp_system(system_id),
    lagrange_point_id      INT NOT NULL REFERENCES cr3bp_lagrange_point(lagrange_point_id),

    scenario_name          TEXT NOT NULL,
    dataset_tag            TEXT,                    -- optional training-phase label
    created_at             TIMESTAMPTZ NOT NULL,
    source_file            TEXT NOT NULL,

    integrator             TEXT,
    step_size              DOUBLE PRECISION,
    terminated             BOOLEAN NOT NULL DEFAULT FALSE,
    termination_reason     TEXT,
    initial_condition_type TEXT
);


-- ===================================================================
-- Table: cr3bp_trajectory_sample
--
-- Stores all per-step samples of a trajectory referenced by run_id.
-- Each row corresponds to exactly one timestamp of one simulation.
-- ===================================================================

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
