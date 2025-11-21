-- data_pipline/transform/hnn_training_view.sql
--
-- Normalized view for CR3BP trajectories used as HNN training data.
--
-- This view joins the system, Lagrange-point and simulation metadata
-- with the per-step trajectory samples.
--
-- Columns:
--   system_id, system_name
--   lagrange_point_id, lagrange_point_name
--   run_id, scenario_name
--   step, t
--   x, y, z, vx, vy, vz
--
-- Only rotating-frame simulations are exposed, which matches the
-- current PPO / CR3BP setup.

CREATE OR REPLACE VIEW hnn_training_view AS
SELECT
    s.id   AS system_id,
    s.name AS system_name,

    lp.id   AS lagrange_point_id,
    lp.name AS lagrange_point_name,

    r.id            AS run_id,
    r.scenario_name AS scenario_name,

    ts.step,
    ts.t,
    ts.x,
    ts.y,
    ts.z,
    ts.vx,
    ts.vy,
    ts.vz
FROM cr3bp_trajectory_sample AS ts
JOIN cr3bp_simulation_run   AS r  ON r.id = ts.run_id
JOIN cr3bp_system           AS s  ON s.id = r.system_id
JOIN cr3bp_lagrange_point   AS lp ON lp.id = r.lagrange_point_id
WHERE s.frame_default = 'rotating';

-- Optional: helpful indexes for faster access from HNN training jobs.
-- These can be created once in the database (not required, but recommended):
--
--   CREATE INDEX IF NOT EXISTS idx_hnn_samples_run_step
--       ON cr3bp_trajectory_sample (run_id, step);
--
--   CREATE INDEX IF NOT EXISTS idx_hnn_runs_system_lpoint
--       ON cr3bp_simulation_run (system_id, lagrange_point_id);
