CREATE OR REPLACE VIEW hnn_training_view AS
SELECT
    run.run_id,
    run.system_id,
    sys.name AS system_name,
    run.lagrange_point_id,
    lp.name AS lagrange_point_name,
    run.scenario_name,

    ts.step,
    ts.t,
    ts.x,
    ts.y,
    ts.z,
    ts.vx,
    ts.vy,
    ts.vz

FROM cr3bp_trajectory_sample AS ts
JOIN cr3bp_simulation_run AS run
    ON run.run_id = ts.run_id
JOIN cr3bp_system AS sys
    ON sys.system_id = run.system_id
JOIN cr3bp_lagrange_point AS lp
    ON lp.lagrange_point_id = run.lagrange_point_id

-- all current simulations are in the rotating frame by construction
-- (frame_default is stored in cr3bp_system if needed later)

ORDER BY run.run_id, ts.step;
