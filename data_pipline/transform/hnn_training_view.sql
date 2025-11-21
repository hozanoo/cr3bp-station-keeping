-- HNN training view for the CR3BP project
--
-- This file defines a (materialized) view that prepares simulation
-- data for consumption by the HNN training code.
--
-- Suggested schema for the underlying tables:
--
--   sim.simulation_run(
--       sim_id          BIGSERIAL PRIMARY KEY,
--       scenario_name   TEXT NOT NULL,
--       created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
--       batch_id        TEXT NULL
--   );
--
--   sim.simulation_state(
--       sim_id          BIGINT REFERENCES sim.simulation_run(sim_id),
--       t               DOUBLE PRECISION NOT NULL,
--       x               DOUBLE PRECISION NOT NULL,
--       y               DOUBLE PRECISION NOT NULL,
--       z               DOUBLE PRECISION NOT NULL,
--       vx              DOUBLE PRECISION NOT NULL,
--       vy              DOUBLE PRECISION NOT NULL,
--       vz              DOUBLE PRECISION NOT NULL
--       -- optionally: ax, ay, az, energy, etc.
--   );
--
-- The view below is only a skeleton and should be adapted to the
-- final table design.

DROP MATERIALIZED VIEW IF EXISTS sim.hnn_training_view;

CREATE MATERIALIZED VIEW sim.hnn_training_view AS
SELECT
    s.sim_id,
    s.t,
    s.x,
    s.y,
    s.z,
    s.vx,
    s.vy,
    s.vz
FROM sim.simulation_state AS s
JOIN sim.simulation_run AS r
  ON r.sim_id = s.sim_id
WHERE r.scenario_name = 'earth-moon-L1-3D';

-- You can later extend this with additional features (e.g. mu, scenario
-- identifiers, engineered invariants). Remember to refresh the view
-- after loading new data:
--
--   REFRESH MATERIALIZED VIEW CONCURRENTLY sim.hnn_training_view;
