CREATE OR REPLACE VIEW hnn_training_view AS
SELECT
    ep.episode_id,
    ep.system_id,
    s.name AS system_name,
    ep.frame_id,
    f.name AS frame_name,
    ep.dim,

    ts.t,
    ts.x,
    ts.y,
    ts.z,
    ts.dx AS vx,
    ts.dy AS vy,
    ts.dz AS vz

FROM cr3bp_trajectory_sample AS ts
JOIN cr3bp_episode AS ep ON ep.episode_id = ts.episode_id
JOIN cr3bp_system  AS s  ON s.system_id = ep.system_id
JOIN cr3bp_frame   AS f  ON f.frame_id = ep.frame_id

-- nur rotating-frame Simulationen verwenden
WHERE f.name = 'rotating'

ORDER BY ep.episode_id, ts.t;
