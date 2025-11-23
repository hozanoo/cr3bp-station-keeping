CREATE TABLE IF NOT EXISTS cr3bp_system (
    system_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    mu REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS cr3bp_frame (
    frame_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS cr3bp_episode (
    episode_id BIGSERIAL PRIMARY KEY,
    system_id INTEGER REFERENCES cr3bp_system(system_id),
    frame_id INTEGER REFERENCES cr3bp_frame(frame_id),
    dim INTEGER NOT NULL,
    start_state REAL[] NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cr3bp_step (
    step_id BIGSERIAL PRIMARY KEY,
    episode_id BIGINT REFERENCES cr3bp_episode(episode_id),
    t REAL NOT NULL,
    state REAL[] NOT NULL,
    dv REAL[] NOT NULL
);

CREATE TABLE IF NOT EXISTS cr3bp_trajectory_sample (
    sample_id BIGSERIAL PRIMARY KEY,
    episode_id BIGINT REFERENCES cr3bp_episode(episode_id),
    t REAL NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    z REAL NOT NULL,
    dx REAL NOT NULL,
    dy REAL NOT NULL,
    dz REAL NOT NULL
);
