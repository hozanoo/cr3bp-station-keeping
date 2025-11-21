-- Initial PostgreSQL setup for the CR3BP project.

CREATE DATABASE cr3bp_db;

-- Basic roles (adapt usernames/passwords to your environment)
CREATE USER cr3bp_user WITH PASSWORD 'cr3bp_password';
GRANT ALL PRIVILEGES ON DATABASE cr3bp_db TO cr3bp_user;

-- Connect to the new database to create schemas.
\connect cr3bp_db;

CREATE SCHEMA IF NOT EXISTS sim;
CREATE SCHEMA IF NOT EXISTS rl;

-- Optional: create extensions you might need
-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- CREATE EXTENSION IF NOT EXISTS timescaledb;
