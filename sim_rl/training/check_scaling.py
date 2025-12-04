"""
Diagnostic script to verify observation scaling for the Repo environment.
Checks if neural network inputs are within a healthy range (approx -3 to 3).
"""
import numpy as np
from sim_rl.cr3bp.scenarios import SCENARIOS
from sim_rl.cr3bp.env_cr3bp_station_keeping_repo import Cr3bpStationKeepingEnvRepo
from sim_rl.cr3bp.constants import SCALE_POS, SCALE_VEL

def check_scaling():
    scenario_name = "earth-moon-L1-3D"
    scenario = SCENARIOS[scenario_name]
    
    print(f"\n--- REPO SCALING CHECK ---")
    print(f"Scenario: {scenario_name}")
    print(f"SCALE_POS: {SCALE_POS}")
    print(f"SCALE_VEL: {SCALE_VEL}")
    
    env = Cr3bpStationKeepingEnvRepo(scenario=scenario, use_reference_orbit=True)
    
    seed = 42
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)

    print(f"Observation Shape: {obs.shape}")
    
    max_val = -float('inf')
    min_val = float('inf')
    
    print("Collecting data from 1000 steps (Random Walk)...")
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        
        curr_max = np.max(obs)
        curr_min = np.min(obs)
        
        if curr_max > max_val: max_val = curr_max
        if curr_min < min_val: min_val = curr_min
        
        if done or truncated:
            obs, _ = env.reset()

    print("-" * 30)
    print(f"Min Observed Value (Net Input): {min_val:.4f}")
    print(f"Max Observed Value (Net Input): {max_val:.4f}")
    print("-" * 30)
    
    limit = max(abs(max_val), abs(min_val))
    if limit < 0.1:
        print("WARNING: Values are very small (< 0.1). Consider increasing SCALE (* 10).")
    elif limit > 10.0:
        print("WARNING: Values are very large (> 10.0). Consider decreasing SCALE (/ 10).")
    else:
        print("SUCCESS: Values are well within range [0.1, 10.0].")

if __name__ == "__main__":
    check_scaling()