import os
import subprocess
import json
import time
import matplotlib.pyplot as plt
import numpy as np

# Configuration
N_VALUES = [2, 4, 8, 16]
PYTHON_EXEC = r"venv\Scripts\python.exe"
OUTPUT_DIR = "analysis_results"

def run_experiment_for_n(n):
    print("=" * 80)
    print(f"RUNNING SCALABILITY EXPERIMENT FOR N = {n} CLASSROOMS")
    print("=" * 80)
    env = os.environ.copy()
    env['NUM_CLASSROOMS'] = str(n)
    env['FAST_MODE'] = '1'
    
    # Measure Centralized Training Time
    print(f"\nTraining Centralized PPO for N={n}...")
    start_time = time.time()
    subprocess.run([PYTHON_EXEC, "ppo_centralized.py"], env=env, check=True)
    centralized_time = time.time() - start_time

    # Measure CTDE Training Time
    print(f"\nTraining CTDE (MAPPO) for N={n}...")
    start_time = time.time()
    subprocess.run([PYTHON_EXEC, "ppo_ctde.py"], env=env, check=True)
    ctde_time = time.time() - start_time

    # Run Analysis
    print(f"\nRunning Analysis for N={n}...")
    subprocess.run([PYTHON_EXEC, "analyze_environment.py"], env=env, check=True)

    # Parse Evaluation Results
    results_path = os.path.join(OUTPUT_DIR, "analysis_results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Could not find {results_path} after analysis.")
        
    if n == 2:
        import shutil
        print("Backing up N=2 results...")
        if os.path.exists("centralized_ppo_results_N2"):
            shutil.rmtree("centralized_ppo_results_N2")
        shutil.copytree("centralized_ppo_results", "centralized_ppo_results_N2")
        
        if os.path.exists("mappo_results_N2"):
            shutil.rmtree("mappo_results_N2")
        shutil.copytree("mappo_results", "mappo_results_N2")
        
        if os.path.exists("analysis_results_N2"):
            shutil.rmtree("analysis_results_N2")
        shutil.copytree("analysis_results", "analysis_results_N2")

    with open(results_path, 'r') as f:
        results = json.load(f)
        
    centralized_rewards = []
    ctde_rewards = []
    
    for omega, data in results.items():
        if "centralized" in data and data["centralized"] is not None:
            centralized_rewards.append(data["centralized"]["mean"])
        if "ctde" in data and data["ctde"] is not None:
            ctde_rewards.append(data["ctde"]["mean"])
            
    centralized_avg_reward = np.mean(centralized_rewards) if centralized_rewards else 0.0
    ctde_avg_reward = np.mean(ctde_rewards) if ctde_rewards else 0.0

    return {
        "N": n,
        "centralized_time": centralized_time,
        "ctde_time": ctde_time,
        "centralized_reward": centralized_avg_reward,
        "ctde_reward": ctde_avg_reward
    }

def main():
    scalability_data = []
    
    for n in N_VALUES:
        data = run_experiment_for_n(n)
        scalability_data.append(data)
        
        print(f"\nResults for N={n}:")
        print(f"  Centralized: Time={data['centralized_time']:.1f}s, Reward={data['centralized_reward']:.2f}")
        print(f"  CTDE:        Time={data['ctde_time']:.1f}s, Reward={data['ctde_reward']:.2f}")
        
    # Plotting
    print("\nGenerating Scalability Plots...")
    n_arr = [d["N"] for d in scalability_data]
    cent_times = [d["centralized_time"] for d in scalability_data]
    ctde_times = [d["ctde_time"] for d in scalability_data]
    cent_rewards = [d["centralized_reward"] for d in scalability_data]
    ctde_rewards = [d["ctde_reward"] for d in scalability_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Computational Scalability (Training Time)
    ax1.plot(n_arr, cent_times, marker='o', label='Centralized (1 Network)', color='steelblue', linewidth=2)
    ax1.plot(n_arr, ctde_times, marker='s', label='Decentralized (N Networks)', color='darkorange', linewidth=2)
    ax1.set_xlabel('Number of Classrooms (N)', fontweight='bold')
    ax1.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax1.set_title('Computational Scalability', fontweight='bold')
    ax1.set_xticks(n_arr)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # Plot 2: Performance Scalability (Final Average Reward)
    ax2.plot(n_arr, cent_rewards, marker='o', label='Centralized', color='steelblue', linewidth=2)
    ax2.plot(n_arr, ctde_rewards, marker='s', label='Decentralized', color='darkorange', linewidth=2)
    ax2.set_xlabel('Number of Classrooms (N)', fontweight='bold')
    ax2.set_ylabel('Final Average Reward', fontweight='bold')
    ax2.set_title('Performance Scalability', fontweight='bold')
    ax2.set_xticks(n_arr)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = "scalability.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Scalability plots saved to {plot_path}")
    
    # Save raw data
    with open('scalability_data.json', 'w') as f:
        json.dump(scalability_data, f, indent=4)

if __name__ == "__main__":
    main()
