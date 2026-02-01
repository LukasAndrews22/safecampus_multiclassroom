import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch

# Import your training functions
from ppo_centralized import run_centralized_training
from ppo_ctde import run_marl_session

# ==========================================
# CONFIGURATION
# ==========================================
CLASSROOM_COUNTS = [2, 4, 8, 16]
BENCHMARK_EPISODES = 1000  # Lower than 3000 to keep the benchmark quick
OMEGA = 0.5                # Balanced preference
LR = 0.002                 # Fixed learning rate for fair comparison
SEED = 42
OUTPUT_DIR = "scalability_results"

def run_benchmark():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    print(f"{'='*60}")
    print(f"SCALABILITY BENCHMARK: Centralized vs. Decentralized")
    print(f"Classrooms to test: {CLASSROOM_COUNTS}")
    print(f"Episodes per run:   {BENCHMARK_EPISODES}")
    print(f"Device:             {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"{'='*60}")

    for n_classrooms in CLASSROOM_COUNTS:
        print(f"\n>>> TESTING N={n_classrooms} CLASSROOMS <<<")

        # -------------------------------------------------
        # 1. Test Centralized PPO
        # -------------------------------------------------
        print(f"  [1/2] Running Centralized PPO...")
        start_time = time.time()
        try:
            # We ignore the returned agent object (_)
            _, cent_history = run_centralized_training(
                omega=OMEGA, 
                seed=SEED, 
                lr=LR, 
                episodes=BENCHMARK_EPISODES, 
                num_classrooms=n_classrooms
            )
            cent_time = time.time() - start_time
            # Average reward of the last 100 episodes
            cent_final_perf = np.mean(cent_history[-100:])
            print(f"    -> Finished in {cent_time:.1f}s | Final Reward: {cent_final_perf:.2f}")
        except Exception as e:
            print(f"    -> FAILED: {e}")
            cent_time = np.nan
            cent_final_perf = np.nan

        # -------------------------------------------------
        # 2. Test Decentralized (MAPPO/CTDE)
        # -------------------------------------------------
        print(f"  [2/2] Running Decentralized CTDE (Beta)...")
        start_time = time.time()
        try:
            _, ctde_history = run_marl_session(
                omega=OMEGA, 
                seed=SEED, 
                lr=LR, 
                episodes=BENCHMARK_EPISODES, 
                num_classrooms=n_classrooms,
                policy_type='beta'
            )
            ctde_time = time.time() - start_time
            ctde_final_perf = np.mean(ctde_history[-100:])
            print(f"    -> Finished in {ctde_time:.1f}s | Final Reward: {ctde_final_perf:.2f}")
        except Exception as e:
            print(f"    -> FAILED: {e}")
            ctde_time = np.nan
            ctde_final_perf = np.nan

        # Store Data
        results.append({
            'num_classrooms': n_classrooms,
            'centralized_time': cent_time,
            'centralized_reward': cent_final_perf,
            'ctde_time': ctde_time,
            'ctde_reward': ctde_final_perf
        })

    # -------------------------------------------------
    # Save and Plot
    # -------------------------------------------------
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "scalability_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nBenchmark complete. Data saved to {csv_path}")
    
    plot_results(df)

def plot_results(df):
    """Generates comparison plots for Time and Performance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Training Time (Lower is better)
    ax1.plot(df['num_classrooms'], df['centralized_time'], 'o-', label='Centralized (1 Network)', color='#1f77b4', linewidth=2)
    ax1.plot(df['num_classrooms'], df['ctde_time'], 's-', label='Decentralized (N Networks)', color='#ff7f0e', linewidth=2)
    ax1.set_xlabel('Number of Classrooms (N)', fontweight='bold')
    ax1.set_ylabel('Training Time (seconds)', fontweight='bold')
    ax1.set_title('Computational Scalability', fontweight='bold')
    ax1.set_xticks(df['num_classrooms'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Performance (Higher is better)
    ax1.set_ylim(bottom=0) # Time usually starts at 0
    
    ax2.plot(df['num_classrooms'], df['centralized_reward'], 'o-', label='Centralized', color='#1f77b4', linewidth=2)
    ax2.plot(df['num_classrooms'], df['ctde_reward'], 's-', label='Decentralized', color='#ff7f0e', linewidth=2)
    ax2.set_xlabel('Number of Classrooms (N)', fontweight='bold')
    ax2.set_ylabel('Final Average Reward', fontweight='bold')
    ax2.set_title('Performance Scalability', fontweight='bold')
    ax2.set_xticks(df['num_classrooms'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, "scalability_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    run_benchmark()