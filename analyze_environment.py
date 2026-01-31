"""
Comprehensive Environment Analysis for Multi-Classroom Epidemic Control

Computes:
1. Dynamic Programming (Backward Induction) Upper Bound
2. Myopic Optimal Policy (one-step lookahead)
3. Evaluates trained RL models (CTDE, Centralized)
4. Random baseline

IMPORTANT: All policies use the actual environment for transitions and rewards.
No dynamics are reimplemented - we query the environment directly.

Author: SafeCampus Project
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import os
import json
import time
import pandas as pd
import copy

from environment.multiclassroom import MultiClassroomEnv

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR = "analysis_results_0.8_gamma"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Environment parameters
TOTAL_STUDENTS = 100
NUM_CLASSROOMS = 2
MAX_WEEKS = 15
COMMUNITY_RISK_FILE = "weekly_risk_sample_b.csv"

# DP Discretization
N_INFECTED_BINS = 21
N_ACTION_BINS = 11  # Actions: 0, 10, 20, ..., 100

# Evaluation
NUM_EVAL_EPISODES = 30
EVAL_SEED = 42

# Omega values to analyze
OMEGA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})


def set_seed(seed):
    np.random.seed(seed)


# ============================================================
# MYOPIC OPTIMAL POLICY
# ============================================================

class MyopicAgent:
    """
    Myopic (one-step lookahead) optimal policy.
    
    For each step, tries all possible action combinations using the actual
    environment and picks the one with best immediate reward.
    
    Uses environment directly - no reimplemented dynamics.
    """
    
    def __init__(
        self,
        omega: float,
        num_classrooms: int = NUM_CLASSROOMS,
        total_students: int = TOTAL_STUDENTS,
        n_action_bins: int = N_ACTION_BINS
    ):
        self.omega = omega
        self.num_classrooms = num_classrooms
        self.total_students = total_students
        
        # Discrete action options to search over
        self.action_options = np.linspace(0, total_students, n_action_bins)
        self.n_actions = n_action_bins
    
    def _create_env_copy(self, env: MultiClassroomEnv) -> MultiClassroomEnv:
        """Create a copy of environment state for lookahead."""
        env_copy = MultiClassroomEnv(
            num_classrooms=self.num_classrooms,
            total_students=self.total_students,
            max_weeks=MAX_WEEKS,
            gamma=self.omega,
            continuous_action=True,
            cooperative_reward=True,
            eval_mode=True,
            community_risk_data_file=COMMUNITY_RISK_FILE
        )
        # Copy current state
        env_copy.student_status = list(env.student_status)
        env_copy.current_week = env.current_week
        if hasattr(env, 'shared_community_risk'):
            env_copy.shared_community_risk = env.shared_community_risk.copy()
        return env_copy
    
    def select_action(self, env: MultiClassroomEnv) -> List[float]:
        """
        Select best joint action via grid search.
        
        Tries all action combinations on copies of the environment
        and returns the one with highest immediate reward.
        """
        agent_ids = sorted(env.agents)
        best_actions = [0.0] * self.num_classrooms
        best_reward = -float('inf')
        
        if self.num_classrooms == 2:
            # Two classrooms - nested loop
            for a1 in self.action_options:
                for a2 in self.action_options:
                    # Create a copy to test this action
                    env_copy = self._create_env_copy(env)
                    
                    actions_for_env = {
                        agent_ids[0]: np.array([a1]),
                        agent_ids[1]: np.array([a2])
                    }
                    
                    # Step the copy to get reward
                    _, rewards, _, _ = env_copy.step(actions_for_env)
                    reward = sum(rewards.values()) / len(agent_ids)
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_actions = [a1, a2]
        else:
            # General case
            from itertools import product
            for actions_tuple in product(self.action_options, repeat=self.num_classrooms):
                actions = list(actions_tuple)
                
                env_copy = self._create_env_copy(env)
                
                actions_for_env = {
                    aid: np.array([actions[i]])
                    for i, aid in enumerate(agent_ids)
                }
                
                _, rewards, _, _ = env_copy.step(actions_for_env)
                reward = sum(rewards.values()) / len(agent_ids)
                
                if reward > best_reward:
                    best_reward = reward
                    best_actions = actions
        
        return best_actions
    
    def evaluate(
        self, 
        num_episodes: int = NUM_EVAL_EPISODES, 
        seed: int = EVAL_SEED
    ) -> Tuple[float, float, List[float]]:
        """Evaluate myopic policy on actual environment."""
        set_seed(seed)
        
        episode_rewards = []
        
        for ep in range(num_episodes):
            env = MultiClassroomEnv(
                num_classrooms=self.num_classrooms,
                total_students=self.total_students,
                max_weeks=MAX_WEEKS,
                gamma=self.omega,
                continuous_action=True,
                cooperative_reward=True,
                eval_mode=True,
                community_risk_data_file=COMMUNITY_RISK_FILE
            )
            
            obs = env.reset()
            agent_ids = sorted(env.agents)
            
            ep_reward = 0
            done = False
            
            while not done:
                # Get myopic optimal action by searching over all options
                actions = self.select_action(env)
                
                # Execute in environment
                actions_for_env = {
                    aid: np.array([actions[i]])
                    for i, aid in enumerate(agent_ids)
                }
                
                obs, rewards, dones, _ = env.step(actions_for_env)
                ep_reward += sum(rewards.values()) / len(agent_ids)
                
                done = any(dones.values())
            
            episode_rewards.append(ep_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards), episode_rewards


# ============================================================
# DYNAMIC PROGRAMMING UPPER BOUND
# ============================================================

class DPUpperBound:
    """
    Backward Induction DP Solver.
    
    For a FIXED risk trajectory (eval_mode), computes optimal policy via 
    backward induction. This is an UPPER BOUND because RL agents don't 
    know future risks.
    
    Uses actual environment for all transitions and rewards.
    
    State: (I_1, I_2, ..., I_N, t) - discretized infected counts and time
    The risk r_t is deterministic in eval_mode so we don't need it in state.
    """
    
    def __init__(
        self,
        omega: float,
        num_classrooms: int = NUM_CLASSROOMS,
        total_students: int = TOTAL_STUDENTS,
        max_weeks: int = MAX_WEEKS,
        n_infected_bins: int = N_INFECTED_BINS,
        n_action_bins: int = N_ACTION_BINS
    ):
        self.omega = omega
        self.num_classrooms = num_classrooms
        self.total_students = total_students
        self.max_weeks = max_weeks
        self.n_infected_bins = n_infected_bins
        self.n_action_bins = n_action_bins
        
        # Discretization grids
        self.infected_vals = np.linspace(0, total_students, n_infected_bins).astype(int)
        self.action_vals = np.linspace(0, total_students, n_action_bins)
        
        # Value function and policy storage
        self.V = None
        self.policy = None
    
    def _get_infected_index(self, infected: int) -> int:
        """Map infected count to nearest grid index."""
        idx = np.argmin(np.abs(self.infected_vals - infected))
        return idx
    
    def _create_env_at_state(self, infected: List[int], week: int) -> MultiClassroomEnv:
        """Create environment at a specific state."""
        env = MultiClassroomEnv(
            num_classrooms=self.num_classrooms,
            total_students=self.total_students,
            max_weeks=self.max_weeks,
            gamma=self.omega,
            continuous_action=True,
            cooperative_reward=True,
            eval_mode=True,
            community_risk_data_file=COMMUNITY_RISK_FILE
        )
        env.reset()
        # Set state
        env.student_status = list(infected)
        env.current_week = week
        return env
    
    def _simulate_step(
        self, 
        infected: List[int], 
        actions: List[float], 
        week: int
    ) -> Tuple[List[int], float]:
        """
        Simulate one step using actual environment.
        
        Returns: (next_infected, reward)
        """
        env = self._create_env_at_state(infected, week)
        agent_ids = sorted(env.agents)
        
        actions_for_env = {
            aid: np.array([actions[i]])
            for i, aid in enumerate(agent_ids)
        }
        
        _, rewards, _, _ = env.step(actions_for_env)
        
        next_infected = list(env.student_status)
        reward = sum(rewards.values()) / len(agent_ids)
        
        return next_infected, reward
    
    def solve(self, verbose: bool = True) -> Dict:
        """
        Run backward induction to compute optimal policy.
        
        For N=2 classrooms:
        State: (i1_idx, i2_idx, t)
        V[t, i1, i2] = max over actions { R + V[t+1, i1', i2'] }
        """
        T = self.max_weeks
        n_inf = self.n_infected_bins
        N = self.num_classrooms
        
        if verbose:
            print("=" * 70)
            print("DP UPPER BOUND SOLVER (Backward Induction)")
            print("=" * 70)
            print(f"Classrooms: {N}, Horizon: {T}, ω: {self.omega}")
            print(f"State grid: {n_inf}^{N} = {n_inf**N} states per time")
            print(f"Action grid: {self.n_action_bins}^{N} = {self.n_action_bins**N} joint actions")
            print()
        
        if N != 2:
            raise NotImplementedError("Currently only supports 2 classrooms for DP")
        
        # Initialize value function: V[t, i1_idx, i2_idx]
        # V[T, :, :] = 0 (terminal value)
        self.V = np.zeros((T + 1, n_inf, n_inf))
        
        # Policy: policy[t, i1_idx, i2_idx, agent] = action value
        self.policy = np.zeros((T, n_inf, n_inf, N))
        
        start_time = time.time()
        
        # Backward induction: t = T-1, T-2, ..., 0
        for t in range(T - 1, -1, -1):
            if verbose:
                print(f"  Processing t={t}...", end=" ", flush=True)
            
            for i1_idx, i1 in enumerate(self.infected_vals):
                for i2_idx, i2 in enumerate(self.infected_vals):
                    infected = [int(i1), int(i2)]
                    
                    best_value = -float('inf')
                    best_actions = [0.0, 0.0]
                    
                    # Grid search over all action combinations
                    for a1 in self.action_vals:
                        for a2 in self.action_vals:
                            actions = [a1, a2]
                            
                            # Use environment to compute transition
                            next_infected, reward = self._simulate_step(infected, actions, t)
                            
                            # Get next state indices
                            next_i1_idx = self._get_infected_index(next_infected[0])
                            next_i2_idx = self._get_infected_index(next_infected[1])
                            
                            # Bellman update: Q = R + V_{t+1}
                            future_value = self.V[t + 1, next_i1_idx, next_i2_idx]
                            q_value = reward + future_value
                            
                            if q_value > best_value:
                                best_value = q_value
                                best_actions = actions
                    
                    self.V[t, i1_idx, i2_idx] = best_value
                    self.policy[t, i1_idx, i2_idx, :] = best_actions
            
            if verbose:
                avg_v = self.V[t].mean()
                print(f"avg V={avg_v:.2f}")
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\nSolve time: {elapsed:.1f}s")
            # Value at typical starting state
            i1_idx = self._get_infected_index(3)
            i2_idx = self._get_infected_index(3)
            print(f"V*(I=(3,3), t=0) = {self.V[0, i1_idx, i2_idx]:.2f}")
        
        return {
            'V': self.V,
            'policy': self.policy,
            'solve_time': elapsed
        }
    
    def get_optimal_action(self, t: int, infected: List[int]) -> List[float]:
        """Get optimal action for given time and state."""
        i1_idx = self._get_infected_index(infected[0])
        i2_idx = self._get_infected_index(infected[1])
        return list(self.policy[t, i1_idx, i2_idx, :])
    
    def evaluate(
        self, 
        num_episodes: int = NUM_EVAL_EPISODES, 
        seed: int = EVAL_SEED
    ) -> Tuple[float, float, List[float]]:
        """Evaluate the DP policy on the environment."""
        if self.policy is None:
            raise ValueError("Must call solve() before evaluate()")
        
        set_seed(seed)
        episode_rewards = []
        
        for ep in range(num_episodes):
            env = MultiClassroomEnv(
                num_classrooms=self.num_classrooms,
                total_students=self.total_students,
                max_weeks=self.max_weeks,
                gamma=self.omega,
                continuous_action=True,
                cooperative_reward=True,
                eval_mode=True,
                community_risk_data_file=COMMUNITY_RISK_FILE
            )
            
            obs = env.reset()
            agent_ids = sorted(env.agents)
            
            ep_reward = 0
            done = False
            t = 0
            
            while not done:
                # Get current infected from environment
                infected = [env.student_status[i] for i in range(self.num_classrooms)]
                
                # Get DP optimal action
                actions = self.get_optimal_action(t, infected)
                
                # Execute
                actions_for_env = {
                    aid: np.array([actions[i]])
                    for i, aid in enumerate(agent_ids)
                }
                
                obs, rewards, dones, _ = env.step(actions_for_env)
                ep_reward += sum(rewards.values()) / len(agent_ids)
                
                done = any(dones.values())
                t += 1
            
            episode_rewards.append(ep_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards), episode_rewards


# ============================================================
# RANDOM BASELINE
# ============================================================

class RandomPolicy:
    """Random uniform actions baseline."""
    
    def __init__(
        self,
        omega: float,
        num_classrooms: int = NUM_CLASSROOMS,
        total_students: int = TOTAL_STUDENTS
    ):
        self.omega = omega
        self.num_classrooms = num_classrooms
        self.total_students = total_students
    
    def evaluate(
        self, 
        num_episodes: int = NUM_EVAL_EPISODES, 
        seed: int = EVAL_SEED
    ) -> Tuple[float, float, List[float]]:
        """Evaluate random policy."""
        set_seed(seed)
        episode_rewards = []
        
        for ep in range(num_episodes):
            env = MultiClassroomEnv(
                num_classrooms=self.num_classrooms,
                total_students=self.total_students,
                max_weeks=MAX_WEEKS,
                gamma=self.omega,
                continuous_action=True,
                cooperative_reward=True,
                eval_mode=True,
                community_risk_data_file=COMMUNITY_RISK_FILE
            )
            
            obs = env.reset()
            agent_ids = sorted(env.agents)
            
            ep_reward = 0
            done = False
            
            while not done:
                actions_for_env = {
                    aid: np.array([np.random.rand() * self.total_students])
                    for aid in agent_ids
                }
                
                obs, rewards, dones, _ = env.step(actions_for_env)
                ep_reward += sum(rewards.values()) / len(agent_ids)
                done = any(dones.values())
            
            episode_rewards.append(ep_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards), episode_rewards


# ============================================================
# RL MODEL EVALUATION
# ============================================================

def evaluate_ctde_model(
    omega: float, 
    num_episodes: int = NUM_EVAL_EPISODES, 
    seed: int = EVAL_SEED
) -> Tuple[Optional[float], Optional[float], Optional[List[float]]]:
    """Load and evaluate trained CTDE model."""
    try:
        import torch
        
        # Try different import paths
        try:
            from mappo_ctde_beta import MAPPO_CTDE, normalize_state
        except ImportError:
            try:
                from ppo_ctde import MAPPO_CTDE, normalize_state
            except ImportError:
                print("    Could not import CTDE module")
                return None, None, None
        
        model_path = f"mappo_results/models/mappo_omega_{omega}_run_0"
        if not os.path.exists(model_path + '.pt'):
            print(f"    CTDE model not found: {model_path}.pt")
            return None, None, None
        
        model = MAPPO_CTDE.load(model_path)
        device = torch.device("cpu")
        
        set_seed(seed)
        episode_rewards = []
        
        for ep in range(num_episodes):
            env = MultiClassroomEnv(
                num_classrooms=NUM_CLASSROOMS,
                total_students=TOTAL_STUDENTS,
                max_weeks=MAX_WEEKS,
                gamma=omega,
                continuous_action=True,
                cooperative_reward=True,
                eval_mode=True,
                community_risk_data_file=COMMUNITY_RISK_FILE
            )
            
            obs = env.reset()
            agent_ids = sorted(env.agents)
            
            ep_reward = 0
            done = False
            
            while not done:
                actions_for_env = {}
                for i, aid in enumerate(agent_ids):
                    state = normalize_state(obs[aid], TOTAL_STUDENTS)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        action = model.actors[i].get_deterministic_action(state_tensor)
                    actions_for_env[aid] = action.cpu().numpy().flatten() * TOTAL_STUDENTS
                
                obs, rewards, dones, _ = env.step(actions_for_env)
                ep_reward += sum(rewards.values()) / len(agent_ids)
                done = any(dones.values())
            
            episode_rewards.append(ep_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards), episode_rewards
    
    except Exception as e:
        print(f"    Warning: Could not evaluate CTDE model: {e}")
        return None, None, None


def evaluate_centralized_model(
    omega: float, 
    num_episodes: int = NUM_EVAL_EPISODES, 
    seed: int = EVAL_SEED
) -> Tuple[Optional[float], Optional[float], Optional[List[float]]]:
    """Load and evaluate trained Centralized model."""
    try:
        import torch
        from ppo_centralized import CentralizedPPO
        
        model_path = f"centralized_ppo_results/models/centralized_omega_{omega}_run_0"
        if not os.path.exists(model_path + '.pt'):
            print(f"    Centralized model not found: {model_path}.pt")
            return None, None, None
        
        model = CentralizedPPO.load(model_path)
        device = torch.device("cpu")
        
        set_seed(seed)
        episode_rewards = []
        
        for ep in range(num_episodes):
            env = MultiClassroomEnv(
                num_classrooms=NUM_CLASSROOMS,
                total_students=TOTAL_STUDENTS,
                max_weeks=MAX_WEEKS,
                gamma=omega,
                continuous_action=True,
                cooperative_reward=True,
                eval_mode=True,
                community_risk_data_file=COMMUNITY_RISK_FILE
            )
            
            obs = env.reset()
            agent_ids = sorted(env.agents)
            
            # Build global state
            global_state = []
            for aid in agent_ids:
                global_state.append(obs[aid][0] / TOTAL_STUDENTS)
                global_state.append(obs[aid][1])
            global_state = np.array(global_state, dtype=np.float32)
            
            ep_reward = 0
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
                with torch.no_grad():
                    joint_action = model.actor.get_deterministic_action(state_tensor)
                joint_action = joint_action.cpu().numpy().flatten()
                
                actions_for_env = {
                    aid: np.array([joint_action[i] * TOTAL_STUDENTS])
                    for i, aid in enumerate(agent_ids)
                }
                
                obs, rewards, dones, _ = env.step(actions_for_env)
                ep_reward += sum(rewards.values()) / len(agent_ids)
                
                # Update global state
                global_state = []
                for aid in agent_ids:
                    global_state.append(obs[aid][0] / TOTAL_STUDENTS)
                    global_state.append(obs[aid][1])
                global_state = np.array(global_state, dtype=np.float32)
                
                done = any(dones.values())
            
            episode_rewards.append(ep_reward)
        
        return np.mean(episode_rewards), np.std(episode_rewards), episode_rewards
    
    except Exception as e:
        print(f"    Warning: Could not evaluate Centralized model: {e}")
        return None, None, None


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_full_analysis():
    """Run comprehensive analysis for all omega values."""
    print("=" * 80)
    print("COMPREHENSIVE ENVIRONMENT ANALYSIS")
    print("Multi-Classroom Epidemic Control")
    print("=" * 80)
    print(f"Classrooms: {NUM_CLASSROOMS}, Students: {TOTAL_STUDENTS}, Horizon: {MAX_WEEKS}")
    print(f"Omega values: {OMEGA_VALUES}")
    print()
    
    results = {}
    
    for omega in OMEGA_VALUES:
        print(f"\n{'=' * 80}")
        print(f"OMEGA = {omega}")
        print("=" * 80)
        
        results[omega] = {}
        
        # # 1. DP Upper Bound
        # print("\n[1] Computing DP Upper Bound...")
        # dp_solver = DPUpperBound(
        #     omega=omega,
        #     num_classrooms=NUM_CLASSROOMS,
        #     n_infected_bins=N_INFECTED_BINS,
        #     n_action_bins=N_ACTION_BINS
        # )
        # dp_solver.solve(verbose=True)
        
        # print("  Evaluating DP policy...")
        # dp_mean, dp_std, _ = dp_solver.evaluate()
        # results[omega]['dp'] = {'mean': dp_mean, 'std': dp_std}
        # print(f"  DP Reward: {dp_mean:.2f} ± {dp_std:.2f}")
        
        # 2. Myopic Optimal
        print("\n[2] Evaluating Myopic Optimal...")
        myopic = MyopicAgent(omega=omega, num_classrooms=NUM_CLASSROOMS)
        myopic_mean, myopic_std, _ = myopic.evaluate()
        results[omega]['myopic'] = {'mean': myopic_mean, 'std': myopic_std}
        print(f"  Myopic Reward: {myopic_mean:.2f} ± {myopic_std:.2f}")
        
        # 3. CTDE
        print("\n[3] Evaluating CTDE (MAPPO)...")
        ctde_mean, ctde_std, _ = evaluate_ctde_model(omega)
        if ctde_mean is not None:
            results[omega]['ctde'] = {'mean': ctde_mean, 'std': ctde_std}
            print(f"  CTDE Reward: {ctde_mean:.2f} ± {ctde_std:.2f}")
        else:
            results[omega]['ctde'] = None
            print("  CTDE: Model not found")
        
        # 4. Centralized
        print("\n[4] Evaluating Centralized PPO...")
        cent_mean, cent_std, _ = evaluate_centralized_model(omega)
        if cent_mean is not None:
            results[omega]['centralized'] = {'mean': cent_mean, 'std': cent_std}
            print(f"  Centralized Reward: {cent_mean:.2f} ± {cent_std:.2f}")
        else:
            results[omega]['centralized'] = None
            print("  Centralized: Model not found")
        
        # 5. Random
        print("\n[5] Evaluating Random...")
        random_policy = RandomPolicy(omega=omega)
        random_mean, random_std, _ = random_policy.evaluate()
        results[omega]['random'] = {'mean': random_mean, 'std': random_std}
        print(f"  Random Reward: {random_mean:.2f} ± {random_std:.2f}")
    
    # Generate outputs
    generate_summary_table(results)
    generate_comparison_plot(results)
    save_results(results)
    
    return results


def generate_summary_table(results: Dict):
    """Generate and print summary table."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    
    rows = []
    for omega in OMEGA_VALUES:
        r = results[omega]
        row = {
            'omega': omega,
            # 'dp_mean': r['dp']['mean'],
            # 'dp_std': r['dp']['std'],
            'myopic_mean': r['myopic']['mean'],
            'myopic_std': r['myopic']['std'],
            'centralized_mean': r['centralized']['mean'] if r.get('centralized') else np.nan,
            'centralized_std': r['centralized']['std'] if r.get('centralized') else np.nan,
            'ctde_mean': r['ctde']['mean'] if r.get('ctde') else np.nan,
            'ctde_std': r['ctde']['std'] if r.get('ctde') else np.nan,
            'random_mean': r['random']['mean'],
            'random_std': r['random']['std']
        }
        rows.append(row)
        
        cent_str = f"{row['centralized_mean']:.1f}" if not np.isnan(row['centralized_mean']) else "N/A"
        ctde_str = f"{row['ctde_mean']:.1f}" if not np.isnan(row['ctde_mean']) else "N/A"
        
        print(f"omega={omega}: "
              f"Myopic={row['myopic_mean']:.1f}±{row['myopic_std']:.1f}, "
              f"Cent={cent_str}, CTDE={ctde_str}, "
              f"Random={row['random_mean']:.1f}")
    
    # Save to CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "analysis_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


def generate_comparison_plot(results: Dict):
    """Generate comparison bar chart."""
    print("\n--- Generating Comparison Plot ---")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(OMEGA_VALUES))
    width = 0.15
    
    methods = ['myopic', 'centralized', 'ctde', 'random']
    colors = ['forestgreen', 'steelblue', 'coral', 'gray']
    labels = ['Myopic', 'Centralized', 'CTDE', 'Random']
    
    for i, (method, color, label) in enumerate(zip(methods, colors, labels)):
        means = []
        stds = []
        for omega in OMEGA_VALUES:
            if results[omega].get(method) and results[omega][method] is not None:
                means.append(results[omega][method]['mean'])
                stds.append(results[omega][method]['std'])
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=label, color=color, capsize=2)
    
    ax.set_xlabel('Omega', fontweight='bold')
    ax.set_ylabel('Reward', fontweight='bold')
    ax.set_title('Performance Comparison: Multi-Classroom Epidemic Control', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{o}' for o in OMEGA_VALUES])
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "performance_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_path}")


def save_results(results: Dict):
    """Save results to JSON."""
    serializable = {}
    for omega in OMEGA_VALUES:
        serializable[str(omega)] = {}
        for method in [ 'myopic', 'centralized', 'ctde', 'random']:
            if results[omega].get(method) and results[omega][method] is not None:
                serializable[str(omega)][method] = {
                    'mean': float(results[omega][method]['mean']),
                    'std': float(results[omega][method]['std'])
                }
    
    json_path = os.path.join(OUTPUT_DIR, "analysis_results.json")
    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=4)
    print(f"Results saved to {json_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results = run_full_analysis()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to {OUTPUT_DIR}/")