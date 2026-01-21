"""
Centralized PPO for Multi-Classroom Epidemic Control

A single central controller observes the global state (all classrooms' infected counts + risk)
and outputs actions for all classrooms simultaneously.

This version uses TanhDeterministicActor (same as CTDE) for consistency.
- Deterministic policy with tanh output scaled to [0, 1]
- Exploration via decaying Gaussian noise
- Pseudo log-prob for PPO compatibility

Author: SafeCampus Project
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import json
import random
import time
from matplotlib.colors import LinearSegmentedColormap
import colorsys

from environment.multiclassroom import MultiClassroomEnv

# ============================================================
# 0. SETUP & CONFIGURATION
# ============================================================
GLOBAL_SEED = 42
OUTPUT_DIR = "centralized_ppo_results"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LR_FILE = os.path.join(OUTPUT_DIR, "optimized_lrs.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Training Hyperparameters
GAMMA_DISCOUNT = 0.8
GAE_LAMBDA = 0.95
K_EPOCHS = 20
EPS_CLIP = 0.2
MAX_WEEKS = 15
UPDATE_TIMESTEP = 2000
FULL_EPISODES = 3000
TUNE_EPISODES = 3000
LR_CANDIDATES = [0.0001, 0.0003, 0.001, 0.003, 0.005, 0.01]

# Omega (Preference weight)
OMEGA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Environment Settings
TOTAL_STUDENTS = 100
NUM_CLASSROOMS = 2
COOPERATIVE_REWARD = True
TUNE_SEED = 123

# Network Architecture
HIDDEN_DIM = 32
NUM_LAYERS = 2

# Evaluation
POLICY_GRID_POINTS = 20
NUM_RUNS = 1

device = torch.device("cpu")

plt.rcParams.update({
    'font.size': 12,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'lines.linewidth': 2.0,
})


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def normalize_global_state(observations, agent_ids, total_students):
    """Convert observations dict to normalized global state vector."""
    state = []
    for aid in agent_ids:
        obs = observations[aid]
        state.append(obs[0] / float(total_students))  # Normalize infected to [0, 1]
        state.append(obs[1])  # Risk already [0, 1]
    return np.array(state, dtype=np.float32)


# ============================================================
# 1. NETWORK ARCHITECTURES (Same as CTDE)
# ============================================================

def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer with orthogonal weights."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class TanhDeterministicActor(nn.Module):
    """
    Deterministic Actor with Tanh output squashing.
    
    This is the SAME architecture used in CTDE for consistency.
    - Network outputs tanh in [-1, 1], scaled to [0, 1]
    - Exploration via external Gaussian noise during training
    - Pseudo log-prob based on MSE for PPO compatibility
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64, num_layers=2):
        super(TanhDeterministicActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Exploration noise std (decays during training)
        self.noise_std = 0.3

        # Build network
        layers = []
        layers.append(init_layer(nn.Linear(state_dim, hidden_dim)))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(init_layer(nn.Linear(hidden_dim, hidden_dim)))
            layers.append(nn.Tanh())
        layers.append(init_layer(nn.Linear(hidden_dim, action_dim), std=0.01))
        layers.append(nn.Tanh())  # Bound to [-1, 1]
        
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Forward pass returning action in [0, 1].
        
        Tanh outputs [-1, 1], we scale to [0, 1].
        """
        tanh_out = self.network(state)  # [-1, 1]
        action = (tanh_out + 1.0) / 2.0  # [0, 1]
        return action

    def act(self, state, add_noise=True):
        """
        Get action with optional exploration noise.
        
        Returns:
            action: Action in [0, 1]
            log_prob: Dummy log_prob (not used in deterministic policy)
        """
        action = self.forward(state)
        
        if add_noise and self.noise_std > 0:
            noise = torch.randn_like(action) * self.noise_std
            action = action + noise
            action = torch.clamp(action, 0.0, 1.0)
        
        # Return dummy log_prob for compatibility
        log_prob = torch.zeros(action.shape[0], device=action.device)
        
        return action.detach(), log_prob.detach()

    def evaluate(self, state, action):
        """
        For deterministic policy, we use a pseudo-likelihood.
        
        Returns MSE-based pseudo log probability and zero entropy.
        """
        predicted_action = self.forward(state)
        
        # Pseudo log-prob based on squared error (for PPO compatibility)
        mse = ((predicted_action - action) ** 2).sum(dim=-1)
        pseudo_log_prob = -mse / (2 * self.noise_std ** 2 + 1e-8)
        
        # No entropy for deterministic policy
        entropy = torch.zeros_like(pseudo_log_prob)
        
        return pseudo_log_prob, entropy

    def get_deterministic_action(self, state):
        """Get deterministic action for evaluation."""
        with torch.no_grad():
            return self.forward(state)
    
    def decay_noise(self, decay_rate=0.995, min_noise=0.05):
        """Decay exploration noise."""
        self.noise_std = max(self.noise_std * decay_rate, min_noise)

    # Alias for compatibility with evaluation scripts
    def get_deterministic_actions(self, state):
        return self.get_deterministic_action(state)


class Critic(nn.Module):
    """Value function critic."""

    def __init__(self, state_dim, hidden_dim=64, num_layers=2):
        super(Critic, self).__init__()

        layers = []
        layers.append(init_layer(nn.Linear(state_dim, hidden_dim)))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(init_layer(nn.Linear(hidden_dim, hidden_dim)))
            layers.append(nn.Tanh())
        layers.append(init_layer(nn.Linear(hidden_dim, 1), std=1.0))
        
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


# ============================================================
# 2. ROLLOUT BUFFER
# ============================================================

class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []

    def __len__(self):
        return len(self.states)


# ============================================================
# 3. CENTRALIZED PPO AGENT
# ============================================================

class CentralizedPPO:
    """
    Centralized PPO that controls all agents with a single policy.
    
    Uses TanhDeterministicActor (same as CTDE) for consistency.
    """

    def __init__(self, global_state_dim, num_actions, lr, 
                 hidden_dim=64, num_layers=2):
        
        self.global_state_dim = global_state_dim
        self.num_actions = num_actions
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gamma = GAMMA_DISCOUNT
        self.gae_lambda = GAE_LAMBDA
        self.eps_clip = EPS_CLIP
        self.K_epochs = K_EPOCHS

        # Actor and Critic networks (TanhDeterministicActor - same as CTDE)
        self.actor = TanhDeterministicActor(global_state_dim, num_actions, hidden_dim, num_layers).to(device)
        self.critic = Critic(global_state_dim, hidden_dim, num_layers).to(device)
        
        # Old actor for PPO ratio computation
        self.actor_old = TanhDeterministicActor(global_state_dim, num_actions, hidden_dim, num_layers).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.MseLoss = nn.MSELoss()

    def select_actions(self, global_state, add_noise=True):
        """
        Select actions with optional exploration noise.
        
        Returns:
            actions: Actions in [0, 1] for each classroom
            log_prob: Pseudo log probability
            value: Value estimate for the state
        """
        state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            actions, log_prob = self.actor_old.act(state_tensor, add_noise=add_noise)
            value = self.critic(state_tensor)
        
        return actions.squeeze(0).cpu().numpy(), log_prob, value

    def update(self, buffer):
        """Update policy using PPO with GAE."""
        if len(buffer) == 0:
            return

        # Convert buffer data to tensors
        old_states = torch.stack(buffer.states).detach().to(device)
        old_actions = torch.stack(buffer.actions).detach().to(device)
        old_logprobs = torch.stack(buffer.logprobs).detach().to(device).squeeze()
        old_values = torch.stack(buffer.values).detach().to(device).squeeze()
        
        rewards = buffer.rewards
        is_terminals = buffer.is_terminals

        # Compute GAE advantages and returns
        advantages = []
        returns = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if is_terminals[t]:
                next_value = 0
                gae = 0
            else:
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = old_values[t + 1].item()

            delta = rewards[t] + self.gamma * next_value - old_values[t].item()
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + old_values[t].item())

        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        for _ in range(self.K_epochs):
            # Evaluate current policy
            logprobs, entropy = self.actor.evaluate(old_states, old_actions)
            logprobs = logprobs.squeeze()

            # Compute ratio (pi_new / pi_old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Actor loss (no entropy bonus for deterministic policy)
            actor_loss = -torch.min(surr1, surr2).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            # Critic loss
            state_values = self.critic(old_states).squeeze()
            critic_loss = self.MseLoss(state_values, returns)

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

        # Sync old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # Decay exploration noise
        self.actor.decay_noise()
        self.actor_old.noise_std = self.actor.noise_std

        buffer.clear()

    def save(self, path):
        """Save model to file."""
        if not path.endswith('.pt'):
            path = path + '.pt'
            
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_old_state_dict': self.actor_old.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'global_state_dim': self.global_state_dim,
            'num_actions': self.num_actions,
            'lr': self.lr,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'noise_std': self.actor.noise_std,
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        """Load model from file."""
        if not path.endswith('.pt'):
            path = path + '.pt'
            
        checkpoint = torch.load(path, map_location=device)
        
        ppo = cls(
            global_state_dim=checkpoint['global_state_dim'],
            num_actions=checkpoint['num_actions'],
            lr=checkpoint['lr'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers']
        )
        
        ppo.actor.load_state_dict(checkpoint['actor_state_dict'])
        ppo.actor_old.load_state_dict(checkpoint['actor_old_state_dict'])
        ppo.critic.load_state_dict(checkpoint['critic_state_dict'])
        ppo.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        ppo.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if 'noise_std' in checkpoint:
            ppo.actor.noise_std = checkpoint['noise_std']
            ppo.actor_old.noise_std = checkpoint['noise_std']
        
        print(f"Model loaded from {path}")
        return ppo

    # Compatibility property for evaluation scripts
    @property
    def policy(self):
        """Compatibility alias - returns actor."""
        return self.actor


# ============================================================
# 4. TRAINING FUNCTIONS
# ============================================================

def run_centralized_training(omega, seed, lr, episodes, num_classrooms=NUM_CLASSROOMS):
    """Run centralized PPO training."""
    set_seed(seed)

    env = MultiClassroomEnv(
        num_classrooms=num_classrooms,
        total_students=TOTAL_STUDENTS,
        max_weeks=MAX_WEEKS,
        gamma=omega,
        continuous_action=True,
        cooperative_reward=COOPERATIVE_REWARD
    )

    agent_ids = sorted(env.agents)
    global_state_dim = 2 * num_classrooms  # (infected, risk) per classroom
    num_actions = num_classrooms

    ppo = CentralizedPPO(
        global_state_dim=global_state_dim,
        num_actions=num_actions,
        lr=lr,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    )

    buffer = RolloutBuffer()
    reward_history = []
    time_step = 0

    for ep in range(episodes):
        obs = env.reset()
        global_state = normalize_global_state(obs, agent_ids, TOTAL_STUDENTS)
        ep_reward = 0
        done = False

        while not done:
            time_step += 1

            actions_normalized, log_prob, value = ppo.select_actions(global_state, add_noise=True)

            actions_env = {
                aid: np.array([actions_normalized[i] * TOTAL_STUDENTS])
                for i, aid in enumerate(agent_ids)
            }

            next_obs, rewards, dones, _ = env.step(actions_env)

            joint_reward = sum(rewards.values()) / len(rewards)

            # Store in buffer
            buffer.states.append(torch.FloatTensor(global_state))
            buffer.actions.append(torch.FloatTensor(actions_normalized))
            buffer.logprobs.append(log_prob)
            buffer.values.append(value.squeeze())
            buffer.rewards.append(joint_reward)
            buffer.is_terminals.append(any(dones.values()))

            global_state = normalize_global_state(next_obs, agent_ids, TOTAL_STUDENTS)
            ep_reward += joint_reward

            if time_step % UPDATE_TIMESTEP == 0:
                ppo.update(buffer)

            done = any(dones.values())

        reward_history.append(ep_reward)

    return ppo, reward_history


# ============================================================
# 5. POLICY EXTRACTION AND MONOTONICITY EVALUATION
# ============================================================

def extract_centralized_policy_grid(ppo, agent_idx=0, total_students=TOTAL_STUDENTS,
                                    grid_points=POLICY_GRID_POINTS, other_infected_frac=0.1):
    """Extract policy grid for a specific agent from centralized controller."""
    num_classrooms = ppo.num_actions
    infected_vals = np.linspace(0, total_students, grid_points)
    risk_vals = np.linspace(0, 1, grid_points)

    other_infected = other_infected_frac * total_students

    policy_grid = np.zeros((grid_points, grid_points), dtype=np.float32)

    for i, inf in enumerate(infected_vals):
        for j, risk in enumerate(risk_vals):
            global_state = []
            for k in range(num_classrooms):
                if k == agent_idx:
                    global_state.extend([inf / total_students, risk])
                else:
                    global_state.extend([other_infected / total_students, risk])
            global_state = np.array(global_state, dtype=np.float32)

            state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
            with torch.no_grad():
                actions = ppo.actor.get_deterministic_action(state_tensor)
                action = actions[0, agent_idx].item()

            policy_grid[i, j] = action

    return policy_grid


def extract_joint_policy_grid(ppo, total_students=TOTAL_STUDENTS, grid_points=POLICY_GRID_POINTS):
    """Extract policy grids for all agents assuming symmetric states."""
    num_classrooms = ppo.num_actions
    infected_vals = np.linspace(0, total_students, grid_points)
    risk_vals = np.linspace(0, 1, grid_points)

    policy_grids = [np.zeros((grid_points, grid_points), dtype=np.float32)
                   for _ in range(num_classrooms)]

    for i, inf in enumerate(infected_vals):
        for j, risk in enumerate(risk_vals):
            global_state = []
            for k in range(num_classrooms):
                global_state.extend([inf / total_students, risk])
            global_state = np.array(global_state, dtype=np.float32)

            state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
            with torch.no_grad():
                actions = ppo.actor.get_deterministic_action(state_tensor)

            for k in range(num_classrooms):
                policy_grids[k][i, j] = actions[0, k].item()

    return policy_grids


def evaluate_dominance_monotonicity(policy_grid):
    """Evaluate dominance monotonicity of a policy grid."""
    I, J = policy_grid.shape
    violations = 0
    total_comparisons = 0

    for i1 in range(I):
        for j1 in range(J):
            for i2 in range(i1, I):
                for j2 in range(j1, J):
                    if i1 == i2 and j1 == j2:
                        continue

                    total_comparisons += 1

                    if policy_grid[i2, j2] > policy_grid[i1, j1] + 1e-6:
                        violations += 1

    score = (total_comparisons - violations) / total_comparisons if total_comparisons > 0 else 1.0

    return score, {'violations': violations, 'total_comparisons': total_comparisons}


def evaluate_adjacent_monotonicity(policy_grid):
    """Adjacent-cell monotonicity check."""
    I, J = policy_grid.shape

    violations_inf = 0
    violations_risk = 0

    for j in range(J):
        for i in range(I - 1):
            if policy_grid[i + 1, j] > policy_grid[i, j] + 1e-6:
                violations_inf += 1

    for i in range(I):
        for j in range(J - 1):
            if policy_grid[i, j + 1] > policy_grid[i, j] + 1e-6:
                violations_risk += 1

    total_inf_transitions = J * (I - 1)
    total_risk_transitions = I * (J - 1)
    total_transitions = total_inf_transitions + total_risk_transitions
    total_violations = violations_inf + violations_risk

    score = (total_transitions - total_violations) / total_transitions if total_transitions > 0 else 1.0

    details = {
        'infected_violations': violations_inf,
        'risk_violations': violations_risk,
        'total_violations': total_violations,
        'total_transitions': total_transitions
    }

    return score, details


def compute_action_diversity(policy_grid, num_bins=10):
    """Compute action diversity metrics."""
    flat_actions = policy_grid.flatten()

    bins = np.linspace(0, 1, num_bins + 1)
    hist, _ = np.histogram(flat_actions, bins=bins)
    num_unique_bins = np.sum(hist > 0)

    return {
        'action_min': float(flat_actions.min()),
        'action_max': float(flat_actions.max()),
        'action_range': float(flat_actions.max() - flat_actions.min()),
        'action_std': float(flat_actions.std()),
        'num_unique_bins': int(num_unique_bins)
    }


# ============================================================
# 6. HYPERPARAMETER TUNING
# ============================================================

def select_best_lr(omega_results):
    """Select best learning rate based on monotonicity and reward."""
    sorted_results = sorted(
        omega_results,
        key=lambda r: (r['dominance_violations'], -r['avg_eval_reward'], -r['num_unique_bins'])
    )

    best = sorted_results[0]
    return best['lr'], best, {'min_violations': best['dominance_violations']}


def grid_search_tuning(num_classrooms=NUM_CLASSROOMS):
    """Grid search for best learning rate per omega."""
    optimized_lrs = {}

    print(f"\n--- Starting Grid Search Tuning (Tanh Policy) ---")
    print(f"LR Candidates: {LR_CANDIDATES}")
    print(f"Number of Classrooms: {num_classrooms}")

    for omega in OMEGA_VALUES:
        print(f"\n{'=' * 60}")
        print(f"*** Tuning Omega = {omega} ***")
        print(f"{'=' * 60}")

        omega_results = []

        for lr in LR_CANDIDATES:
            print(f"\n  Testing LR={lr}...")

            ppo, history = run_centralized_training(omega, TUNE_SEED, lr, TUNE_EPISODES, num_classrooms)

            # Evaluation
            env = MultiClassroomEnv(
                num_classrooms=num_classrooms,
                total_students=TOTAL_STUDENTS,
                max_weeks=MAX_WEEKS,
                gamma=omega,
                continuous_action=True,
                cooperative_reward=COOPERATIVE_REWARD,
                eval_mode=True,
                community_risk_data_file="weekly_risk_sample_b.csv"
            )
            agent_ids = sorted(env.agents)
            obs = env.reset()
            global_state = normalize_global_state(obs, agent_ids, TOTAL_STUDENTS)
            done = False
            avg_eval_reward = 0
            
            while not done:
                state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
                with torch.no_grad():
                    actions_normalized = ppo.actor.get_deterministic_action(state_tensor).cpu().numpy().flatten()
                actions_env = {aid: np.array([actions_normalized[i] * TOTAL_STUDENTS]) for i, aid in enumerate(agent_ids)}
                next_obs, rewards, dones, _ = env.step(actions_env)
                avg_eval_reward += sum(rewards.values()) / len(rewards)
                global_state = normalize_global_state(next_obs, agent_ids, TOTAL_STUDENTS)
                done = any(dones.values())

            # Extract and evaluate policy
            policy_grids = extract_joint_policy_grid(ppo)

            total_violations = 0
            mono_scores = []
            diversity_metrics = []

            for agent_idx, policy_grid in enumerate(policy_grids):
                mono, details = evaluate_dominance_monotonicity(policy_grid)
                mono_scores.append(mono)
                total_violations += details['violations']
                diversity_metrics.append(compute_action_diversity(policy_grid))

            avg_mono_score = np.mean(mono_scores)
            avg_unique_bins = np.mean([d['num_unique_bins'] for d in diversity_metrics])
            avg_action_range = np.mean([d['action_range'] for d in diversity_metrics])

            result = {
                'lr': lr,
                'avg_eval_reward': avg_eval_reward,
                'dominance_violations': total_violations,
                'avg_mono_score': avg_mono_score,
                'mono_scores': mono_scores,
                'num_unique_bins': avg_unique_bins,
                'action_range': avg_action_range
            }
            omega_results.append(result)

            print(f"    Reward: {avg_eval_reward:.2f}")
            print(f"    Avg Monotonicity: {avg_mono_score:.4f}")
            print(f"    Total Violations: {total_violations}")

        best_lr, best_metrics, _ = select_best_lr(omega_results)
        optimized_lrs[str(omega)] = best_lr
        print(f"\n  --> Best LR: {best_lr}")

    with open(LR_FILE, 'w') as f:
        json.dump(optimized_lrs, f, indent=4)

    return {float(k): v for k, v in optimized_lrs.items()}


def load_lrs():
    """Load optimized LRs from file."""
    if os.path.exists(LR_FILE):
        with open(LR_FILE, 'r') as f:
            return {float(k): float(v) for k, v in json.load(f).items()}
    return {omega: 0.001 for omega in OMEGA_VALUES}


# ============================================================
# 7. FULL TRAINING AND EVALUATION
# ============================================================

def train_and_evaluate_optimal(optimized_lrs, num_classrooms=NUM_CLASSROOMS):
    """Run full training with optimized LRs and save models."""
    print(f"\n--- Starting Full Training (Tanh Policy) ---")
    print(f"Number of Classrooms: {num_classrooms}")

    all_rewards = {}
    trained_agents = {}
    monotonicity_scores = {}

    for omega in OMEGA_VALUES:
        lr = optimized_lrs.get(omega, 0.001)
        print(f"\n{'=' * 60}")
        print(f"*** Training Omega = {omega}, LR = {lr} ***")
        print(f"{'=' * 60}")

        for run in range(NUM_RUNS):
            seed = TUNE_SEED
            print(f"\n  Run {run + 1}/{NUM_RUNS} (seed={seed})")

            ppo, history = run_centralized_training(omega, seed, lr, FULL_EPISODES, num_classrooms)

            # Save model
            model_path = os.path.join(MODEL_DIR, f"centralized_omega_{omega}_run_{run}")
            ppo.save(model_path)

            if run == 0:
                all_rewards[omega] = np.array([history])
                trained_agents[omega] = ppo

        # Evaluate monotonicity
        policy_grids = extract_joint_policy_grid(trained_agents[omega])

        dom_scores = []
        dom_violations = 0
        adj_scores = []
        adj_violations = 0

        for agent_idx, policy_grid in enumerate(policy_grids):
            mono, details = evaluate_dominance_monotonicity(policy_grid)
            dom_scores.append(mono)
            dom_violations += details['violations']

            adj, adj_details = evaluate_adjacent_monotonicity(policy_grid)
            adj_scores.append(adj)
            adj_violations += adj_details['total_violations']

        monotonicity_scores[omega] = {
            'dominance_score': float(np.mean(dom_scores)),
            'dominance_violations': int(dom_violations),
            'adjacent_score': float(np.mean(adj_scores)),
            'adjacent_violations': int(adj_violations),
            'agent_scores': [float(s) for s in dom_scores]
        }

        print(f"  Final Reward: {np.mean(history[-50:]):.2f}")
        print(f"  Avg Monotonicity: {np.mean(dom_scores):.4f}")

    # Plot results
    plot_combined_rewards(all_rewards)
    plot_policy_grids(trained_agents, num_classrooms)
    plot_monotonicity_summary(monotonicity_scores, num_classrooms)

    # Save results
    with open(os.path.join(OUTPUT_DIR, "monotonicity_scores.json"), 'w') as f:
        json.dump({str(k): v for k, v in monotonicity_scores.items()}, f, indent=4)

    print(f"\nTraining complete. Models saved to {MODEL_DIR}")
    
    return trained_agents, monotonicity_scores


# ============================================================
# 8. PLOTTING FUNCTIONS
# ============================================================

def generate_distinct_colors(n):
    HSV = [(x / n, 0.6, 0.95) for x in range(n)]
    RGB = [colorsys.hsv_to_rgb(*x) for x in HSV]
    return ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in RGB]


def plot_combined_rewards(all_rewards):
    """Plot training rewards."""
    print("\n--- Plotting Rewards ---")

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(OMEGA_VALUES)))

    for idx, omega in enumerate(OMEGA_VALUES):
        data = all_rewards[omega]
        mean_rewards = np.mean(data, axis=0)

        window = 100
        kernel = np.ones(window) / window
        smoothed = np.convolve(mean_rewards, kernel, mode='valid')

        plt.plot(smoothed, label=f'ω={omega}', color=colors[idx], linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Centralized PPO Training Rewards (Tanh Policy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_rewards.png"), dpi=300)
    plt.close()


def plot_policy_grids(trained_agents, num_classrooms=NUM_CLASSROOMS):
    """Plot policy grids for all agents across omega values."""
    print("\n--- Plotting Policy Grids ---")

    max_agents_to_plot = min(num_classrooms, 4)

    fig, axes = plt.subplots(max_agents_to_plot, len(OMEGA_VALUES),
                             figsize=(3.5 * len(OMEGA_VALUES), 3 * max_agents_to_plot),
                             sharey=True)

    if max_agents_to_plot == 1:
        axes = axes.reshape(1, -1)

    base_colors = generate_distinct_colors(12)
    cmap = LinearSegmentedColormap.from_list("custom", base_colors, N=256)

    for idx, omega in enumerate(OMEGA_VALUES):
        ppo = trained_agents[omega]
        policy_grids = extract_joint_policy_grid(ppo)

        for agent_idx in range(max_agents_to_plot):
            policy_grid = policy_grids[agent_idx]
            ax = axes[agent_idx, idx]

            im = ax.imshow(
                policy_grid,
                extent=[0, 1, 0, TOTAL_STUDENTS],
                origin='lower',
                aspect='auto',
                cmap=cmap,
                vmin=0.0,
                vmax=1.0
            )

            ax.set_title(f'ω={omega} | Agent {agent_idx}')

            if agent_idx == max_agents_to_plot - 1:
                ax.set_xlabel('Risk')
            if idx == 0:
                ax.set_ylabel('Infected Count')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Action (Capacity Fraction)')

    title = f'Centralized PPO Policies ({num_classrooms} Classrooms)'
    if num_classrooms > max_agents_to_plot:
        title += f' (showing first {max_agents_to_plot})'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "policy_grids.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_monotonicity_summary(monotonicity_scores, num_classrooms=NUM_CLASSROOMS):
    """Plot monotonicity scores."""
    print("\n--- Plotting Monotonicity Summary ---")

    omegas = list(monotonicity_scores.keys())
    max_agents_to_plot = min(num_classrooms, 4)

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(omegas))
    width = 0.8 / max_agents_to_plot

    colors = plt.cm.Set2(np.linspace(0, 1, max_agents_to_plot))

    for agent_idx in range(max_agents_to_plot):
        scores = []
        for o in omegas:
            agent_scores = monotonicity_scores[o]['agent_scores']
            if agent_idx < len(agent_scores):
                scores.append(agent_scores[agent_idx])
            else:
                scores.append(0)

        offset = (agent_idx - max_agents_to_plot / 2 + 0.5) * width
        ax.bar(x + offset, scores, width, label=f'Agent {agent_idx}', color=colors[agent_idx])

    ax.set_xlabel('Omega (ω)')
    ax.set_ylabel('Monotonicity Score')
    ax.set_title(f'Centralized PPO - Policy Monotonicity ({num_classrooms} Classrooms)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{o}' for o in omegas])
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "monotonicity_summary.png"), dpi=300)
    plt.close()


# ============================================================
# 9. MAIN
# ============================================================

def main(mode='tune_and_train', num_classrooms=NUM_CLASSROOMS):
    """
    Main function.
    
    Modes:
    - 'tune': Grid search for best LRs
    - 'train': Train with saved LRs
    - 'tune_and_train': Both
    """
    print(f"\n{'='*60}")
    print(f"Centralized PPO Training (Tanh Policy)")
    print(f"Number of Classrooms: {num_classrooms}")
    print(f"{'='*60}")

    if mode in ['tune', 'tune_and_train']:
        optimized_lrs = grid_search_tuning(num_classrooms)
    else:
        optimized_lrs = load_lrs()

    if mode in ['train', 'tune_and_train']:
        train_and_evaluate_optimal(optimized_lrs, num_classrooms)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main(mode='tune_and_train', num_classrooms=NUM_CLASSROOMS)