import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium.spaces import Discrete, Box
from environment.simulation import simulate_infections_n_classrooms
import itertools
import random
import os
import copy


class MultiClassroomEnv(ParallelEnv):
    def __init__(self, num_classrooms=1, total_students=100, max_weeks=15,
                 action_levels_per_class=None, continuous_action=False,
                 alpha=0.008, beta=0.01, phi=0.2, gamma=0.5, seed=None,
                 community_risk_data_file=None, eval_mode=False,
                 cooperative_reward=True):
        """
        Parameters:
          num_classrooms (int): Number of classrooms/agents
          total_students (int): Total students per classroom
          max_weeks (int): Episode length
          action_levels_per_class (list): Number of discrete action levels per classroom
          continuous_action (bool): If True, actions are continuous [0, total_students]
          alpha (float): Infection rate parameter
          beta (float): Recovery rate parameter
          phi (float): Cross-classroom transmission rate
          gamma (float): Balance weight between allowed students and infections (omega)
          seed (int): Random seed
          community_risk_data_file (str): Path to risk data CSV (optional)
          eval_mode (bool): If True, use deterministic risk pattern
          cooperative_reward (bool): If True, all agents receive the average reward
        """
        self.num_classrooms = num_classrooms
        self.total_students = total_students
        self.max_weeks = max_weeks
        self.phi = phi
        self.gamma = gamma
        self.current_week = 0
        self.continuous_action = continuous_action
        self.eval_mode = eval_mode
        self.cooperative_reward = cooperative_reward

        # Parameter setup (alpha, beta, etc.)
        self.alpha_m = [alpha] * num_classrooms
        self.beta = [beta] * num_classrooms

        self.agents = [f"classroom_{i}" for i in range(self.num_classrooms)]
        self.possible_agents = self.agents[:]

        # Spaces
        if self.continuous_action:
            self.action_spaces = {
                agent: Box(low=0.0, high=float(self.total_students), shape=(1,), dtype=np.float32)
                for agent in self.agents
            }
        else:
            # Discrete actions
            if action_levels_per_class is None:
                action_levels_per_class = [11] * num_classrooms
            self.action_levels = action_levels_per_class
            self.action_spaces = {
                agent: Discrete(self.action_levels[i])
                for i, agent in enumerate(self.agents)
            }
            # Precompute discrete action values for each agent
            # Maps action index to actual allowed students value
            self.discrete_action_values = {
                agent: np.linspace(0, self.total_students, self.action_levels[i])
                for i, agent in enumerate(self.agents)
            }

        # Observation Space: [Current Infected, Community Risk]
        self.observation_spaces = {
            agent: Box(low=0.0, high=float(self.total_students), shape=(2,), dtype=np.float32)
            for agent in self.agents
        }

        # State initialization
        self.student_status = [0] * self.num_classrooms
        self.allowed_students = [0] * self.num_classrooms

        # Risk Data Loading
        self.community_risk_data = []
        if community_risk_data_file and os.path.exists(community_risk_data_file):
            # Load CSV logic here if needed, for now we simulate risk
            pass

        if seed is not None:
            self.seed(seed)

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def _generate_shared_episode_risk(self):
        # Generate a random risk curve for the episode
        length = self.max_weeks + 5
        base_risk = np.random.uniform(0.1, 0.5)
        trend = np.linspace(0, np.random.uniform(-0.2, 0.4), length)
        noise = np.random.normal(0, 0.05, length)
        risk = np.clip(base_risk + trend + noise, 0.0, 1.0)
        return risk

    def set_mode(self, eval_mode):
        self.eval_mode = eval_mode

    def _get_observations(self):
        obs = {}
        curr_risk = 0.0
        if self.eval_mode:
            # Fixed pattern for evaluation
            curr_risk = min(1.0, self.current_week / 20.0)
        else:
            if hasattr(self, 'shared_community_risk'):
                curr_risk = self.shared_community_risk[min(self.current_week, len(self.shared_community_risk) - 1)]
            else:
                curr_risk = np.random.random()

        for i, agent in enumerate(self.agents):
            obs[agent] = np.array([float(self.student_status[i]), float(curr_risk)], dtype=np.float32)
        return obs

    def step(self, actions):
        # Parse actions
        if self.continuous_action:
            for i, agent in enumerate(self.agents):
                act = actions[agent]
                # If array, extract val
                if isinstance(act, (np.ndarray, list)):
                    act = act[0]
                self.allowed_students[i] = float(act)
        else:
            # Discrete action mapping: action index -> allowed students value
            for i, agent in enumerate(self.agents):
                action_idx = actions[agent]
                # Handle if action is array
                if isinstance(action_idx, (np.ndarray, list)):
                    action_idx = int(action_idx[0])
                else:
                    action_idx = int(action_idx)
                # Map to actual value
                self.allowed_students[i] = self.discrete_action_values[agent][action_idx]

        # Get current risk
        risk = 0.0
        if self.eval_mode:
            risk = min(1.0, self.current_week / 20.0)
        else:
            risk = self.shared_community_risk[min(self.current_week, len(self.shared_community_risk) - 1)]

        # Simulate Dynamics
        self.student_status = simulate_infections_n_classrooms(
            self.num_classrooms,
            self.alpha_m,
            self.beta,
            self.phi,
            self.student_status,
            self.allowed_students,
            [risk] * self.num_classrooms
        )

        # --- REWARD CALCULATION ---
        individual_rewards = []
        for i, agent in enumerate(self.agents):
            allowed = self.allowed_students[i]
            infected = self.student_status[i]

            # Base individual reward
            r_i = self.gamma * allowed - (1 - self.gamma) * infected
            individual_rewards.append(r_i)

        rewards = {}
        if self.cooperative_reward:
            # Cooperative: Everyone gets the Mean Reward
            avg_reward = np.mean(individual_rewards)
            for agent in self.agents:
                rewards[agent] = avg_reward
        else:
            # Competitive/Individual
            for i, agent in enumerate(self.agents):
                rewards[agent] = individual_rewards[i]

        self.current_week += 1
        dones = {agent: self.current_week >= self.max_weeks for agent in self.agents}

        return self._get_observations(), rewards, dones, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.student_status = [np.random.randint(0, 5) for _ in range(self.num_classrooms)]
        self.allowed_students = [0] * self.num_classrooms
        self.current_week = 0

        if not self.eval_mode:
            self.shared_community_risk = self._generate_shared_episode_risk()

        return self._get_observations()

    def render(self):
        pass

    def clone(self):
        """
        Creates a deep copy of the environment in its current state.
        Faster than creating a new environment from scratch.
        """
        return copy.deepcopy(self)