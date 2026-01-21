# SafeCampus MultiClassroom Project

## Overview
This project implements multi-agent reinforcement learning approaches to optimize classroom policies in a multi-classroom environment during an epidemic scenario. The system uses advanced PPO-based algorithms to learn policies that balance infection control with classroom attendance.

## Project Structure
The project consists of the following main components:

- `environment/`: Contains the `MultiClassroomEnv` class, which simulates the multi-classroom epidemic environment
  - `multiclassroom.py`: Main environment implementation
  - `simulation.py`: Epidemic simulation logic
- `ppo_centralized.py`: Centralized PPO implementation with a single controller for all classrooms
- `ppo_ctde.py`: Multi-Agent PPO with Centralized Training and Decentralized Execution (MAPPO-CTDE)
- `analyze_environment.py`: Comprehensive analysis tool for comparing different policies
- `epidemic_model_analysis.py`: Additional epidemic model analysis utilities

## Installation

1. Create a conda environment (recommended):
   ```bash
   conda create -n safecampus python=3.10
   conda activate safecampus
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Agents

### 1. PPO Centralized Training

The centralized PPO agent uses a single controller that observes all classrooms and outputs actions for all of them.

```bash
python ppo_centralized.py
```

**Features:**
- Single global policy network observing all classrooms
- Tanh-deterministic actor with exploration noise
- Automatic hyperparameter tuning for different omega values
- Model checkpoints saved in `centralized_ppo_results/models/`

**Configuration:**
- Edit constants at the top of `ppo_centralized.py` to customize:
  - `NUM_CLASSROOMS`: Number of classrooms (default: 2)
  - `TOTAL_STUDENTS`: Total student population (default: 100)
  - `OMEGA_VALUES`: Preference weights to train (default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
  - `FULL_EPISODES`: Number of training episodes (default: 3000)

**Output:**
- Training reward curves: `centralized_ppo_results/training_rewards.png`
- Policy visualizations: `centralized_ppo_results/policy_grids.png`
- Monotonicity analysis: `centralized_ppo_results/monotonicity_summary.png`
- Trained models: `centralized_ppo_results/models/`

### 2. PPO CTDE (Centralized Training, Decentralized Execution)

The MAPPO-CTDE agent trains decentralized policies for each classroom using a centralized critic.

```bash
python ppo_ctde.py
```

**Features:**
- Decentralized actors (one per classroom) with centralized critic
- Beta distribution policy for naturally bounded actions
- Option to use Tanh-deterministic policy (change `policy_type='tanh'` in main)
- Independent learning during execution, coordinated during training
- Model checkpoints saved in `mappo_results/models/`

**Configuration:**
- Edit constants at the top of `ppo_ctde.py` to customize:
  - `NUM_CLASSROOMS`: Number of classrooms (default: 2)
  - `POLICY_TYPE`: 'beta' or 'tanh' (set in main function)
  - `OMEGA_VALUES`: Preference weights to train
  - `FULL_EPISODES`: Number of training episodes

**Output:**
- Training reward curves: `mappo_results/combined_mappo_rewards_ci.png`
- Policy visualizations: `mappo_results/combined_mappo_optimal_policies.png`
- Monotonicity analysis: `mappo_results/monotonicity_summary.png`
- Trained models: `mappo_results/models/`

### 3. Environment Analysis

Run comprehensive analysis comparing different policies including DP upper bound, myopic optimal, and trained RL models.

```bash
python analyze_environment.py
```

**Features:**
- Dynamic Programming (Backward Induction) upper bound computation
- Myopic optimal policy (one-step lookahead)
- Evaluation of trained CTDE and Centralized models
- Random baseline comparison

**Configuration:**
- Edit constants at the top of `analyze_environment.py`:
  - `OUTPUT_DIR`: Directory for analysis results
  - `OMEGA_VALUES`: Which omega values to analyze
  - `NUM_EVAL_EPISODES`: Number of evaluation episodes (default: 30)

**Output:**
- Performance comparison plot: `analysis_results/performance_comparison.png`
- Summary table: `analysis_results/analysis_results.csv`
- Detailed results: `analysis_results/analysis_results.json`

## Training Modes

Both PPO implementations support three training modes (set in the `main()` function):

- `mode='tune'`: Run hyperparameter tuning only (finds optimal learning rates)
- `mode='train'`: Train with previously saved optimal learning rates
- `mode='tune_and_train'`: Run tuning followed by full training (recommended for first run)

Example:
```python
if __name__ == '__main__':
    main(mode='tune_and_train', num_classrooms=2)
```

## Understanding Omega Values

The `omega` (or `gamma` in the environment) parameter controls the trade-off between:
- **Infection control**: Reducing infected students (health objective)
- **Classroom attendance**: Maximizing in-person learning (education objective)

Lower omega values prioritize infection control, while higher values prioritize attendance.

## Output Structure

After running the training scripts, you'll find:

```
safecampus_multiclassroom/
├── centralized_ppo_results/
│   ├── models/
│   │   ├── centralized_omega_0.1_run_0.pt
│   │   └── ...
│   ├── training_rewards.png
│   ├── policy_grids.png
│   └── monotonicity_summary.png
├── mappo_results/
│   ├── models/
│   │   ├── mappo_omega_0.1_run_0.pt
│   │   └── ...
│   ├── combined_mappo_rewards_ci.png
│   └── combined_mappo_optimal_policies.png
└── analysis_results/
    ├── performance_comparison.png
    ├── analysis_results.csv
    └── analysis_results.json
```

## Notes

- The environment uses continuous action spaces (capacity fractions from 0 to 1)
- Policies are evaluated for monotonicity: actions should decrease as infections/risk increase
- All training uses cooperative rewards where agents share a common objective
- Evaluation mode uses fixed community risk trajectories from `weekly_risk_sample_b.csv`



