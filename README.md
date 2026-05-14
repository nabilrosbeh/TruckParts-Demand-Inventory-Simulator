# TruckParts Deep RL Inventory Controller

## 🔍 Overview
This repository provides the implementation and supplementary materials for the bachelor's thesis *"Can Reinforcement Learning Match the Standard Inventory Policy for Spare Parts?"* at Halmstad University.

The system extends the demand simulator by [Fukuhara et al. (2026)](https://arxiv.org/abs/2601.21844) with a reinforcement learning layer that replaces the classical ordering rule. It consists of two major components:

1. **Demand Generator** — generates synthetic daily demand time-series data for truck spare parts under a dealer–truck–part hierarchy.
2. **RL Inventory Controller** — trains four deep RL algorithms (PPO, SAC, A2C, TD3) directly on the inventory environment and evaluates them against the classical Standard Inventory Policy (SIP) on total cost and service level.

The system produces:
- Per-policy evaluation metrics: total cost, annualised cost, immediate service level (ISL), rush events, stockout rate
- Cost comparison charts across all policies
- Learning curves for each algorithm across training episodes
- Per-part inventory traces over the test period

## 🚀 Getting Started
### 1. Clone the repository
```
git clone https://github.com/nabilrosbeh/TruckParts-Demand-Inventory-Simulator.git
```

### 2. Install dependencies
```
pip install -r requirements.txt
```
(Recommended Python version: Python 3.11)

## 🚀 Running the Workflow
Open and execute `notebooks/rl_inventory_mask.ipynb` for the main experiment (with action mask), or `notebooks/rl_inventory_nomask.ipynb` for the comparison condition. The workflow is structured into two phases:

---

### Phase 1: Demand Generator
The demand generator produces daily demand time-series data based on the hierarchical structure of dealers, trucks, and parts. Parameterisation includes start time, end time, time step, random seed, number of dealers, truck fleet size range, and number of part types per truck.

```python
from datetime import datetime

start_time    = datetime(2025, 1, 1)
end_time      = datetime(2031, 1, 1)
delta_time    = 1
seed          = 42

n_dealers     = 1
n_truck_range = [150, 200]
n_part_range  = [4, 5]

cfg = SimulationConfig(
    start_time = start_time,
    end_time   = end_time,
    delta_time = delta_time
)
sim = Simulator(
    config        = cfg,
    seed          = seed,
    n_dealers     = n_dealers,
    n_truck_range = n_truck_range,
    n_part_range  = n_part_range
)
events = sim.run()
```

The generated demand is saved to `data/demand/demand_series.csv`. If the file already exists, the cell skips generation automatically — delete the file and re-run to regenerate with different parameters.

---

### Phase 2: RL Training & Inventory Control
The RL layer replaces both the forecasting step and the ordering policy from the original simulator. The agent observes the full inventory state (stock levels, backorders, orders in transit, rolling demand statistics, reorder point, EOQ) and learns directly which quantity to order each day. An action mask enforces that orders are only placed when the SIP trigger fires and caps the order at the dynamic EOQ — this prevents the agent from converging to a "never order" strategy.

All four algorithms share the same hyperparameters, defined once at the top of the training section:

```python
LR         = 3e-4    # learning rate
GAMMA      = 0.995   # discount factor
BATCH_SIZE = 256
NET_ARCH   = [256, 256]
```

**PPO — Proximal Policy Optimisation**

On-policy, stochastic. Updates directly from freshly collected experience and clips large gradient steps to stay stable.

```python
PPO_EPISODES = 2000

ppo_model = PPO(
    'MlpPolicy', env_ppo, verbose=0,
    learning_rate = LR,
    gamma         = GAMMA,
    batch_size    = BATCH_SIZE,
    policy_kwargs = dict(net_arch=NET_ARCH),
    n_steps       = 30,
    gae_lambda    = 0.97,
    ent_coef      = 0.05,
    n_epochs      = 4,
    seed          = 42,
)
ppo_model.learn(total_timesteps=T_train * PPO_EPISODES)
```

**SAC — Soft Actor-Critic**

Off-policy, stochastic. Stores past experience in a replay buffer and adds an entropy bonus to encourage exploration.

```python
SAC_EPISODES = 2000

sac_model = SAC(
    'MlpPolicy', env_sac, verbose=0,
    learning_rate   = LR,
    gamma           = GAMMA,
    batch_size      = BATCH_SIZE,
    policy_kwargs   = dict(net_arch=NET_ARCH),
    buffer_size     = 500_000,
    tau             = 0.005,
    gradient_steps  = 128,
    ent_coef        = 'auto',
    seed            = 42,
)
sac_model.learn(total_timesteps=T_train * SAC_EPISODES)
```

**A2C — Advantage Actor-Critic**

On-policy, stochastic. Lighter and faster than PPO; updates more frequently but with higher variance.

```python
A2C_EPISODES = 2000

a2c_model = A2C(
    'MlpPolicy', env_a2c, verbose=0,
    learning_rate = LR,
    gamma         = GAMMA,
    policy_kwargs = dict(net_arch=NET_ARCH),
    n_steps       = 30,
    gae_lambda    = 0.97,
    ent_coef      = 0.05,
    seed          = 42,
)
a2c_model.learn(total_timesteps=T_train * A2C_EPISODES)
```

**TD3 — Twin Delayed DDPG**

Off-policy, deterministic. Trains two critics and takes the minimum to avoid overestimating value; delays actor updates to let the critics stabilise first.

```python
TD3_EPISODES = 2000

td3_model = TD3(
    'MlpPolicy', env_td3, verbose=0,
    learning_rate  = LR,
    gamma          = GAMMA,
    batch_size     = BATCH_SIZE,
    policy_kwargs  = dict(net_arch=NET_ARCH),
    buffer_size    = 500_000,
    tau            = 0.005,
    gradient_steps = 128,
    action_noise   = NormalActionNoise(np.zeros(n_act), 20.0 * np.ones(n_act)),
    policy_delay   = 2,
    seed           = 42,
)
td3_model.learn(total_timesteps=T_train * TD3_EPISODES)
```

---

### Output & Comparison
All trained models and the SIP baseline are evaluated on the same held-out test period (last 20% of the data). Results are saved to `results/policy_comparison.csv` and visualised automatically.

```python
results = {}
results['SIP'] = run_episode(make_test_env(), baseline_action)

for name, model in trained_models.items():
    results[name] = run_episode(make_test_env(), make_action_fn(model))

metrics = {name: aggregate(rows) for name, rows in results.items()}
```

## 📁 Repository Structure
```
├── train_with_mask.py           # Standalone training script — mask ON
├── train_without_mask.py        # Standalone training script — mask OFF
├── requirements.txt
│
├── lib/
│   └── cost/
│       ├── simulationLogic.py
│       ├── inventoryPolices.py
│       ├── costTracker.py
│       ├── orderManagement.py
│       ├── stateManagement.py
│       ├── eventManagement.py
│       ├── timeManagement.py
│       └── plotMetrics.py
│   └── demand/
│       ├── Environment.py
│       ├── dealer.py
│       ├── truck.py
│       ├── demand_management.py
│       ├── Noise_model.py
│       ├── Parameter.py
│       └── ...
│   └── ResultComparison.py
│
├── notebooks/
│   ├── rl_inventory_mask.ipynb      # Main experiment — mask ON
│   ├── rl_inventory_nomask.ipynb    # Comparison — mask OFF
│   └── main.ipynb                   # Original SIP simulator (Fukuhara et al.)
│
├── data/
│   └── demand/                      # Generated demand datasets
│
└── results/
    ├── policy_comparison.csv
    ├── Figures_episode_800_with_mask/
    ├── Figures_episode_800_no_mask/
    ├── Figures_episode_1500_with_mask/
    └── Figures_episode_1500_no_mask/
```

## 📝 References
The demand simulator used in Phase 1 is described in:

```bibtex
@article{IDA2026,
  author  = {Fukuhara, So and Alabdallah, Abdallah and Gunasekara, Nuwan and Nowaczyk, Slawomir},
  title   = {Bridging Forecast Accuracy and Inventory KPIs: A Simulation-Based Evaluation Framework},
  journal = {arXiv preprint arXiv:2601.21844},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.21844}
}
```

The continuous action masking framework used in Phase 2:

```bibtex
@inproceedings{stolz2024,
  author    = {Stolz, Dimitri and Gros, Sebastien and Goodwin, Morten},
  title     = {Continuous Action Masking for Reinforcement Learning},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2024}
}
```
