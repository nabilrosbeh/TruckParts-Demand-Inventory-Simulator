# Deep Reinforcement Learning for Spare Parts Inventory Control

A bachelor's thesis project comparing four deep reinforcement learning algorithms against the classical Standard Inventory Policy (SIP) for automotive aftermarket spare parts inventory management.

Built on top of the synthetic demand simulator developed by Fukuhara et al. (see [References](#references)).

---

## Overview

The core question this project investigates:

> *Can a deep RL agent learn an ordering policy that matches or outperforms the classical (s, Q) Standard Inventory Policy on total cost and service level?*

A single dealer manages **4 spare part types** independently. Each day the agent observes inventory state across all parts and decides how much of each part to order. The environment runs a 7-step daily simulation (receive deliveries → fill backorders → serve demand → trigger rush orders on stockout → place proactive orders → charge holding cost).

**Four algorithms are compared:**

| Algorithm | Policy type | Policy class |
|-----------|-------------|--------------|
| PPO | On-policy | Stochastic |
| A2C | On-policy | Stochastic |
| SAC | Off-policy | Stochastic |
| TD3 | Off-policy | Deterministic |

**Two experimental conditions:**
- **Masked** — A SIP-triggered interval mask constrains each order to `[0, Q*_t]` when stock falls at or below the dynamic reorder point. When the trigger is inactive the agent is forced to output zero.
- **Unmasked** — Full action space `[0, Q_max]` available at all times.

The SIP is the performance baseline because it is the policy currently used in practice in Volvo Group's Service Market Logistics.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/nabilrosbeh/TruckParts-Demand-Inventory-Simulator.git
cd TruckParts-Demand-Inventory-Simulator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.11. Key dependencies: `stable-baselines3`, `gymnasium`, `pandas`, `numpy`, `matplotlib`.

### 3. Train all four algorithms (masked condition)

```bash
python train_with_mask.py
```

This trains PPO, SAC, A2C, and TD3 sequentially, evaluates them against the SIP on a held-out test period, and saves:
- `results/policy_comparison.csv` — cost and service level metrics for all policies
- `results/figures/` — learning curves and per-part inventory traces

### 4. Train without the action mask (unmasked condition)

```bash
python train_without_mask.py
```

Identical setup with the mask disabled. Comparing results from both scripts isolates the contribution of the SIP-triggered mask to agent performance.

### 5. Interactive exploration

Open `notebooks/rl_inventory_mask.ipynb` for a step-by-step walkthrough of the masked condition: environment setup, training, evaluation plots, and cost decomposition. Use `notebooks/rl_inventory_nomask.ipynb` for the unmasked condition.

---

## Repository Structure

```
├── train_with_mask.py          # Training script — masked condition
├── train_without_mask.py       # Training script — unmasked condition
├── requirements.txt            # Python dependencies
│
├── lib/
│   ├── demand/
│   │   ├── Environment.py      # Gymnasium inventory environment (step, reset, obs)
│   │   ├── dealer.py           # Dealer-level simulation logic
│   │   ├── demand_management.py
│   │   └── ...                 # Demand generator components (Fukuhara et al.)
│   ├── cost/
│   │   ├── simulationLogic.py  # Daily simulation step (7 operations)
│   │   ├── inventoryPolices.py # SIP baseline implementation
│   │   ├── costTracker.py      # Cost accounting (ordering, rush, holding)
│   │   ├── orderManagement.py  # Order pipeline management
│   │   └── ...
│   └── ResultComparison.py     # Aggregation and comparison utilities
│
├── notebooks/
│   ├── rl_inventory_mask.ipynb     # Main RL notebook — masked condition
│   ├── rl_inventory_nomask.ipynb   # RL notebook — unmasked condition
│   └── main.ipynb                  # Original SIP simulator notebook
│
├── data/
│   └── demand/
│       └── demand_series.csv   # Synthetic daily demand data (4 parts, ~2200 days)
│
├── results/
│   ├── policy_comparison.csv   # Final evaluation metrics for all policies
│   ├── rl_vs_sip.png           # Cost comparison: RL agents vs SIP
│   ├── decision_trace_all_policies.png  # Order decisions over test period
│   ├── inventory_results.png   # On-hand stock levels over time
│   ├── Figures_episode_800_with_mask/   # Learning curves at 800 episodes (masked)
│   ├── Figures_episode_800_no_mask/     # Learning curves at 800 episodes (unmasked)
│   ├── Figures_episode_1500_with_mask/  # Learning curves at 1500 episodes (masked)
│   └── Figures_episode_1500_no_mask/   # Learning curves at 1500 episodes (unmasked)
│
└── figures/                    # Thesis figures (demand series, train/test split, daily step diagram)
```

---

## MDP Formulation

| Component | Definition |
|-----------|-----------|
| **State** | 13-dimensional feature vector per part × N parts = 52-dim observation |
| **Action** | Continuous order quantity per part ∈ [0, Q_max], Q_max = 5,000 units |
| **Reward** | Negative total daily cost across all parts |
| **Discount** | γ = 0.995 |
| **Episode length** | 1,752 days (~5 years) training / 439 days test |

**The 13 per-part observation features:**

| Feature | Description |
|---------|-------------|
| `x` | On-hand inventory |
| `b` | Backorders outstanding |
| `z` | Inventory position (on-hand + pipeline − backorders) |
| `u_nu` | Non-urgent units in transit |
| `u_u` | Urgent units in transit |
| `τ` | Days until next non-urgent delivery (15 if none in transit) |
| `d` | Long-run mean daily demand (normalised) |
| `μ_30` | 30-day rolling mean demand |
| `σ_30` | 30-day rolling std of demand |
| `doy` | Fractional day-of-year ∈ [0, 1] |
| `s_t` | Dynamic reorder point (ROP from 30-day window) |
| `Q*_t` | Dynamic EOQ (from 365-day window) |
| `ξ` | SIP trigger signal ∈ {0, 1} |

**Cost structure (per part, per day):**

| Cost component | Value |
|----------------|-------|
| Non-urgent order fixed cost | 150 SEK |
| Rush order fixed cost | 215 SEK |
| Transport cost | 0.002 SEK / unit |
| Holding cost | 0.15 × 0.13 SEK / unit / year |

**Lead times:** 14 days (non-urgent), 2 days (urgent/rush).

---

## SIP-Triggered Action Mask

When the SIP trigger fires (on-hand ≤ dynamic ROP and no non-urgent order in transit), the agent's raw output is capped at the dynamic EOQ:

```
ã_p = clip(round(a_p), 1, max(1, floor(Q*_p)))   if trigger active
ã_p = 0                                            otherwise
```

This narrows the decision from *whether and how much to order* to simply *how much to order within [0, Q*]*. The mask is applied inside `step()` before any inventory update, following the continuous action masking framework of Stolz et al. (2024).

---

## Results

Evaluation metrics are in `results/policy_comparison.csv`. Key columns:

| Column | Description |
|--------|-------------|
| `Total_Cost_SEK` | Cumulative cost over 439-day test period |
| `ISL` | Immediate service level (fraction of demand met from on-hand stock) |
| `vs_SIP_ratio` | Cost relative to SIP (1.0 = SIP performance) |
| `N_Rush_Events` | Number of automatic rush orders triggered |
| `Stockout_Rate` | Fraction of days with a stockout |

---

## References

The synthetic demand simulator and cost framework used in this project are described in:

```bibtex
@article{fukuhara2026,
  author  = {Fukuhara, So and Alabdallah, Abdallah and Gunasekara, Nuwan and Nowaczyk, Slawomir},
  title   = {Bridging Forecast Accuracy and Inventory KPIs: A Simulation-Based Evaluation Framework},
  journal = {arXiv preprint arXiv:2601.21844},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.21844}
}
```

The continuous action masking framework:

```bibtex
@inproceedings{stolz2024,
  author    = {Stolz, Dimitri and Gros, Sebastien and Goodwin, Morten},
  title     = {Continuous Action Masking for Reinforcement Learning},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2024}
}
```

The RL algorithms are implemented via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).
