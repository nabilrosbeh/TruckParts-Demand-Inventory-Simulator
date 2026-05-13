#!/usr/bin/env python3
"""
train_with_mask.py
==================
Trains PPO, SAC, A2C, and TD3 on the spare parts inventory problem
with the action mask turned on.

How the mask works:
  - If stock is above the reorder point, or there is already an order
    on its way, the agent is forced to order nothing (output = 0).
  - If stock has dropped to or below the reorder point and no order is
    in transit, the agent can order anywhere between 1 and the EOQ.

This keeps the agent from discovering that doing nothing is always
"safe" in the short term, which is a common failure mode without
any constraints on the action.

Run from the project root:
    python train_with_mask.py

Results are saved to results/with_mask_results.csv.
"""

import sys, os, time, math, warnings
import importlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='stable_baselines3')

# Add the library folders to the path so we can import from them
sys.path.append(os.path.abspath('lib/demand'))
sys.path.append(os.path.abspath('lib/cost'))

import Environment as env_mod
importlib.reload(env_mod)
SimulationConfig = env_mod.SimulationConfig
Simulator        = env_mod.Simulator
from inventoryPolices import StandardInventoryPolicy, InventoryPolicyParams

os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'stable-baselines3'])
from stable_baselines3 import PPO, SAC, A2C, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise


# =============================================================================
# Load demand data
# =============================================================================
def load_all_parts_for_dealer(dealer_id: str,
                               csv_path: str = 'data/demand/demand_series.csv'):
    df  = pd.read_csv(csv_path)
    ddf = df[df['dealer_id'] == dealer_id].copy()
    if ddf.empty:
        raise ValueError(f"dealer_id '{dealer_id}' not found in {csv_path}.")

    part_types = sorted(ddf['part_type'].unique().tolist())
    start_date = pd.to_datetime(ddf.sort_values('time')['date'].iloc[0]).to_pydatetime()

    demand_dict = {}
    for pt in part_types:
        arr = (
            ddf[ddf['part_type'] == pt]
            .sort_values('time')['failure']
            .values.astype(int)
        )
        demand_dict[pt] = arr

    print(f"Dealer: {dealer_id} | Start: {start_date.date()} | Parts: {part_types}")
    for pt in part_types:
        arr = demand_dict[pt]
        print(f"  {pt}: {len(arr)} days | total demand={arr.sum()} | avg/day={arr.mean():.4f}")

    return demand_dict, start_date, part_types


# =============================================================================
# Inventory environment — action mask ON
# =============================================================================
class MultiPartInventoryEnv(gym.Env):
    """
    Inventory environment for a single dealer managing multiple part types.
    The action mask is active in this version.

    Each day the agent decides how much of each part to order. The mask
    steps in before the order is placed: if the stock is still above the
    reorder point, or there is already an order coming, the order is set
    to zero regardless of what the agent outputs. If the reorder point
    has been crossed and no order is on its way, the agent's output is
    accepted but capped at the EOQ.

    Observation: 13 features per part (52 total for 4 parts)
    Action:      continuous order quantity per part, clipped by the mask
    Reward:      negative total daily cost across all parts
    """

    metadata = {'render_modes': ['human']}

    HOLDING_RATE   = 0.15 * 0.13 / 365.0
    ORDER_COST     = 100.0
    RUSH_COST      = 165.0
    BADWILL_PROXY  = 50.0
    TRANSPORT_RATE = 0.002

    def __init__(self,
                 demand_dict,
                 start_date,
                 part_types,
                 lead_time=14,
                 urgent_lead=2,
                 initial_stock=120,
                 max_order=200,
                 demand_history_window=30):
        super().__init__()
        self.demand_dict = {pt: np.asarray(v, dtype=np.float32) for pt, v in demand_dict.items()}
        self.start_date  = start_date
        self.part_types  = list(part_types)
        self.n_parts     = len(self.part_types)
        self.lead_time   = int(lead_time)
        self.urgent_lead = int(urgent_lead)
        self.initial_stock = float(initial_stock)
        self.max_order   = int(max_order)
        self.max_stock   = float(max_order * 3)
        self.T           = len(next(iter(self.demand_dict.values())))
        self.demand_history_window = int(demand_history_window)

        self._HOLDING_COST_ANNUAL = 0.15 * 0.13
        self._Z95 = 1.6449

        self._avg  = {pt: float(self.demand_dict[pt].mean()) for pt in self.part_types}
        max_avg    = max(max(self._avg.values()), 1.0)
        self._rate = {pt: self._avg[pt] / max_avg for pt in self.part_types}

        self.obs_per_part = 13
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_per_part * self.n_parts,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=float(self.max_order),
            shape=(self.n_parts,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day         = 0
        self.stock       = {pt: float(self.initial_stock) for pt in self.part_types}
        self.backorders  = {pt: 0.0 for pt in self.part_types}
        self.nu_pipe     = {pt: [] for pt in self.part_types}
        self.urg_pipe    = {pt: [] for pt in self.part_types}
        self.demand_hist = {pt: [] for pt in self.part_types}
        return self._obs(), {}

    def _deliver_due(self, pipeline, pt, day):
        due          = [item for item in pipeline[pt] if item[0] <= day]
        pipeline[pt] = [item for item in pipeline[pt] if item[0] > day]
        return float(sum(qty for _, qty in due))

    def _forecast_sum(self, pt, start_day, horizon):
        if horizon <= 0:
            return 0.0
        end_day = min(start_day + horizon, self.T)
        if start_day >= end_day:
            return 0.0
        return float(self.demand_dict[pt][start_day:end_day].sum())

    def _compute_dyn_rop_eoq(self, pt):
        """Work out today's reorder point and order quantity from recent demand."""
        recent  = self.demand_hist[pt][-self.demand_history_window:]
        mu_30   = float(np.mean(recent)) if recent else self._avg[pt]
        std_30  = float(np.std(recent))  if len(recent) > 1 else 0.0
        dyn_rop = mu_30 * self.lead_time + self._Z95 * std_30 * math.sqrt(float(self.lead_time))

        lw_365  = self.demand_hist[pt][-365:]
        mu_365  = float(np.mean(lw_365)) if lw_365 else self._avg[pt]
        D_ann   = mu_365 * 365.0
        dyn_eoq = (math.sqrt(2.0 * D_ann * self.ORDER_COST / self._HOLDING_COST_ANNUAL)
                   if D_ann > 0 else 0.0)
        dyn_eoq = min(dyn_eoq, float(self.max_order))
        return dyn_rop, dyn_eoq

    def step(self, action):
        if self.day >= self.T:
            raise RuntimeError('Episode finished. Call reset() before calling step() again.')

        action = np.asarray(action, dtype=np.float32)

        # Step 1: apply the action mask before anything else happens.
        # If the reorder point hasn't been crossed yet, or an order is already
        # on its way, set the order to zero. Otherwise let the agent order up
        # to the EOQ.
        order_qtys = {}
        for i, pt in enumerate(self.part_types):
            raw_qty = int(np.clip(np.round(float(action[i])), 0, self.max_order))
            dyn_rop, dyn_eoq = self._compute_dyn_rop_eoq(pt)
            sip_fires = (self.stock[pt] <= dyn_rop and len(self.nu_pipe[pt]) == 0)
            if not sip_fires:
                order_qtys[pt] = 0                      # not time to order yet
            else:
                eoq_cap        = max(int(dyn_eoq), 1)   # at least 1 unit
                order_qtys[pt] = min(raw_qty, eoq_cap)  # cap at the EOQ

        total_cost = 0.0
        info       = {'day': self.day}

        for i, pt in enumerate(self.part_types):
            d = self.day

            # Step 2: receive any deliveries that are due today
            self.stock[pt] += self._deliver_due(self.nu_pipe, pt, d)
            self.stock[pt] += self._deliver_due(self.urg_pipe, pt, d)

            # Step 3: use available stock to clear any outstanding backorders
            if self.backorders[pt] > 0:
                fulfilled = min(self.stock[pt], self.backorders[pt])
                self.stock[pt]      -= fulfilled
                self.backorders[pt] -= fulfilled

            # Step 4: serve today's demand, add any shortfall to backorders
            demand = float(self.demand_dict[pt][d])
            self.demand_hist[pt].append(demand)

            fulfilled_now = min(self.stock[pt], demand)
            self.stock[pt] -= fulfilled_now
            shortage        = demand - fulfilled_now
            if shortage > 0:
                self.backorders[pt] += shortage

            # Step 5: if we ran out of stock, place an urgent order automatically
            rush_cost  = 0.0
            urgent_qty = 0.0
            if shortage > 0:
                next_nonurgent_day = min(
                    [arr_day for arr_day, _ in self.nu_pipe[pt]],
                    default=d + self.lead_time,
                )
                days_until_arrival = max(0, next_nonurgent_day - d)
                expected_until_arr = self._forecast_sum(pt, d + 1, days_until_arrival)
                urgent_qty         = shortage + expected_until_arr
                if urgent_qty > 0:
                    self.urg_pipe[pt].append((d + self.urgent_lead, float(urgent_qty)))
                    rush_cost = self.RUSH_COST + self.BADWILL_PROXY + self.TRANSPORT_RATE * urgent_qty

            # Step 6: place the agent's regular (non-urgent) order if the mask allows it
            order_cost   = 0.0
            order_placed = False
            order_qty    = 0.0
            qty          = order_qtys[pt]

            has_nonurgent_in_transit = len(self.nu_pipe[pt]) > 0

            if qty > 0 and not has_nonurgent_in_transit:
                on_order_all = (sum(q for _, q in self.nu_pipe[pt])
                                + sum(q for _, q in self.urg_pipe[pt]))
                inv_pos_now  = self.stock[pt] + on_order_all - self.backorders[pt]
                qty          = min(qty, max(0, int(self.max_stock - inv_pos_now)))

            if qty > 0 and not has_nonurgent_in_transit:
                self.nu_pipe[pt].append((d + self.lead_time, float(qty)))
                order_placed = True
                order_qty    = float(qty)
                order_cost   = self.ORDER_COST + self.BADWILL_PROXY + self.TRANSPORT_RATE * order_qty

            # Step 7: charge end-of-day holding cost on whatever stock remains
            holding  = self.HOLDING_RATE * self.stock[pt]
            day_cost = rush_cost + order_cost + holding
            total_cost += day_cost

            on_order_nu = float(sum(q for _, q in self.nu_pipe[pt]))
            on_order_u  = float(sum(q for _, q in self.urg_pipe[pt]))

            info[pt] = {
                'stock':              float(self.stock[pt]),
                'backorders':         float(self.backorders[pt]),
                'on_order':           float(on_order_nu + on_order_u),
                'on_order_nonurgent': float(on_order_nu),
                'on_order_urgent':    float(on_order_u),
                'demand':             float(demand),
                'shortage':           float(shortage),
                'urgent_qty':         float(urgent_qty),
                'order_placed':       bool(order_placed),
                'order_qty':          float(order_qty),
                'day_cost':           float(day_cost),
                'ordering':           float(order_cost),
                'rush':               float(rush_cost),
                'holding':            float(holding),
            }

        self.day += 1
        terminated         = self.day >= self.T
        info['total_day_cost'] = float(total_cost)
        return self._obs(), -float(total_cost), terminated, False, info

    def _obs(self):
        blocks = []
        for pt in self.part_types:
            stock       = float(self.stock[pt])
            backorders  = float(self.backorders[pt])
            on_order_nu = float(sum(q for _, q in self.nu_pipe[pt]))
            on_order_u  = float(sum(q for _, q in self.urg_pipe[pt]))
            inv_pos = stock + on_order_nu + on_order_u - backorders

            arrivals        = [day for day, _ in self.nu_pipe[pt] + self.urg_pipe[pt]]
            days_to_arrival = float(max(0, min(arrivals) - self.day)) if arrivals else 0.0

            recent      = self.demand_hist[pt][-self.demand_history_window:]
            recent_mean = float(np.mean(recent)) if recent else 0.0
            recent_std  = float(np.std(recent))  if len(recent) > 1 else 0.0

            from datetime import timedelta as _td
            cur_date = self.start_date + _td(days=self.day)
            doy      = cur_date.timetuple().tm_yday / 366.0

            dyn_rop, dyn_eoq = self._compute_dyn_rop_eoq(pt)
            dyn_sip = 1.0 if (stock <= dyn_rop and on_order_nu < 1.0) else 0.0

            blocks.append(np.array([
                stock        / self.max_stock,                          # 0  on-hand stock
                backorders   / self.max_stock,                          # 1  backorders
                inv_pos      / self.max_stock,                          # 2  inventory position
                on_order_nu  / self.max_stock,                          # 3  regular order in transit
                on_order_u   / self.max_stock,                          # 4  urgent order in transit
                days_to_arrival / float(max(self.lead_time, 1)),        # 5  days until next delivery
                self._rate[pt],                                         # 6  relative demand rate
                recent_mean / max(self._avg[pt] * 2.0, 1.0),           # 7  30-day average demand
                recent_std  / max(self._avg[pt] * 2.0, 1.0),           # 8  30-day demand variability
                doy,                                                    # 9  day of year
                dyn_rop / self.max_stock,                               # 10 reorder point
                dyn_eoq / float(self.max_order),                        # 11 suggested order quantity
                dyn_sip,                                                # 12 mask active (1) or not (0)
            ], dtype=np.float32))

        return np.concatenate(blocks).astype(np.float32)

    def render(self):
        print(f'Day {self.day}')
        for pt in self.part_types:
            print(
                f"  {pt}: stock={self.stock[pt]:.1f}  backorders={self.backorders[pt]:.1f}"
                f"  nu_pipe={self.nu_pipe[pt]}  urg_pipe={self.urg_pipe[pt]}"
            )


# =============================================================================
# Training progress callback — prints cost every N episodes
# =============================================================================
class EpisodeProgressCallback(BaseCallback):
    def __init__(self, total_episodes, algo_name='ALGO', print_every=200):
        super().__init__()
        self.total_episodes  = total_episodes
        self.algo_name       = algo_name
        self.print_every     = print_every
        self.ep_count        = 0
        self._recent_rewards = []
        self._t0             = None

    def _on_training_start(self):
        self._t0 = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.ep_count += 1
                self._recent_rewards.append(info['episode']['r'])
                if self.ep_count % self.print_every == 0:
                    elapsed = time.time() - self._t0
                    eta     = elapsed / self.ep_count * (self.total_episodes - self.ep_count)
                    window  = self._recent_rewards[-self.print_every:]
                    mean_r  = sum(window) / len(window)
                    print(f'  {self.algo_name}  ep {self.ep_count:>5}/{self.total_episodes}'
                          f'  mean_cost {-mean_r:>12,.0f} SEK'
                          f'  elapsed {elapsed/60:.1f}m  ETA {eta/60:.1f}m')
        return True


# =============================================================================
# Load data and split into train / test
# =============================================================================
print('\n' + '='*70)
print('  TRAINING WITH SIP ACTION MASK')
print('='*70)

demand_dict, start_date, part_types = load_all_parts_for_dealer('D00')

T_total = len(demand_dict[part_types[0]])
SPLIT   = int(T_total * 0.80)

train_demand_dict = {pt: arr[:SPLIT] for pt, arr in demand_dict.items()}
test_demand_dict  = {pt: arr[SPLIT:] for pt, arr in demand_dict.items()}
train_start_date  = start_date
test_start_date   = start_date + timedelta(days=SPLIT)
T_train           = SPLIT
T_test            = T_total - SPLIT

print(f'\nTrain: {T_train} days  |  Test: {T_test} days')
print(f'Observation size: {13 * len(part_types)} features  |  SIP mask: ON')


# =============================================================================
# Shared hyperparameters
# =============================================================================
LR         = 3e-4
GAMMA      = 0.995
BATCH_SIZE = 256
NET_ARCH   = [256, 256]
DEVICE     = 'cpu'

# On-policy settings (PPO, A2C)
N_STEPS    = 30
GAE_LAMBDA = 0.97
ENT_COEF   = 0.05
MAX_GRAD   = 0.5

# Off-policy settings (SAC, TD3)
BUFFER     = 500_000
TAU        = 0.005
TRAIN_FREQ = T_train
GRAD_STEPS = 128
LRN_START  = T_train


def make_env():
    return Monitor(MultiPartInventoryEnv(
        demand_dict  = train_demand_dict,
        start_date   = train_start_date,
        part_types   = part_types,
        lead_time    = 14,
        urgent_lead  = 2,
        initial_stock= 120,
        max_order    = 200,
    ))


# =============================================================================
# Train each algorithm — adjust episode counts here if needed
# =============================================================================
trained_models = {}

# PPO
PPO_EPISODES = 2000
print(f'\n--- PPO (with mask) — {PPO_EPISODES} episodes ---')
env_ppo = make_env()
ppo_model = PPO(
    'MlpPolicy', env_ppo, verbose=0,
    learning_rate=LR, gamma=GAMMA, batch_size=BATCH_SIZE,
    policy_kwargs=dict(net_arch=NET_ARCH), device=DEVICE,
    n_steps=N_STEPS, gae_lambda=GAE_LAMBDA, ent_coef=ENT_COEF,
    max_grad_norm=MAX_GRAD, n_epochs=4, seed=42, clip_range=0.2,
)
t0 = time.time()
ppo_model.learn(
    total_timesteps = T_train * PPO_EPISODES,
    callback        = EpisodeProgressCallback(PPO_EPISODES, 'PPO'),
)
print(f'PPO done in {(time.time()-t0)/60:.1f} min')
ep_r_ppo = env_ppo.get_episode_rewards()
print(f'Last-20-ep mean cost: {-np.mean(ep_r_ppo[-20:]):,.0f} SEK')
trained_models['PPO'] = (ppo_model, PPO_EPISODES)

# SAC
SAC_EPISODES = 2000
print(f'\n--- SAC (with mask) — {SAC_EPISODES} episodes ---')
env_sac = make_env()
sac_model = SAC(
    'MlpPolicy', env_sac, verbose=0,
    learning_rate=LR, gamma=GAMMA, batch_size=BATCH_SIZE,
    policy_kwargs=dict(net_arch=NET_ARCH), device=DEVICE,
    buffer_size=BUFFER, tau=TAU, train_freq=TRAIN_FREQ,
    gradient_steps=GRAD_STEPS, learning_starts=LRN_START,
    seed=42, ent_coef='auto',
)
t0 = time.time()
sac_model.learn(
    total_timesteps = T_train * SAC_EPISODES,
    callback        = EpisodeProgressCallback(SAC_EPISODES, 'SAC'),
)
print(f'SAC done in {(time.time()-t0)/60:.1f} min')
ep_r_sac = env_sac.get_episode_rewards()
print(f'Last-20-ep mean cost: {-np.mean(ep_r_sac[-20:]):,.0f} SEK')
trained_models['SAC'] = (sac_model, SAC_EPISODES)

# A2C
A2C_EPISODES = 2000
print(f'\n--- A2C (with mask) — {A2C_EPISODES} episodes ---')
env_a2c = make_env()
a2c_model = A2C(
    'MlpPolicy', env_a2c, verbose=0,
    learning_rate=LR, gamma=GAMMA,
    policy_kwargs=dict(net_arch=NET_ARCH), device=DEVICE,
    n_steps=N_STEPS, gae_lambda=GAE_LAMBDA, ent_coef=ENT_COEF,
    max_grad_norm=MAX_GRAD, seed=42, vf_coef=0.5,
)
t0 = time.time()
a2c_model.learn(
    total_timesteps = T_train * A2C_EPISODES,
    callback        = EpisodeProgressCallback(A2C_EPISODES, 'A2C'),
)
print(f'A2C done in {(time.time()-t0)/60:.1f} min')
ep_r_a2c = env_a2c.get_episode_rewards()
print(f'Last-20-ep mean cost: {-np.mean(ep_r_a2c[-20:]):,.0f} SEK')
trained_models['A2C'] = (a2c_model, A2C_EPISODES)

# TD3
TD3_EPISODES = 2000
print(f'\n--- TD3 (with mask) — {TD3_EPISODES} episodes ---')
env_td3 = make_env()
n_act   = env_td3.action_space.shape[0]
td3_model = TD3(
    'MlpPolicy', env_td3, verbose=0,
    learning_rate=LR, gamma=GAMMA, batch_size=BATCH_SIZE,
    policy_kwargs=dict(net_arch=NET_ARCH), device=DEVICE,
    buffer_size=BUFFER, tau=TAU, train_freq=TRAIN_FREQ,
    gradient_steps=GRAD_STEPS, learning_starts=LRN_START,
    action_noise=NormalActionNoise(np.zeros(n_act), 20.0*np.ones(n_act)),
    seed=42, policy_delay=2,
)
t0 = time.time()
td3_model.learn(
    total_timesteps = T_train * TD3_EPISODES,
    callback        = EpisodeProgressCallback(TD3_EPISODES, 'TD3'),
)
print(f'TD3 done in {(time.time()-t0)/60:.1f} min')
ep_r_td3 = env_td3.get_episode_rewards()
print(f'Last-20-ep mean cost: {-np.mean(ep_r_td3[-20:]):,.0f} SEK')
trained_models['TD3'] = (td3_model, TD3_EPISODES)


# =============================================================================
# Evaluate all policies on the held-out test period
# =============================================================================
def make_test_env():
    return MultiPartInventoryEnv(
        demand_dict  = test_demand_dict,
        start_date   = test_start_date,
        part_types   = part_types,
        lead_time    = 14,
        urgent_lead  = 2,
        initial_stock= 120,
        max_order    = 200,
    )

def run_episode(env, action_fn):
    obs, _ = env.reset()
    rows, done = [], False
    while not done:
        action = np.asarray(action_fn(obs, env), dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        info['action'] = action.tolist()
        info['reward'] = float(reward)
        rows.append(info)
        done = term or trunc
    return rows

def baseline_action(obs, env):
    # Standard Inventory Policy: order the EOQ whenever stock hits the reorder point
    action    = np.zeros(env.n_parts, dtype=np.float32)
    max_stock = float(env.max_stock)
    max_order = float(env.max_order)
    for i in range(env.n_parts):
        blk                = obs[i * env.obs_per_part : (i + 1) * env.obs_per_part]
        stock              = float(blk[0])  * max_stock
        on_order_nonurgent = float(blk[3])  * max_stock
        dyn_rop            = float(blk[10]) * max_stock
        dyn_eoq            = float(blk[11]) * max_order
        if stock <= dyn_rop and on_order_nonurgent < 1.0:
            action[i] = min(dyn_eoq, max_order)
    return action

def aggregate(rows):
    T_ep           = len(rows)
    total_cost     = sum(r['total_day_cost'] for r in rows)
    annual_cost    = total_cost / max(T_ep, 1) * 365.0
    total_demand   = sum(r[pt]['demand']    for r in rows for pt in part_types)
    total_shortage = sum(r[pt]['shortage']  for r in rows for pt in part_types)
    total_orders   = sum(1 for r in rows for pt in part_types if r[pt]['order_placed'])
    total_rush_ev  = sum(1 for r in rows for pt in part_types if r[pt]['urgent_qty'] > 0)
    total_order_qty= sum(r[pt]['order_qty'] for r in rows for pt in part_types)
    end_backorders = sum(rows[-1][pt]['backorders'] for pt in part_types) if rows else 0.0
    stockout_days  = sum(1 for r in rows if any(r[pt]['shortage'] > 0 for pt in part_types))
    breakdown      = {
        'ordering': sum(r[pt]['ordering'] for r in rows for pt in part_types),
        'rush':     sum(r[pt]['rush']     for r in rows for pt in part_types),
        'holding':  sum(r[pt]['holding']  for r in rows for pt in part_types),
    }
    return {
        'total_cost':      total_cost,
        'annual_cost':     annual_cost,
        'isl':             1.0 - total_shortage / max(total_demand, 1.0),
        'total_demand':    total_demand,
        'total_shortage':  total_shortage,
        'end_backorders':  end_backorders,
        'urgent_units':    sum(r[pt]['urgent_qty'] for r in rows for pt in part_types),
        'n_orders':        total_orders,
        'n_rush_events':   total_rush_ev,
        'total_order_qty': total_order_qty,
        'breakdown':       breakdown,
        'stockout_days':   stockout_days,
        'stockout_rate':   stockout_days / max(T_ep, 1),
        'cost_per_demand': total_cost / max(total_demand, 1.0),
    }


print('\n' + '='*80)
print('  EVALUATION — held-out test set (with mask)')
print('='*80)

results = {}
results['SIP'] = run_episode(make_test_env(), baseline_action)
print('  SIP  ✓')

for name, (model, eps) in trained_models.items():
    def make_action_fn(m):
        def fn(obs, env): a, _ = m.predict(obs, deterministic=True); return a
        return fn
    results[name] = run_episode(make_test_env(), make_action_fn(model))
    print(f'  {name}  ✓')

metrics  = {name: aggregate(rows) for name, rows in results.items()}
pnames   = list(metrics.keys())
sip_cost = metrics['SIP']['total_cost']

print('\n' + '='*100)
print(f"  RESULTS (with mask) — {T_test}-day test period")
print('='*100)
print(f"{'Policy':<8} {'Total SEK':>10} {'Ann. SEK':>9} {'ISL':>6} {'vs SIP':>7}"
      f" {'RegOrd':>7} {'RushEv':>8} {'Stockout%':>10}")
print('-'*75)
for n, m in metrics.items():
    ratio = m['total_cost'] / sip_cost
    print(f"{n:<8} {m['total_cost']:>10,.0f} {m['annual_cost']:>9,.0f} {m['isl']:>6.3f}"
          f" {ratio:>7.2f}x {m['n_orders']:>7d} {m['n_rush_events']:>8d}"
          f" {m['stockout_rate']*100:>9.1f}%")
print('='*100)

# Save results to CSV
rows_csv = []
for n, m in metrics.items():
    b   = m['breakdown']
    eps = trained_models[n][1] if n != 'SIP' else 'N/A'
    rows_csv.append({
        'Policy':          n,
        'Mask':            'Yes',
        'Episodes':        eps,
        'Total_Cost_SEK':  round(m['total_cost'], 2),
        'Annual_Cost_SEK': round(m['annual_cost'], 2),
        'ISL':             round(m['isl'], 4),
        'vs_SIP_ratio':    round(m['total_cost'] / sip_cost, 4),
        'N_Regular_Orders':m['n_orders'],
        'N_Rush_Events':   m['n_rush_events'],
        'Stockout_Rate':   round(m['stockout_rate'], 4),
        'Ordering_Cost':   round(b['ordering'], 2),
        'Rush_Cost':       round(b['rush'], 2),
        'Holding_Cost':    round(b['holding'], 4),
        'End_Backorders':  round(m['end_backorders'], 2),
    })

df_out = pd.DataFrame(rows_csv)
csv_out = 'results/with_mask_results.csv'
df_out.to_csv(csv_out, index=False)
print(f'\nResults saved to {csv_out}')
print('\nDone.')
