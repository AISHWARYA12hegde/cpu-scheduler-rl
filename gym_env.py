"""
Gym Environment Wrapper
--------------------------
Wraps CPUSimulator in the standard Gymnasium interface (reset / step)
so a reinforcement learning agent can interact with it like any other
RL environment.

This file is responsible for two RL-specific things the simulator
itself doesn't know about:
  1. STATE representation  — turning the ready queue into a fixed-size
     numeric vector the agent can read.
  2. REWARD function        — turning "what happened this step" into a
     single number that tells the agent whether that was a good or
     bad scheduling decision.

State per process (normalized to roughly [0, 1]):
    [remaining_burst_time, waiting_time, priority]
Padded with zeros up to MAX_QUEUE_SIZE processes, so the state vector
is always the same length regardless of how many processes are
actually waiting.

Reward (per report section 4.5.1 — weighted sum of):
    - waiting time incurred this step      (negative)
    - a context-switch penalty             (negative, only if switched)
    - a fairness bonus based on Jain's Index of current waiting times
    - a small completion bonus when a process finishes
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulator import CPUSimulator
from workload_generator import generate_workload

# --- Fixed sizes so the state/action space never changes shape ---
MAX_QUEUE_SIZE = 10          # max number of processes the agent can choose among
FEATURES_PER_PROCESS = 3     # [remaining_burst, waiting_time, priority]

# --- Normalization constants (rough upper bounds, used to squash values to ~[0,1]) ---
MAX_BURST_TIME = 20.0
MAX_WAITING_TIME = 50.0
MAX_PRIORITY = 10.0

# --- Reward weights (tune these later once training is working) ---
W_WAITING = 1.0
W_CONTEXT_SWITCH = 2.0
W_FAIRNESS = 5.0
W_COMPLETION_BONUS = 1.0


class CPUSchedulingEnv(gym.Env):
    """A Gymnasium-compatible environment wrapping CPUSimulator."""

    metadata = {"render_modes": []}

    def __init__(self, num_processes: int = 8, seed: int = None):
        super().__init__()
        self.num_processes = num_processes
        self._seed = seed

        # Action: pick an index into the ready queue (0 .. MAX_QUEUE_SIZE-1).
        # If the chosen index is beyond the current queue length, we clip
        # it down to the last valid index (see step() below).
        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)

        # Observation: a flat vector of [burst, wait, priority] per process
        # slot, all normalized to roughly [0, 1].
        obs_size = MAX_QUEUE_SIZE * FEATURES_PER_PROCESS
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        self.sim = None  # created fresh in reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        use_seed = seed if seed is not None else self._seed

        # A fresh random workload each episode keeps the agent from
        # memorizing one specific process sequence.
        workload = generate_workload(
            num_processes=self.num_processes,
            seed=use_seed if use_seed is not None else np.random.randint(0, 1_000_000),
        )
        self.sim = CPUSimulator(workload)

        # Advance past any initial idle time before the first process arrives.
        self._skip_idle_time()

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: int):
        if self.sim is None:
            raise RuntimeError("Call reset() before step().")

        queue_len_before = len(self.sim.ready_queue)

        # Clip the action to a valid index in case the agent picked a
        # slot beyond the current (shorter) queue.
        action_index = min(action, queue_len_before - 1)
        action_index = max(action_index, 0)

        _, done, info = self.sim.step(action_index=action_index, time_slice=None)

        reward = self._compute_reward(info, queue_len_before)

        # Skip forward in time if the queue is empty but processes remain.
        if not done:
            self._skip_idle_time()

        obs = self._get_observation()
        terminated = done
        truncated = False  # no artificial episode length cap for now

        return obs, reward, terminated, truncated, info

    def _skip_idle_time(self):
        """If the ready queue is empty but processes are still pending,
        jump the simulator clock forward to the next arrival."""
        while not self.sim.ready_queue and self.sim._pending:
            self.sim.time = self.sim._pending[0].arrival_time
            self.sim._admit_arrivals()

    def _get_observation(self) -> np.ndarray:
        """Build the fixed-size, normalized state vector from the
        current ready queue."""
        obs = np.zeros(MAX_QUEUE_SIZE * FEATURES_PER_PROCESS, dtype=np.float32)

        queue = self.sim.ready_queue[:MAX_QUEUE_SIZE] if self.sim else []
        for i, proc in enumerate(queue):
            base = i * FEATURES_PER_PROCESS
            obs[base + 0] = min(proc.remaining_burst_time / MAX_BURST_TIME, 1.0)
            obs[base + 1] = min(proc.waiting_time / MAX_WAITING_TIME, 1.0)
            obs[base + 2] = proc.priority / MAX_PRIORITY

        return obs

    def _compute_reward(self, info: dict, queue_len_before: int) -> float:
        """Turn the outcome of one step into a single reward value."""
        run_time = info["run_time"]

        # Waiting time cost: every process still queued waited `run_time`
        # longer. More processes waiting or longer runs = worse.
        processes_left_waiting = max(queue_len_before - 1, 0)
        waiting_penalty = W_WAITING * run_time * processes_left_waiting

        # Context switch cost: only applied if the CPU actually switched
        # to a different process than the one before.
        switch_penalty = W_CONTEXT_SWITCH if info["context_switch"] else 0.0

        # Fairness bonus: Jain's Fairness Index over current waiting
        # times in the queue (1.0 = perfectly fair, closer to 0 = unfair).
        fairness_bonus = W_FAIRNESS * self._current_fairness_index()

        # Small bonus for successfully finishing a process.
        completion_bonus = W_COMPLETION_BONUS if info["finished"] else 0.0

        reward = -waiting_penalty - switch_penalty + fairness_bonus + completion_bonus
        return float(reward)

    def _current_fairness_index(self) -> float:
        """Jain's Fairness Index over the waiting times of processes
        currently sitting in the ready queue."""
        waits = [p.waiting_time for p in self.sim.ready_queue]
        if not waits or sum(waits) == 0:
            return 1.0  # nobody waiting, or all waited equally (0) -> perfectly fair
        n = len(waits)
        sum_w = sum(waits)
        sum_w2 = sum(w ** 2 for w in waits)
        return (sum_w ** 2) / (n * sum_w2)


if __name__ == "__main__":
    # Quick manual test: run one full episode with a random policy,
    # just to confirm the environment doesn't crash and produces
    # sensible-looking rewards.
    env = CPUSchedulingEnv(num_processes=5, seed=42)
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    total_reward = 0.0
    steps = 0
    terminated = False
    while not terminated:
        action = env.action_space.sample()  # random action for this test
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        print(f"Step {steps}: ran {info['ran_process_id']}, reward={reward:.2f}")

    print(f"\nEpisode finished in {steps} steps. Total reward: {total_reward:.2f}")
    print("Final metrics:", env.sim.get_metrics())