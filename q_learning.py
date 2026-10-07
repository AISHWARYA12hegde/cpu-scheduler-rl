"""
Q-Learning Agent
-------------------
This is the actual "learner" in the project. It maintains a Q-table —
a lookup table that estimates, for every (state, action) pair, how
much total future reward we expect if we take that action in that
state.

Since our Gym environment's state is a continuous vector (normalized
burst time / waiting time / priority per process, ~[0,1]), and tabular
Q-learning needs DISCRETE states, we first "bin" each continuous value
into one of a small number of buckets (e.g. "low / medium / high").
This turns each state into a fixed tuple of integers, which we can use
as a dictionary key.

Training uses an epsilon-greedy policy (per report Appendix B):
  - Start with epsilon = 1.0 (act completely randomly -> explore)
  - Decay epsilon toward 0.01 (mostly act on learned knowledge -> exploit)
  - Learning rate (alpha) = 0.1
  - Discount factor (gamma) = 0.95
  - Episodes = 1000 (configurable; fewer for a quick test run)

Q-learning update rule:
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
"""

import numpy as np
from collections import defaultdict

from gym_env import CPUSchedulingEnv, MAX_QUEUE_SIZE, FEATURES_PER_PROCESS

# --- Hyperparameters (matches report Appendix B) ---
ALPHA = 0.1            # learning rate
GAMMA = 0.95            # discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995    # multiplied into epsilon after every episode
NUM_BINS = 4             # how many buckets each continuous feature is split into


def discretize(obs: np.ndarray, num_bins: int = NUM_BINS) -> tuple:
    """
    Convert the continuous observation vector (values roughly in [0,1])
    into a tuple of small integers, so it can be used as a dictionary
    key in the Q-table.

    Example: a value of 0.42 with num_bins=4 becomes bucket 1 (of 0,1,2,3).
    """
    bins = np.clip((obs * num_bins).astype(int), 0, num_bins - 1)
    return tuple(bins.tolist())


class QLearningAgent:
    def __init__(self, num_actions: int, alpha=ALPHA, gamma=GAMMA,
                 epsilon=EPSILON_START, epsilon_min=EPSILON_MIN,
                 epsilon_decay=EPSILON_DECAY):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: maps a discretized state (tuple) -> array of expected
        # rewards, one per possible action. defaultdict means any
        # state we haven't seen before automatically starts at all zeros.
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

    def choose_action(self, state: tuple, training: bool = True) -> int:
        """Epsilon-greedy action selection.
        training=True  -> may explore randomly (used during training)
        training=False -> always exploit the best known action (used during evaluation)
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)  # explore
        return int(np.argmax(self.q_table[state]))       # exploit best known action

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple, done: bool):
        """Apply the Q-learning update rule after one step."""
        best_next_value = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(num_episodes: int = 1000, num_processes: int = 8, verbose_every: int = 50,
          num_training_workloads: int = 10):
    """
    Train a Q-learning agent on the CPU scheduling environment.

    Parameters
    ----------
    num_training_workloads : int
        Instead of generating a brand-new random workload every single
        episode (which makes the state space too sparse for tabular
        Q-learning to ever revisit the same situation), we cycle
        through a small, fixed pool of workloads. This lets the agent
        actually see similar situations repeatedly and learn from them
        -- the standard fix for tabular methods with limited episodes.

    Returns
    -------
    agent : QLearningAgent
        The trained agent (its Q-table holds the learned policy).
    episode_rewards : list[float]
        Total reward per episode — used later to plot the learning curve.
    """
    env = CPUSchedulingEnv(num_processes=num_processes)
    agent = QLearningAgent(num_actions=env.action_space.n)

    # Fixed pool of workload seeds, reused (cycled through) every episode.
    training_seeds = list(range(num_training_workloads))

    episode_rewards = []

    for episode in range(num_episodes):
        seed = training_seeds[episode % len(training_seeds)]
        obs, _ = env.reset(seed=seed)
        state = discretize(obs)
        total_reward = 0.0
        terminated = False

        while not terminated:
            action = agent.choose_action(state, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize(next_obs)

            agent.update(state, action, reward, next_state, terminated)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if verbose_every and (episode + 1) % verbose_every == 0:
            recent_avg = np.mean(episode_rewards[-verbose_every:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"avg reward (last {verbose_every}): {recent_avg:.2f} | "
                  f"epsilon: {agent.epsilon:.3f} | "
                  f"Q-table size: {len(agent.q_table)}")

    return agent, episode_rewards


if __name__ == "__main__":
    # Quick test run with fewer episodes, just to confirm training runs
    # end-to-end without errors and that reward trends upward over time.
    print("Training Q-learning agent (full run: 1000 episodes)...\n")
    agent, rewards = train(num_episodes=1000, num_processes=8, verbose_every=100)

    print(f"\nFirst 20 episodes avg reward: {np.mean(rewards[:20]):.2f}")
    print(f"Last 20 episodes avg reward:  {np.mean(rewards[-20:]):.2f}")
    print(f"Final Q-table size (unique states seen): {len(agent.q_table)}")