import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

# ==============================================================================
# 1. Configuration Class
# ==============================================================================
@dataclass
class ProjectConfig:
    """Centralized configuration for hyperparameters and settings."""
    env_name: str = 'FrozenLake-v1'
    map_name: str = "8x8"
    is_slippery: bool = True
    render_mode: Optional[str] = None
    
    success_rate: float = 0.75 

    # Training
    episodes: int = 15000
    learning_rate: float = 0.999
    lr_decay: float = 0.999
    lr_min: float = 0.01
    gamma: float = 0.95
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_min: float = 0.04
    epsilon_decay_factor: float = 0.95

    # Paths
    model_path: str = "frozen_lake8x8.pkl"
    plot_path: str = "frozen_lake8x8.png"

# ==============================================================================
# 2. Result Visualizer
# ==============================================================================
class ResultVisualizer:
    """Handles plotting and saving results."""
    @staticmethod
    def plot_training_curve(rewards: np.ndarray, save_path: str):
        if len(rewards) == 0: return

        # Calculate rolling success rate (last 100 episodes)
        episodes = len(rewards)
        rolling_success = np.zeros(episodes)
        window = 100
        
        for i in range(episodes):
            start = max(0, i - window)
            rolling_success[i] = np.sum(rewards[start : i+1])
            
        plt.figure(figsize=(10, 6))
        plt.plot(rolling_success, label='Success Count (Last 100)')
        plt.title('Frozen Lake 8x8 - Training Performance')
        plt.xlabel('Episodes')
        plt.ylabel('Successes')
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        print(f"📊 Plot saved to {save_path}")

# ==============================================================================
# 3. Custom Environment Wrapper
# ==============================================================================
class FrozenLakeCustomEnv(gym.Wrapper):
    """
    Wraps FrozenLake to modify physics (success rate) and shape rewards.
    """
    def __init__(self, config: ProjectConfig):
        env = gym.make(config.env_name, map_name=config.map_name, 
                       is_slippery=config.is_slippery, render_mode=config.render_mode)
        super().__init__(env)
        self.config = config
        self.nrow = env.unwrapped.nrow
        self.ncol = env.unwrapped.ncol
        self.goal_pos = (self.nrow - 1, self.ncol - 1)
        
        # Apply physics hack if configured
        if self.config.success_rate:
            self._apply_physics_hack(self.config.success_rate)

    def _apply_physics_hack(self, rate: float):
        """Modifies the P-table to increase success probability."""
        env = self.env.unwrapped
        p_success, p_slip = rate, (1.0 - rate) / 2.0
        
        def get_next(r, c, a):
            if a == 0: c = max(c - 1, 0)
            elif a == 1: r = min(r + 1, self.nrow - 1)
            elif a == 2: c = min(c + 1, self.ncol - 1)
            elif a == 3: r = max(r - 1, 0)
            return r, c

        for s in env.P:
            r, c = s // self.ncol, s % self.ncol
            for a in env.P[s]:
                trans = env.P[s][a]
                # Only modify valid slip states (len=3)
                if len(trans) == 3 and env.desc[r, c] not in b'GH':
                    new_trans = []
                    for offset in [0, -1, 1]:
                        eff_a = (a + offset) % 4
                        nr, nc = get_next(r, c, eff_a)
                        ns = nr * self.ncol + nc
                        letter = env.desc[nr, nc]
                        done = bytes(letter) in b"GH"
                        rew = float(letter == b"G")
                        prob = p_success if offset == 0 else p_slip
                        new_trans.append((prob, ns, rew, done))
                    env.P[s][a] = new_trans

    def _manhattan_dist(self, state):
        r, c = state // self.ncol, state % self.ncol
        return abs(r - self.goal_pos[0]) + abs(c - self.goal_pos[1])

    def step(self, action):
        prev_s = self.env.unwrapped.s
        ns, rew, term, trunc, info = self.env.step(action)
        
        # Reward Shaping Logic
        if term:
            rew = 10.0 if rew > 0 else -10.0
        else:
            # Potential-based shaping: old_dist - new_dist
            prev_dist = self._manhattan_dist(prev_s)
            curr_dist = self._manhattan_dist(ns)
            rew = float(prev_dist - curr_dist)
            
        return ns, rew, term, trunc, info

# ==============================================================================
# 4. Agent Class
# ==============================================================================
class QLearningAgent:
    """Manages Q-Table, Action Selection, and Learning."""
    def __init__(self, n_states, n_actions, config: ProjectConfig):
        self.n_states = n_states
        self.n_actions = n_actions
        self.cfg = config
        self.q_table = np.zeros((n_states, n_actions))
        
        # Hyperparameters
        self.lr = config.learning_rate
        self.eps = config.epsilon_start
        self.rng = np.random.default_rng()

    def select_action(self, state, training=True):
        """Epsilon-Greedy Policy."""
        if training and self.rng.random() < self.eps:
            return self.rng.integers(0, self.n_actions)
        return np.argmax(self.q_table[state, :])

    def update(self, state, action, reward, next_state):
        """Bellman Update Rule."""
        best_next = np.max(self.q_table[next_state, :])
        target = reward + self.cfg.gamma * best_next
        error = target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * error

    def decay(self):
        """Decay Epsilon and Learning Rate."""
        decay_step = (self.cfg.epsilon_start - self.cfg.epsilon_min) / \
                     (self.cfg.episodes * self.cfg.epsilon_decay_factor)
        self.eps = max(self.cfg.epsilon_min, self.eps - decay_step)
        self.lr = max(self.cfg.lr_min, self.lr * self.cfg.lr_decay)

    def save(self):
        try:
            with open(self.cfg.model_path, "wb") as f:
                pickle.dump(self.q_table, f)
            print(f"💾 Model saved to {self.cfg.model_path}")
        except IOError: print("❌ Error saving model")

    def load(self):
        if not os.path.exists(self.cfg.model_path): return False
        try:
            with open(self.cfg.model_path, "rb") as f:
                self.q_table = pickle.load(f)
            print(f"📂 Model loaded from {self.cfg.model_path}")
            return True
        except: return False

# ==============================================================================
# 5. Training Manager
# ==============================================================================
class TrainingManager:
    """Controls the main loop for training and evaluation."""
    def __init__(self, config: ProjectConfig, force_train=False):
        self.cfg = config
        self.env = FrozenLakeCustomEnv(config)
        self.agent = QLearningAgent(self.env.observation_space.n, self.env.action_space.n, config)
        self.force_train = force_train

    def run(self):
        has_model = self.agent.load()
        if self.force_train or not has_model:
            print("⚠️ Starting Training Mode.")
            self.train()
        else:
            print("✅ Starting Evaluation Mode.")
            self.evaluate()
        self.env.close()

    def train(self):
        print(f"🚀 Training for {self.cfg.episodes} episodes...")
        history = np.zeros(self.cfg.episodes)
        
        for i in range(self.cfg.episodes):
            s, _ = self.env.reset()
            done = False
            
            while not done:
                a = self.agent.select_action(s, training=True)
                ns, r, term, trunc, _ = self.env.step(a)
                self.agent.update(s, a, r, ns)
                s = ns
                done = term or trunc
            
            self.agent.decay()
            if r >= 10: history[i] = 1 # Record success
            
            if (i+1) % 100 == 0:
                rate = np.sum(history[max(0, i-100):i+1])
                print(f"Ep {i+1} | Eps: {self.agent.eps:.4f} | LR: {self.agent.lr:.4f} | Rate: {rate:.0f}%", end='\r')

        print("\n✅ Training Complete.")
        self.agent.save()
        ResultVisualizer.plot_training_curve(history, self.cfg.plot_path)

    def evaluate(self, episodes=10):
        print(f"\n🔍 Evaluating for {episodes} episodes...")
        successes = 0
        for i in range(episodes):
            s, _ = self.env.reset()
            done = False
            steps = 0
            while not done:
                a = self.agent.select_action(s, training=False)
                s, r, term, trunc, _ = self.env.step(a)
                done = term or trunc
                steps += 1
                if self.cfg.render_mode == 'human': time.sleep(0.1)
            
            result = "SUCCESS" if r >= 10 else "FAIL"
            if r >= 10: successes += 1
            print(f"Episode {i+1}: {result} (Steps: {steps})")
        
        print(f"🎯 Final Success Rate: {(successes/episodes)*100:.2f}%")

# ==============================================================================
# 6. Main Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="OOP FrozenLake Project")
    parser.add_argument('--train', action='store_true', help='Force training')
    parser.add_argument('--episodes', type=int, default=15000, help='Training episodes')
    parser.add_argument('--render', action='store_true', help='Visualize agent')
    args = parser.parse_args()

    config = ProjectConfig()
    config.episodes = args.episodes
    
    # Intelligent mode switching logic
    if args.render:
        config.render_mode = 'human' if not args.train else None
    
    # Run Manager
    manager = TrainingManager(config, force_train=args.train)
    manager.run()

if __name__ == "__main__":
    main()