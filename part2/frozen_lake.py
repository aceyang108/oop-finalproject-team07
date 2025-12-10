import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os


class QLearningAgent:
    def __init__(self, env, learning_rate_a=0.035, discount_factor_g=0.995):
        self.env = env
        self.lr_a = learning_rate_a           
        self.gamma = discount_factor_g        
        self.epsilon = 1.0                    
        self.epsilon_min = 0.0                
        self.i = 0                            
        
        self.q_table = np.random.uniform(low=0, high=0.01, size=(env.observation_space.n, env.action_space.n))

    def choose_action(self, state, is_training):
        if is_training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state, :])

    def train(self, episodes, save_path="frozen_lake8x8.pkl"):
        rewards_history = np.zeros(episodes)
        best_success_rate = 0.0
        print(f"Start Training for {episodes} episodes")
        for self.i in range(episodes):
            state, _ = self.env.reset()
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                action = self.choose_action(state, is_training=True)
                new_state, reward, terminated, truncated, _ = self.env.step(action)
                target = reward
                if not terminated:
                    target += self.gamma * np.max(self.q_table[new_state, :])
                current_lr = max(0.005, self.lr_a * (1 - self.i/episodes))
                
                self.q_table[state, action] = self.q_table[state, action] + current_lr * (target - self.q_table[state, action])
                
                state = new_state
            if reward == 1 and terminated:
                rewards_history[self.i] = 1
            self.epsilon = max(self.epsilon_min, 1 - self.i / (episodes * 0.85))
            past_reward_100 = rewards_history[max(0, self.i-100):(self.i+1)]
            current_success_rate = np.sum(past_reward_100 == 1) 

            if self.i > 100 and current_success_rate > best_success_rate:
                best_success_rate = current_success_rate
                with open(save_path, "wb") as f:
                    pickle.dump(self.q_table, f)

            if (self.i + 1) % 100 == 0:
                print(f"Episode {self.i + 1}/{episodes} | Epsilon: {self.epsilon:.4f} | Current Rate: {current_success_rate}% | Best: {best_success_rate}%", end='\r')

        print(f"\nTraining Complete. Best success rate: {best_success_rate}%")
        return rewards_history

    def evaluate(self, episodes=10):
        success_count = 0
        print(f"\n Start Evaluation")
        for _ in range(episodes):
            state, _ = self.env.reset()
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action = np.argmax(self.q_table[state, :])
                state, reward, terminated, truncated, _ = self.env.step(action)
                if reward == 1 and terminated:
                    success_count += 1
        return success_count / episodes

def plot_results(rewards, episodes, filename='frozen_lake8x8.png'):
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards[max(0, t-100):(t+1)])
        
    plt.figure(figsize=(10, 5))
    plt.plot(sum_rewards)
    plt.title('Frozen Lake 8x8 - Last 100')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate % (Sum of Rewards)')
    plt.savefig(filename)
    print(f"📊 Plot saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FrozenLake Agent")
    
    parser.add_argument('--train', action='store_true', help='Run in training mode')
    parser.add_argument('--episodes', type=int, default=15000, help='Number of episodes (Default: 15000)')
    parser.add_argument('--render', action='store_true', help='Render the environment')

    args = parser.parse_args()
    
    model_file = "frozen_lake8x8.pkl"
    render_mode = 'human' if args.render else None
    
    if not args.train and not args.render:
        if not os.path.exists(model_file):
            print("No model found. Defaulting to Training Mode")
            run_train = True
            run_episodes = 15000
            run_render = False
        else:
            print("Model found. Defaulting to Render Mode")
            run_train = False
            run_episodes = 10
            run_render = True
    else:
        run_train = args.train
        run_episodes = args.episodes
        run_render = args.render

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode=render_mode if run_render else None)
    agent = QLearningAgent(env)

    if run_train:
        if os.path.exists(model_file):
            print(f"Warning  Overwriting existing model '{model_file}'")
        history = agent.train(run_episodes, save_path=model_file)
        plot_results(history, run_episodes)
    else:
        if os.path.exists(model_file):
            with open(model_file, "rb") as f:
                agent.q_table = pickle.load(f)
            rate = agent.evaluate(run_episodes)
            print(f"Final Success Rate: {rate * 100:.2f}%")
        else:
            print("No trained model found")

    env.close()