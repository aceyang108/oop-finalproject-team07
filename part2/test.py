import gymnasium as gym
import numpy as np
import pickle
import os
import argparse
import time

def run_test(model_file, episodes, render=False):
    if not os.path.exists(model_file):
        print(f"Model file '{model_file}' not found!")
        return

    try:
        with open(model_file, 'rb') as f:
            q_table = pickle.load(f)
        print(f"Model loaded successfully: {model_file}")
    except Exception as e:
        print(f"Error loading file")
        return
    render_mode = 'human' if render else None
    
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode=render_mode)

    success_count = 0
    print(f"\nStarting Evaluation")
    print("-" * 40)

    for i in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        step_count = 0

        while not terminated and not truncated:
            action = np.argmax(q_table[state, :])
            state, _, terminated, truncated, _ = env.step(action)
            step_count += 1

        if state == 63: 
            success_count += 1
            result = "SUCCESS"
        else:
            result = "FAIL  "

        if render:
            print(f"Episode {i+1}: {result} (Steps: {step_count})")
            time.sleep(0.5) 

    env.close()


    success_rate = (success_count / episodes) * 100
    print("-" * 40)
    print(f"Evaluation Report")
    print(f"Model: {model_file}")
    print(f"Total Episodes: {episodes}")
    print(f"Success Count:  {success_count}")
    print(f"Success Rate:   {success_rate:.2f}%")
    print("=" * 40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained Q-Learning agent on FrozenLake 8x8.")
    
    parser.add_argument('--model', type=str, default='frozen_lake8x8.pkl', 
                        help='Path to the .pkl model file (default: frozen_lake8x8.pkl)')
    
    parser.add_argument('--episodes', type=int, default=10, 
                        help='Number of episodes to run (default: 10)')
    
    parser.add_argument('--benchmark', action='store_true', 
                        help='Run in benchmark mode (1000 episodes, no render, fast)')

    args = parser.parse_args()

    if args.benchmark:
        print("\n[Mode: Benchmark]")
        run_test(args.model, episodes=1000, render=False)
    else:
        print("\n[Mode: Visual Demo]")
        run_test(args.model, episodes=args.episodes, render=True)