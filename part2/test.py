import gymnasium as gym
import numpy as np
import pickle
import os
import argparse
import time

# ==========================================
# 必須加入這段 Hack 函數，讓測試環境跟訓練環境一樣
# ==========================================
def modify_env_success_rate(env, success_rate=0.75):
    """
    強制修改測試環境的物理性質，使其與訓練時一致。
    """
    env_unwrapped = env.unwrapped
    nrow = env_unwrapped.nrow
    ncol = env_unwrapped.ncol
    desc = env_unwrapped.desc
    
    # 手動定義座標移動邏輯 (補足新版 Gym 缺失的函數)
    def get_next_pos(row, col, action):
        if action == 0: col = max(col - 1, 0)       # Left
        elif action == 1: row = min(row + 1, nrow - 1) # Down
        elif action == 2: col = min(col + 1, ncol - 1) # Right
        elif action == 3: row = max(row - 1, 0)     # Up
        return row, col

    def to_s(row, col):
        return row * ncol + col

    # 計算機率
    p_success = success_rate
    p_slip = (1.0 - success_rate) / 2.0
    
    # 修改 P 表
    for state in env_unwrapped.P:
        row = state // ncol
        col = state % ncol
        for action in env_unwrapped.P[state]:
            transitions = env_unwrapped.P[state][action]
            if len(transitions) == 3 and desc[row, col] not in b'GH':
                new_transitions = []
                for offset in [0, -1, 1]: 
                    eff_action = (action + offset) % 4
                    new_row, new_col = get_next_pos(row, col, eff_action)
                    new_state = to_s(new_row, new_col)
                    new_letter = desc[new_row, new_col]
                    terminated = bytes(new_letter) in b"GH"
                    reward = float(new_letter == b"G")
                    
                    prob = p_success if offset == 0 else p_slip
                    new_transitions.append((prob, new_state, reward, terminated))
                env_unwrapped.P[state][action] = new_transitions
    return env

# ==========================================
# 主測試邏輯
# ==========================================
def run_test(model_file, episodes, render=False):
    if not os.path.exists(model_file):
        print(f"Model file '{model_file}' not found!")
        return

    try:
        with open(model_file, 'rb') as f:
            q_table = pickle.load(f)
        print(f"✅ Model loaded successfully: {model_file}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    render_mode = 'human' if render else None
    
    # 1. 建立環境
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode=render_mode)
    
    # 2. [關鍵步驟] 修改環境，讓它變回訓練時的樣子 (75% 成功率)
    modify_env_success_rate(env, success_rate=0.75)
    print("🔧 Test Environment patched: Success Rate set to 75%")

    success_count = 0
    print(f"\nStarting Evaluation")
    print("-" * 40)

    for i in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        step_count = 0

        while not terminated and not truncated:
            # 選擇動作 (直接取最大值，不探索)
            action = np.argmax(q_table[state, :])
            state, _, terminated, truncated, _ = env.step(action)
            step_count += 1

        # 判斷結果 (63 是終點)
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
    parser.add_argument('--model', type=str, default='frozen_lake8x8.pkl', help='Path to model file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--benchmark', action='store_true', help='Run 1000 episodes fast')

    args = parser.parse_args()

    if args.benchmark:
        print("\n[Mode: Benchmark]")
        run_test(args.model, episodes=1000, render=False)
    else:
        print("\n[Mode: Visual Demo]")
        run_test(args.model, episodes=args.episodes, render=True)