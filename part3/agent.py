import gymnasium as gym
import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# 引入自訂的環境模組
import my_env 

# ===========================
# Agent Class
# ===========================

class CrossyRoadAgent:
    """
    CrossyRoad 強化學習代理
    
    負責管理 RL 訓練生命週期：
    - 模型初始化與建構
    - 訓練與自動存檔
    - 模型載入與儲存
    - 實際遊玩測試與評估
    """
    
    def __init__(self, env_id='crossy-road-v0', model_name="dqn_crossy_road", log_dir="./logs/"):
        """
        初始化 Agent
        
        Parameters:
            env_id (str): Gym 環境 ID
            model_name (str): 模型檔案名稱
            log_dir (str): 日誌與模型存檔路徑
        """
        self.env_id = env_id
        self.model_name = model_name
        self.log_dir = log_dir
        self.model_path = os.path.join(log_dir, model_name)
        
        # 建立日誌目錄
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化訓練環境（不渲染以加速訓練）
        self.env = gym.make(self.env_id, render_mode=None)
        
        # 模型實例（稍後在 train 或 load 中初始化）
        self.model = None

    def build_model(self, learning_rate=1e-4):
        """
        建構 DQN 模型
        
        Parameters:
            learning_rate (float): 學習率
        """
        print(f"Building new DQN model for {self.env_id}...")
        
        # 定義神經網路架構
        # net_arch=[256, 256] 代表兩層隱藏層，每層 256 個神經元
        policy_kwargs = dict(net_arch=[256, 256])

        # 建立 DQN 模型
        self.model = DQN(
            "MlpPolicy",                    # 使用多層感知機策略
            self.env,                       # 訓練環境
            verbose=1,                      # 顯示訓練進度
            learning_rate=learning_rate,    # 學習率
            buffer_size=100000,             # 經驗回放緩衝區大小
            gamma=0.995,                    # 折扣因子
            exploration_fraction=0.5,       # 探索階段佔比
            exploration_final_eps=0.05,     # 最終探索率
            tensorboard_log=self.log_dir,   # TensorBoard 日誌路徑
            policy_kwargs=policy_kwargs     # 神經網路架構參數
        )

    def train(self, total_timesteps=500000):
        """
        執行訓練流程
        
        Parameters:
            total_timesteps (int): 總訓練步數
        """
        # 若模型未建立，先建構模型
        if self.model is None:
            self.build_model()
            
        print(f"Start training for {total_timesteps} steps...")
        
        # 設定自動存檔回調機制
        # 每 10000 步自動儲存一次
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, 
            save_path=self.log_dir, 
            name_prefix=self.model_name
        )
        
        # 開始訓練
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        print("Training finished.")
        
        # 訓練完成後自動存檔
        self.save()

    def save(self):
        """儲存模型到檔案"""
        if self.model:
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")
        else:
            print("No model to save.")

    def load(self, path=None):
        """
        從檔案載入模型
        
        Parameters:
            path (str): 模型檔案路徑（若為 None 則使用預設路徑）
        """
        load_path = path if path else self.model_path
        
        if os.path.exists(load_path + ".zip"):
            print(f"Loading model from {load_path}...")
            self.model = DQN.load(load_path, env=self.env)
        else:
            print(f"Model file not found at {load_path}, please train first.")

    def play(self, episodes=5):
        """
        讓 Agent 實際遊玩並顯示畫面（推論模式）
        
        Parameters:
            episodes (int): 遊玩場次
        """
        if self.model is None:
            print("No model loaded. Please load or train a model first.")
            return

        from stable_baselines3.common.vec_env import DummyVecEnv

        # 使用 DummyVecEnv 包裝環境（啟用渲染）
        play_env = DummyVecEnv([lambda: gym.make(self.env_id, render_mode='human')])
        
        print(f"Agent playing for {episodes} episodes...")
        
        # 取得第一場遊戲的初始觀察
        obs = play_env.reset()
        
        for i in range(episodes):
            done = False
            total_reward = 0
            
            while not done:
                # 使用訓練好的策略預測動作
                action, _ = self.model.predict(obs, deterministic=True)
                
                # 執行動作並取得結果
                # 注意：DummyVecEnv 會在遊戲結束時自動重置
                obs, reward, done_array, info = play_env.step(action)
                
                total_reward += reward[0]
                done = done_array[0]
                
            print(f"Episode {i+1} finished. Total Reward: {total_reward}")
        
        play_env.close()

    def evaluate(self, episodes=100):
        """
        評估模型性能（無渲染，高效率）
        
        計算平均獎勵與勝率
        
        Parameters:
            episodes (int): 評估場次
        """
        if self.model is None:
            print("No model loaded.")
            return

        print(f"Starting evaluation over {episodes} episodes...")
        
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # 建立評估環境（不渲染以加速）
        eval_env = DummyVecEnv([lambda: gym.make(self.env_id, render_mode=None)])

        total_rewards = []
        success_count = 0

        for i in range(episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # 使用確定性策略（不探索）
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done_array, info = eval_env.step(action)
                
                episode_reward += reward[0]
                done = done_array[0]
                
                # 檢測是否獲勝（獎勵 >= 100 代表到達終點）
                if reward[0] >= 100: 
                    success_count += 1

            total_rewards.append(episode_reward)

        # 計算統計數據
        avg_reward = sum(total_rewards) / episodes
        win_rate = success_count / episodes * 100

        print(f"Evaluation Results ({episodes} episodes):")
        print(f"  - Average Reward: {avg_reward:.2f}")
        print(f"  - Win Rate: {win_rate:.2f}%")
        
        eval_env.close()

# ===========================
# Main Entry Point
# ===========================

if __name__ == "__main__":
    # 實例化 Agent
    agent = CrossyRoadAgent()

    # 選擇運行模式
    mode = input("Select mode (train/play/evaluate/benchmark): ").strip().lower()

    if mode == "train":
        # 訓練模式
        if os.path.exists(agent.model_path + ".zip"):
            cont = input("Model exists. Continue training? (y/n): ")
            if cont.lower() == 'y':
                agent.load()
        
        agent.train(total_timesteps=100000)
        
    elif mode == "play":
        # 遊玩展示模式
        agent.load()
        agent.play(episodes=10)
    
    elif mode == "evaluate":
        # 評估模式
        agent.load()
        agent.evaluate(episodes=100)
    
    elif mode == "benchmark":
        # 基準測試：隨機代理性能
        env = gym.make('crossy-road-v0', render_mode=None)
        episodes = 100
        success_count = 0
        total_reward = 0
        
        print(f"Benchmarking Random Agent over {episodes} episodes...")
        
        for _ in range(episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                # 隨機選擇動作（無策略）
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                
                # 檢測是否獲勝
                if reward >= 100:
                    success_count += 1
        
        # 顯示隨機代理的性能
        print(f"Random Agent Win Rate: {success_count/episodes*100:.2f}%")
        print(f"Random Agent Avg Reward: {total_reward/episodes:.2f}")
        
        env.close()
    
    else:
        print("Invalid mode. Please select 'train', 'play', 'evaluate', or 'benchmark'.")