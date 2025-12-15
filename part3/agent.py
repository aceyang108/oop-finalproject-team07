import gymnasium as gym
import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# 引入你定義好的環境
import my_env 

class CrossyRoadAgent:
    """
    CrossyRoadAgent 負責管理強化學習的生命週期：
    初始化、訓練、存檔、讀檔、以及實際遊玩測試。
    """
    def __init__(self, env_id='crossy-road-v0', model_name="dqn_crossy_road", log_dir="./logs/"):
        self.env_id = env_id
        self.model_name = model_name
        self.log_dir = log_dir
        self.model_path = os.path.join(log_dir, model_name)
        
        # 建立目錄
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化環境 (訓練時不需要 render，速度較快)
        self.env = gym.make(self.env_id, render_mode=None)
        
        # 初始化模型 (這裡是空的，稍後在 train 或 load 中填入)
        self.model = None

    def build_model(self, learning_rate=1e-4):
        print(f"Building new DQN model for {self.env_id}...")
        
        # [新增] 定義更大的神經網路
        # net_arch=[256, 256] 代表兩層隱藏層，每層 256 個神經元 (比預設的 64 強很多)
        policy_kwargs = dict(net_arch=[256, 256])

        self.model = DQN(
            "MlpPolicy", 
            self.env, 
            verbose=1, 
            learning_rate=learning_rate,
            buffer_size=100000,
            gamma=0.995,
            exploration_fraction=0.5,
            exploration_final_eps=0.05,
            tensorboard_log=self.log_dir,
            policy_kwargs=policy_kwargs
        )

    def train(self, total_timesteps=500000):
        """執行訓練流程"""
        if self.model is None:
            self.build_model()
            
        print(f"Start training for {total_timesteps} steps...")
        
        # 設定自動存檔機制
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, 
            save_path=self.log_dir, 
            name_prefix=self.model_name
        )
        
        # 開始學習 (封裝了複雜的數學運算)
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        print("Training finished.")
        
        # 訓練完自動存檔
        self.save()

    def save(self):
        """儲存模型"""
        if self.model:
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")
        else:
            print("No model to save.")

    def load(self, path=None):
        """載入模型"""
        load_path = path if path else self.model_path
        if os.path.exists(load_path + ".zip"):
            print(f"Loading model from {load_path}...")
            self.model = DQN.load(load_path, env=self.env)
        else:
            print(f"Model file not found at {load_path}, please train first.")

    def play(self, episodes=5):
        """讓 Agent 實際玩給你看 (Inference Mode)"""
        if self.model is None:
            print("No model loaded. Please load or train a model first.")
            return

        from stable_baselines3.common.vec_env import DummyVecEnv

        # 使用 DummyVecEnv 包裝
        play_env = DummyVecEnv([lambda: gym.make(self.env_id, render_mode='human')])
        
        print(f"Agent playing for {episodes} episodes...")
        
        # --- 修改點 1: 將 reset 移到迴圈外面 ---
        # 取得第一場遊戲的初始畫面
        obs = play_env.reset()
        
        for i in range(episodes):
            done = False
            total_reward = 0
            
            while not done:
                # 預測動作
                action, _ = self.model.predict(obs, deterministic=True)
                
                # 執行動作
                # 注意：如果這一步導致遊戲結束 (done=True)，
                # DummyVecEnv 會自動幫你 reset，這裡回傳的 obs 已經是「新遊戲」的畫面了
                obs, reward, done_array, info = play_env.step(action)
                
                total_reward += reward[0]
                done = done_array[0]
                
            print(f"Episode {i+1} finished. Total Reward: {total_reward}")
            # --- 修改點 2: 這裡不需要再呼叫 play_env.reset() 了 ---
            # 因為下一場遊戲的 obs 已經在上面最後一次 step 拿到了
        
        play_env.close()
    def evaluate(self, episodes=100):
        """
        科學驗證模式：跑 N 場，計算勝率與平均分
        """
        if self.model is None:
            print("No model loaded.")
            return

        print(f"Starting evaluation over {episodes} episodes...")
        
        # 使用 DummyVecEnv 避免維度錯誤，且不需要 render_mode (加速)
        from stable_baselines3.common.vec_env import DummyVecEnv
        eval_env = DummyVecEnv([lambda: gym.make(self.env_id, render_mode=None)])

        total_rewards = []
        success_count = 0

        for i in range(episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # deterministic=True 代表使用訓練出的最佳策略
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done_array, info = eval_env.step(action)
                
                episode_reward += reward[0]
                done = done_array[0]
                
                # 根據 my_env.py，到達終點會給 +100 獎勵
                # 如果單步獲得 +100 (或更多)，代表這場贏了
                if reward[0] >= 100: 
                    success_count += 1

            total_rewards.append(episode_reward)

        avg_reward = sum(total_rewards) / episodes
        win_rate = success_count / episodes * 100

        print(f"Evaluation Results ({episodes} episodes):")
        print(f"  - Average Reward: {avg_reward:.2f}")
        print(f"  - Win Rate: {win_rate:.2f}%")
        
        eval_env.close()
# --- 主程式進入點 ---
if __name__ == "__main__":
    # 實例化 Agent (這就是 OOP 的精髓，透過物件來操作)
    agent = CrossyRoadAgent()

    # 選擇模式：要訓練還是要看結果？
    mode = input("Select mode (train/play): ").strip().lower()

    if mode == "train":
        # 如果有舊存檔，可以選擇是否繼續訓練
        if os.path.exists(agent.model_path + ".zip"):
            cont = input("Model exists. Continue training? (y/n): ")
            if cont.lower() == 'y':
                agent.load()
        
        agent.train(total_timesteps=100000)
        
    elif mode == "play":
        agent.load()
        agent.play(episodes=10)
    # ... 在 main 區塊中增加 ...
    elif mode == "benchmark":
        # 建立一個臨時環境
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
                # 隨機選動作 (瞎猜)
                action = env.action_space.sample() 
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if reward >= 100:
                    success_count += 1
        
        print(f"Random Agent Win Rate: {success_count/episodes*100:.2f}%")
        print(f"Random Agent Avg Reward: {total_reward/episodes:.2f}")