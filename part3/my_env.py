import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import crossy_road as cr
import numpy as np
import random

# ===========================
# Environment Registration
# ===========================

# 註冊此模組為 gym 環境，註冊後可透過 gym.make() 使用
register(
    id='crossy-road-v0',
    entry_point='my_env:CrossyRoadEnv',
)

# ===========================
# Custom Gym Environment
# ===========================

# 自訂 Gym 環境，必須繼承自 gym.Env
# 參考文件：https://gymnasium.farama.org/api/env/
class CrossyRoadEnv(gym.Env):
    """
    Crossy Road 的 Gymnasium 環境實作
    
    提供標準化的 RL 環境介面，讓 AI 代理可以學習如何過馬路
    """
    
    # metadata 是必要屬性
    # render_modes 可為 None 或 'human'
    # render_fps 在此環境中未使用，但必須宣告非零值
    metadata = {"render_modes": ["human"], 'render_fps': 2}

    def __init__(self, grid_rows=10, grid_cols=11, render_mode=None):
        """
        初始化環境
        
        Parameters:
            grid_rows (int): 網格列數
            grid_cols (int): 網格欄數
            render_mode (str): 渲染模式，可為 None 或 'human'
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.render_mode = render_mode

        # 初始化 CrossyRoad 遊戲實例
        self.crossy_road = cr.CrossyRoad(
            grid_rows=grid_rows, 
            grid_cols=grid_cols, 
            fps=self.metadata['render_fps'], 
            road_count=6, 
            rand_seed=None
        )
        
        # 定義動作空間：4 個方向移動
        self.action_space = spaces.Discrete(len(cr.PedestrianAction))
        
        # 環境限制參數
        self.max_roads = 10
        self.max_cars = 20
        self.max_episode_steps = 200
        
        # 計算觀察空間大小
        # 每台車的特徵：5 個 (Row, Col, Dir, Speed, Length)
        # 公式：4 (行人+目標資訊) + (路數 * 2) + (車數 * 5)
        self.max_obs_len = 4 + (self.max_roads * 2) + (self.max_cars * 5)
        
        # 定義觀察空間
        self.observation_space = spaces.Box(
            low=-999,
            high=999,
            shape=(self.max_obs_len,),
            dtype=np.int32
        )

    def _get_obs(self):
        """
        取得當前環境的觀察狀態
        
        觀察向量結構：
        [行人列位置, 行人欄位置, 目標列, 道路數量, 
         道路資訊 (每條路：列, 方向), 
         車輛資訊 (每台車：列, 欄, 方向, 速度, 長度)]
        
        Returns:
            np.ndarray: 觀察向量
        """
        # 收集基礎資訊：行人位置、目標列、道路數量
        numeric_data_list = list(self.crossy_road.pedestrian_pos) + [
            self.crossy_road.goal_row, 
            self.crossy_road.road_count
        ]
        
        # 收集道路資訊（最多 10 條）
        for road in self.crossy_road.road_rows[:self.max_roads]:
            dir_num = 0 if road[1] == 'L' else 1
            numeric_data_list.extend([road[0], dir_num])

        # 收集車輛資訊（最多 20 台）
        # 包含：行、列、方向、速度、長度
        for car in self.crossy_road.cars[:self.max_cars]:
            dir_num = 0 if car.direction == 'L' else 1
            numeric_data_list.extend([
                car.row, 
                car.column, 
                dir_num,
                car.speed,
                car.length
            ])

        # 轉換為 numpy array
        current_data = np.array(numeric_data_list, dtype=np.int32)
        
        # 補零至固定長度（Padding）
        obs = np.zeros(self.max_obs_len, dtype=np.int32)
        length = min(len(current_data), self.max_obs_len)
        obs[:length] = current_data[:length]
        
        return obs

    def reset(self, seed=None, options=None):
        """
        重置環境到初始狀態
        
        Parameters:
            seed (int): 隨機種子
            options (dict): 額外選項
            
        Returns:
            tuple: (觀察狀態, 資訊字典)
        """
        super().reset(seed=seed) 
        
        # 隨機選擇一個地圖策略
        # 權重可調整：50% 安全島，30% 繁忙交通，20% 高速公路
        selected_strategy = random.choice([
            cr.SafeIslandMap(), 
            cr.HeavyTrafficMap(), 
            cr.HighwayMap()
        ])
        
        # 使用選定的策略重置遊戲
        self.crossy_road.reset(seed=seed, map_strategy=selected_strategy)
        
        # 重置步數計數器
        self.current_step = 0
        
        # 取得初始觀察狀態
        obs = self._get_obs()
        info = {}
        
        # 若為人類模式，渲染畫面
        if self.render_mode == 'human':
            self.render()
            
        return obs, info

    def step(self, action):
        """
        執行一個動作並更新環境狀態
        
        Parameters:
            action (int): 動作索引 (0-3，對應上下左右)
            
        Returns:
            tuple: (觀察狀態, 獎勵, 是否終止, 是否截斷, 資訊字典)
        """
        # 執行行人動作
        target_reached = self.crossy_road.perform_action(cr.PedestrianAction(action))

        # 優先檢查：若到達終點，立即結束並給予獎勵
        if target_reached:
            reward = 100
            terminated = True
            truncated = False
            info = {}
            obs = self._get_obs()
            return obs, reward, terminated, truncated, info

        # 初始化回傳值
        reward = 0
        terminated = False
        truncated = False
        
        # 步數計數
        self.current_step += 1
        
        # 移動所有車輛並更新位置
        self.crossy_road.car_positions = []
        
        for i, car in enumerate(self.crossy_road.cars):
            car.move()
            
            # 若車輛離開畫面，隨機生成新車輛
            if car.is_out_of_bounds(self.crossy_road.grid_cols):
                # 保留原本的行數和方向
                row = car.row
                direction = car.direction
                
                # 隨機選擇新車種
                VehicleClass = random.choice([cr.Bus, cr.Taxi, cr.Bike, cr.SportCar])
                
                # 建立新車並替換
                new_car = VehicleClass(row, direction, grid_cols=self.crossy_road.grid_cols)
                self.crossy_road.cars[i] = new_car
                car = new_car
            
            # 更新車輛佔用位置
            current_car_pixels = car.get_occupied_pos()
            for pos in current_car_pixels:
                self.crossy_road.car_positions.append(pos)
            
            # 碰撞檢測
            if self.crossy_road.pedestrian_pos in current_car_pixels:
                reward -= 200
                terminated = True

        # 時間懲罰：鼓勵快速通過
        reward -= 0.01

        # 超時檢測
        if self.current_step >= self.max_episode_steps:
            truncated = True
            
        # 取得新的觀察狀態
        obs = self._get_obs()
        info = {}
        
        # 若為人類模式，渲染畫面
        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        """渲染遊戲畫面"""
        self.crossy_road.render()

# ===========================
# Unit Testing
# ===========================

if __name__ == "__main__":
    # 建立環境實例
    env = gym.make('crossy-road-v0', render_mode='human')
    
    # 環境檢查（取消註解以啟用）
    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")

    # 重置環境
    obs = env.reset()[0]

    # 執行隨機動作測試
    while True:
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)

        # 若遊戲結束，重置環境
        if terminated:
            obs = env.reset()[0]
