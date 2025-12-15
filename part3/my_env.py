import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import crossy_road as cr
import numpy as np
import random

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='crossy-road-v0',                                # call it whatever you want
    entry_point='my_env:CrossyRoadEnv', # module_name:class_name
)

# Implement our own gym env, must inherit from gym.Env
# https://gymnasium.farama.org/api/env/
class CrossyRoadEnv(gym.Env):
    # metadata is a required attribute
    # render_modes in our environment is either None or 'human'.
    # render_fps is not used in our env, but we are require to declare a non-zero value.
    metadata = {"render_modes": ["human"], 'render_fps': 2}

    def __init__(self, grid_rows=10, grid_cols=11, render_mode=None):

        self.grid_rows=grid_rows
        self.grid_cols=grid_cols
        self.render_mode = render_mode

        # Initialize the CrossyRoad problem
        # ✅ 必須明確傳入 grid_rows, grid_cols 等參數
        self.crossy_road = cr.CrossyRoad(
            grid_rows=grid_rows, 
            grid_cols=grid_cols, 
            fps=self.metadata['render_fps'], 
            road_count=6, 
            rand_seed=None
        )
        self.action_space = spaces.Discrete(len(cr.PedestrianAction))
        
        self.max_roads = 10 
        self.max_cars = 20
        self.max_episode_steps = 200
        
        # 【修改點】每台車的特徵從 3 個變成了 5 個 (Row, Col, Dir, Speed, Length)
        # 公式：4 + (路數 * 2) + (車數 * 5)
        self.max_obs_len = 4 + (self.max_roads * 2) + (self.max_cars * 5)
        
        self.observation_space = spaces.Box(
            low=-999,
            high=999,
            shape=(self.max_obs_len,),
            dtype=np.int32
        )

    # Get the observation state:
    # [pedestrian_row_pos, pedestrian_col_pos, goal_row, road_count, road_rows, road_direction (for each road), car_row_pos, car_col_pos, car_direction (for each car)]
    def _get_obs(self):
        # 1. 收集當前所有資訊
        numeric_data_list = list(self.crossy_road.pedestrian_pos) + [self.crossy_road.goal_row, self.crossy_road.road_count]
        
        for road in self.crossy_road.road_rows[:10]:
            dir_num = 0 if road[1] == 'L' else 1
            numeric_data_list.extend([road[0], dir_num])

        # 【修改點】加入車子的速度與長度資訊
        for car in self.crossy_road.cars[:20]:
            dir_num = 0 if car.direction == 'L' else 1
            
            # 加入 5 個特徵：行, 列, 方向, 速度, 長度
            numeric_data_list.extend([
                car.row, 
                car.column, 
                dir_num,
                car.speed,  # 新增：讓 AI 知道這台車有多快
                car.length  # 新增：讓 AI 知道這台車有多長
            ])

        # 2. 轉成 numpy array
        current_data = np.array(numeric_data_list, dtype=np.int32)
        
        # 3. 補零 (Padding) - 注意這裡會自動使用上面更新過的 self.max_obs_len
        obs = np.zeros(self.max_obs_len, dtype=np.int32)
        length = min(len(current_data), self.max_obs_len)
        obs[:length] = current_data[:length]
        
        return obs

    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        # 呼叫 crossy_road.py 裡面的重置邏輯
        self.crossy_road.reset(seed=seed)
        
        # 初始化計步器
        self.current_step = 0 

        obs = self._get_obs()
        info = {}
        
        if(self.render_mode=='human'):
            self.render()

        return obs, info
    # Gym required function (and parameters) to perform an action
    def step(self, action):
        # 1. 執行動作
        target_reached = self.crossy_road.perform_action(cr.PedestrianAction(action))

        # 2. [關鍵修改] 最高優先級檢查：如果贏了，直接結束！
        # 不要再讓車子移動，也不要再檢查碰撞，直接發獎勵並回傳。
        if target_reached:
            reward = 100
            terminated = True
            truncated = False
            info = {}
            # print("Goal Reached!")
            
            # 必須在這裡直接 return obs (雖然車沒動，但反正遊戲結束了)
            # 這裡回傳當前畫面即可
            obs = self._get_obs()
            return obs, reward, terminated, truncated, info

        # --- 以下是沒贏的情況，才需要計算車子移動和碰撞 ---
        
        reward = 0
        terminated = False
        truncated = False
        
        # 3. 步數計數
        self.current_step += 1
        
        # 4. 移動車子 & 碰撞檢測 (保持原本邏輯)
        self.crossy_road.car_positions = [] 
        for i, car in enumerate(self.crossy_road.cars):
            car.move()
            
            # [修改] 如果車子跑出邊界，不僅重置位置，還 "換一台新車"
            if car.is_out_of_bounds(self.crossy_road.grid_cols):
                # 1. 取得舊車的資訊 (保留原本的行數和方向)
                row = car.row
                direction = car.direction
                
                # 2. 隨機選一個新車種 (需要引入 random 和 crossy_road classes)
                import random
                # 這裡需要引用 crossy_road 裡的類別
                VehicleClass = random.choice([cr.Bus, cr.Taxi, cr.Bike, cr.SportCar])
                
                # 3. 建立新車 (記得傳入 grid_cols)
                new_car = VehicleClass(row, direction, grid_cols=self.crossy_road.grid_cols)
                self.crossy_road.cars[i] = new_car
                car = new_car
                # 4. 替換掉列表中的舊車
                self.crossy_road.cars[i] = new_car
                
                # 讓這台新車也加入當前的渲染位置 (避免剛生成閃爍)
                car = new_car
            current_car_pixels = car.get_occupied_pos()
            for pos in current_car_pixels:
                self.crossy_road.car_positions.append(pos)
            
            # 碰撞檢測
            if self.crossy_road.pedestrian_pos in current_car_pixels:
                reward -= 200 
                terminated = True
                # print("Crashed!")

        # 5. 時間懲罰 (保持原本邏輯)
        reward -= 0.01

        # 6. 超時檢測 (保持原本邏輯)
        if self.current_step >= self.max_episode_steps:
            truncated = True
            
        # 7. 回傳
        obs = self._get_obs()
        info = {}
        if(self.render_mode=='human'):
            self.render()

        return obs, reward, terminated, truncated, info
    # Gym required function to render environment
    def render(self):
        self.crossy_road.render()

# For unit testing
if __name__=="__main__":
    env = gym.make('crossy-road-v0', render_mode='human')
    # Use this to check our custom environment
    # print("Check environment begin")
    # check_env(env.unwrapped)
    # print("Check environment end")

    # Reset environment
    obs = env.reset()[0]

    # Take some random actions
    while(True):
        rand_action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(rand_action)

        if(terminated):
            obs = env.reset()[0]
