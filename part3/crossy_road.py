import random
from enum import Enum
import pygame
import sys
from os import path

# ===========================
# Enumerations
# ===========================

# 定義行人可執行的動作方向
class PedestrianAction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

# 定義網格上的物件類型
class GridTile(Enum):
    _FLOOR = 0
    _ROAD = 1
    PEDESTRIAN = 2
    GOAL = 3
    CAR = 4
    X_CRASH = 5

    def __str__(self):
        # 回傳物件名稱的第一個字母，用於 console 顯示
        return self.name[:1]

# ===========================
# Vehicle Classes
# ===========================

# 交通工具父類別，定義所有車輛的共同屬性與行為
class Vehicle:
    def __init__(self, row, direction, grid_cols, speed=1, length=2, type_name="Taxi"):
        """
        初始化交通工具
        
        Parameters:
            row (int): 車輛所在的列
            direction (str): 行駛方向 ('L' 或 'R')
            grid_cols (int): 網格的欄數
            speed (int): 移動速度
            length (int): 車身長度
            type_name (str): 車輛類型名稱
        """
        self.row = row
        self.direction = direction
        self.speed = speed
        self.length = length
        self.type_name = type_name
        
        # 動態計算重生點，確保永遠在畫面外
        if direction == 'L':
            # 向左行駛的車輛，重生在右邊界之外
            self.starting_pos = [grid_cols + i for i in range(5)]
        else:
            # 向右行駛的車輛,重生在左邊界之外
            self.starting_pos = [-2 - i for i in range(5)]

        self.reset()

    def reset(self, seed=None):
        """重置車輛位置到隨機重生點"""
        import random
        self.column = random.choice(self.starting_pos)
    
    def move(self):
        """根據速度與方向移動車輛"""
        if self.direction == 'L':
            self.column -= self.speed
        else:
            self.column += self.speed

    def is_out_of_bounds(self, grid_cols):
        """
        判斷車輛是否已完全離開畫面
        
        Parameters:
            grid_cols (int): 網格欄數
            
        Returns:
            bool: 是否已超出邊界
        """
        if self.direction == 'L':
            # 向左行駛：車尾完全離開左邊界
            return self.column + self.length < -2 
        else:
            # 向右行駛：車頭完全離開右邊界
            return self.column > grid_cols + 2
    
    def get_front_pos(self):
        """取得車頭位置"""
        return [self.row, self.column]
    
    def get_occupied_pos(self):
        """
        根據車身長度計算所有佔用的格子位置
        
        Returns:
            list: 所有被佔用的 [row, col] 位置列表
        """
        positions = []
        for i in range(self.length):
            if self.direction == 'L':
                # 向左行駛：車頭在 column，車身向右延伸
                positions.append([self.row, self.column + i])
            else:
                # 向右行駛：車頭在 column，車身向左延伸
                positions.append([self.row, self.column - i])
        return positions

# 公車類別：速度慢、車身長
class Bus(Vehicle):
    def __init__(self, row, direction, grid_cols):
        super().__init__(row, direction, grid_cols, speed=1, length=3, type_name="Bus")
    
    def get_type_name(self):
        return 'Bus'

# 計程車類別：標準速度、標準車身
class Taxi(Vehicle):
    def __init__(self, row, direction, grid_cols):
        super().__init__(row, direction, grid_cols, speed=1, length=2, type_name="Taxi")
    
    def get_type_name(self):
        return 'Taxi'

# 機車類別：速度快、車身短
class Bike(Vehicle):
    def __init__(self, row, direction, grid_cols):
        super().__init__(row, direction, grid_cols, speed=2, length=1, type_name="Bike")
    
    def get_type_name(self):
        return 'Bike'

# 跑車類別：速度快、標準車身
class SportCar(Vehicle):
    def __init__(self, row, direction, grid_cols):
        super().__init__(row, direction, grid_cols, speed=2, length=2, type_name="SportCar")
    
    def get_type_name(self):
        return 'SportCar'

# ===========================
# Main Game Class
# ===========================

class CrossyRoad:
    def __init__(self, grid_rows=10, grid_cols=5, fps=2, road_count=3, rand_seed=None):
        """
        初始化遊戲環境
        
        Parameters:
            grid_rows (int): 網格列數
            grid_cols (int): 網格欄數
            fps (int): 每秒幀數
            road_count (int): 道路數量
            rand_seed: 隨機種子
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.road_count = road_count
        self.reset(rand_seed)

        self.fps = fps
        self.last_action = ''
        self._init_pygame()

    def _init_pygame(self):
        """初始化 Pygame 視窗與載入所有圖片資源"""
        # 初始化 pygame 模組
        pygame.init()
        pygame.display.init()

        # 遊戲時鐘
        self.clock = pygame.time.Clock()

        # 字體設定
        self.action_font = pygame.font.SysFont("Calibre", 30)
        self.action_info_height = self.action_font.get_height()

        # 每個格子的尺寸設定
        self.cell_height = 64
        self.cell_width = 64
        self.cell_size = (self.cell_width, self.cell_height)        

        # 計算視窗大小
        self.window_size = (
            self.cell_width * self.grid_cols, 
            self.cell_height * self.grid_rows + self.action_info_height
        )

        # 建立遊戲視窗
        self.window_surface = pygame.display.set_mode(self.window_size) 

        # 載入並縮放基礎圖片
        file_name = path.join(path.dirname(__file__), "sprites/pedestrian.png")
        img = pygame.image.load(file_name)
        self.pedestrian_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/floor.png")
        img = pygame.image.load(file_name)
        self.floor_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/road.png")
        img = pygame.image.load(file_name)
        self.road_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/goal.jpg")
        img = pygame.image.load(file_name)
        self.goal_img = pygame.transform.scale(img, self.cell_size) 

        # 載入車輛圖片
        self.vehicle_sprites = {}
        self.vehicle_sprites['Bus'] = self._load_and_split_vehicle("bus.png")
        self.vehicle_sprites['Taxi'] = self._load_and_split_vehicle("taxi.png")
        self.vehicle_sprites['Bike'] = self._load_and_split_vehicle("bike.png")
        self.vehicle_sprites['SportCar'] = self._load_and_split_vehicle("car.png")

    def reset(self, seed=None, map_strategy=None):
        """
        重置遊戲環境
        
        Parameters:
            seed: 隨機種子
            map_strategy: 地圖生成策略，若無則使用預設的安全島模式
        """
        # 設定行人起始位置（底部中央）
        self.pedestrian_pos = [self.grid_rows - 1, self.grid_cols // 2]
        # 設定目標位置（頂部）
        self.goal_row = 0

        random.seed(seed)
        
        # 若未指定地圖策略，使用預設的安全島模式
        if map_strategy is None:
            map_strategy = SafeIslandMap()
            
        # 使用策略模式生成地圖
        self.road_rows, self.cars = map_strategy.generate(self.grid_rows, self.grid_cols)
        
        self.road_count = len(self.road_rows)
        
        # 初始化車輛位置列表
        self.car_positions = [] 
        for car in self.cars:
            car.reset()
            for pos in car.get_occupied_pos():
                self.car_positions.append(pos)
    
    def perform_action(self, pedestrian_action: PedestrianAction) -> bool:
        """
        執行行人動作
        
        Parameters:
            pedestrian_action (PedestrianAction): 行人移動方向
            
        Returns:
            bool: 是否到達終點
        """
        self.last_action = pedestrian_action

        # 取得目前位置
        x, y = self.pedestrian_pos

        # 根據動作移動行人
        # 若行人試圖撞向車輛側面或邊界，則不移動
        if pedestrian_action == PedestrianAction.LEFT:
            if self.pedestrian_pos[1] > 0 and ([x, y - 1] not in self.car_positions):
                self.pedestrian_pos[1] -= 1
        elif pedestrian_action == PedestrianAction.RIGHT:
            if self.pedestrian_pos[1] < self.grid_cols - 1 and ([x, y + 1] not in self.car_positions):
                self.pedestrian_pos[1] += 1
        elif pedestrian_action == PedestrianAction.UP:
            if self.pedestrian_pos[0] > 0 and ([x - 1, y] not in self.car_positions):
                self.pedestrian_pos[0] -= 1
        elif pedestrian_action == PedestrianAction.DOWN:
            if self.pedestrian_pos[0] < self.grid_rows - 1 and ([x + 1, y] not in self.car_positions):
                self.pedestrian_pos[0] += 1

        # 回傳是否到達終點
        return self.pedestrian_pos[0] == self.goal_row

    def render(self):
        """渲染遊戲畫面 (包含終端機文字輸出與 Pygame 視窗繪圖)"""
        self._process_events()

        # --- Part 1: 終端機文字輸出 (Console Output) ---        
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                
                # 1. 檢查是否是行人 (最優先顯示)
                if [r, c] == self.pedestrian_pos:
                    # 如果行人跟車子重疊 -> 撞車 (X)
                    if [r, c] in self.car_positions:
                        print(GridTile.X_CRASH, end=' ')
                    else:
                        print(GridTile.PEDESTRIAN, end=' ')
                
                # 2. 檢查是否是終點
                elif r == self.goal_row:
                    print(GridTile.GOAL, end=' ')
                
                # 3. 檢查是否是道路
                elif r in [row for row, _ in self.road_rows]:
                    # 如果這格有車 -> 顯示車 (C)
                    if [r, c] in self.car_positions:
                        print(GridTile.CAR, end=' ')
                    else:
                        print(GridTile._ROAD, end=' ')
                
                # 4. 其他就是地板
                else:
                    print(GridTile._FLOOR, end=' ')
            
            # 每一列印完換行
            print()
        
        # 整個畫面印完，多空一行分隔
        print()


        # --- Part 2: Pygame 視窗繪圖 (Graphical Output) ---        
        # 清空畫面為白色背景
        self.window_surface.fill((255, 255, 255))

        # 繪製所有格子
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                # 計算繪圖座標
                pos = (c * self.cell_width, r * self.cell_height)

                # 繪製終點
                if r == self.goal_row:
                    self.window_surface.blit(self.goal_img, pos)

                # 繪製道路與車輛
                elif r in [row for row, _ in self.road_rows]:
                    # 先繪製道路地板
                    self.window_surface.blit(self.road_img, pos)

                    # 若此格有車輛，繪製車輛
                    if [r, c] in self.car_positions:
                        # 找出該格所屬的車輛
                        current_car = None
                        for car in self.cars:
                            if car.row == r:
                                current_car = car
                                break
                        
                        if current_car:
                            # 計算此格是車輛的第幾個部分
                            segment_index = abs(c - current_car.column)
                            car_images = self.vehicle_sprites[current_car.type_name][current_car.direction]
                            
                            if segment_index < len(car_images):
                                self.window_surface.blit(car_images[segment_index], pos)
                            else:
                                self.window_surface.blit(car_images[-1], pos)
                
                # 繪製一般地板
                else:
                    self.window_surface.blit(self.floor_img, pos)

                # 繪製行人
                if [r, c] == self.pedestrian_pos:
                    self.window_surface.blit(self.pedestrian_img, pos)
                
        # 顯示最後執行的動作
        text_img = self.action_font.render(
            f'Action: {self.last_action}', 
            True, 
            (0, 0, 0), 
            (255, 255, 255)
        )
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos)       

        pygame.display.update()
        self.clock.tick(self.fps)
              
    def _process_events(self):
        """處理使用者事件（關閉視窗、按鍵等）"""
        for event in pygame.event.get():
            # 使用者點擊視窗右上角的 X
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # 使用者按下鍵盤
            if event.type == pygame.KEYDOWN:
                # 按下 ESC 鍵
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
    
    def _load_and_split_vehicle(self, filename):
        """
        載入並切分車輛圖片，生成左右兩個方向的圖片列表
        
        Parameters:
            filename (str): 圖片檔名
            
        Returns:
            dict: {'L': [左向圖片列表], 'R': [右向圖片列表]}
        """
        full_path = path.join(path.dirname(__file__), f"sprites/{filename}")
        full_img = pygame.image.load(full_path)
        
        # 計算可切分的塊數
        num_chunks = full_img.get_width() // self.cell_width
        
        images_left = []
        images_right = []
        
        for i in range(num_chunks):
            # 切分出單個格子的圖片
            rect = pygame.Rect(i * self.cell_width, 0, self.cell_width, self.cell_height)
            chunk = full_img.subsurface(rect)
            
            # 縮放至標準格子大小
            chunk = pygame.transform.scale(chunk, self.cell_size)
            
            images_left.append(chunk)
            
            # 製作水平翻轉的右向版本
            images_right.append(pygame.transform.flip(chunk, True, False))
            
        return {'L': images_left, 'R': images_right}

# ===========================
# Map Strategy Pattern
# ===========================

# 地圖生成策略的抽象父類別
class MapStrategy:
    def generate(self, grid_rows, grid_cols):
        """
        生成地圖（需由子類別實作）
        
        Parameters:
            grid_rows (int): 網格列數
            grid_cols (int): 網格欄數
            
        Returns:
            tuple: (道路列表, 車輛列表)
        """
        raise NotImplementedError

# 安全島模式：適合新手 AI，每連續兩條道路後會有安全島
class SafeIslandMap(MapStrategy):
    def generate(self, grid_rows, grid_cols):
        """生成具有安全島的地圖"""
        road_rows = []
        cars = []
        current_roads = 0
        
        for r in range(1, grid_rows - 1):
            # 連續兩條道路後強制生成安全島
            if current_roads >= 2:
                is_road = False
                current_roads = 0
            else:
                is_road = random.random() < 0.7
                if is_road:
                    current_roads += 1
                else:
                    current_roads = 0
            
            if is_road:
                direction = random.choice(['L', 'R'])
                road_rows.append((r, direction))
                # 隨機選擇車輛類型
                VehicleClass = random.choice([Bus, Taxi, Bike, SportCar])
                cars.append(VehicleClass(r, direction, grid_cols=grid_cols))
        
        return road_rows, cars

# 高速公路模式：道路寬廣、車速快、幾乎無安全島
class HighwayMap(MapStrategy):
    def generate(self, grid_rows, grid_cols):
        """生成高速公路模式的地圖"""
        road_rows = []
        cars = []
        
        for r in range(1, grid_rows - 1):
            # 80% 機率為道路（安全島少）
            is_road = random.random() < 0.8 
            
            if is_road:
                direction = random.choice(['L', 'R'])
                road_rows.append((r, direction))
                # 主要使用快速車輛
                VehicleClass = random.choice([SportCar, Bike]) 
                cars.append(VehicleClass(r, direction, grid_cols=grid_cols))
                
        return road_rows, cars

# 繁忙交通模式：車輛較多但速度較慢
class HeavyTrafficMap(MapStrategy):
    def generate(self, grid_rows, grid_cols):
        """生成繁忙交通模式的地圖"""
        road_rows = []
        cars = []
        
        for r in range(1, grid_rows - 1):
            # 60% 機率為道路
            if random.random() < 0.6:
                direction = random.choice(['L', 'R'])
                road_rows.append((r, direction))
                # 主要使用大型慢速車輛
                VehicleClass = random.choice([Bus, Taxi])
                cars.append(VehicleClass(r, direction, grid_cols=grid_cols))
                
        return road_rows, cars

# ===========================
# Main Entry Point
# ===========================

if __name__ == "__main__":
    # 建立遊戲實例
    crossyRoad = CrossyRoad(road_count=7)
    crossyRoad.render()

    import time

    while True:
        # 隨機移動行人
        rand_action = random.choice(list(PedestrianAction))
        print(f"Pedestrian Action: {rand_action}")
        crossyRoad.perform_action(rand_action)

        # 移動所有車輛
        crossyRoad.car_positions = []
        
        for car in crossyRoad.cars:
            car.move()
            
            # 檢查車輛是否超出邊界，若是則重生
            if car.is_out_of_bounds(crossyRoad.grid_cols):
                car.reset()
            
            # 更新車輛位置列表
            for pos in car.get_occupied_pos():
                crossyRoad.car_positions.append(pos)

        # 渲染畫面
        crossyRoad.render()
        
        # 控制執行速度
        time.sleep(0.5)