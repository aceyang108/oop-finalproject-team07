import random
from enum import Enum
import pygame
import sys
from os import path

# Actions the pedestrian is capable of performing i.e. go in a certain direction
class PedestrianAction(Enum):
    LEFT=0
    DOWN=1
    RIGHT=2
    UP=3

# The Warehouse is divided into a grid. Use these 'tiles' to represent the objects on the grid.
class GridTile(Enum):
    _FLOOR=0
    _ROAD=1
    PEDESTRIAN=2
    GOAL=3
    CAR=4
    X_CRASH=5

    # Return the first letter of tile name, for printing to the console.
    def __str__(self):
        return self.name[:1]

# Car class representing each car on the road
# [修改 1] 定義交通工具父類別
class Vehicle:
    # [修改] 新增 grid_cols 參數，預設為 9 (或你當前的寬度)
    def __init__(self, row, direction, grid_cols, speed=1, length=2, type_name="Taxi"):
        self.row = row
        self.direction = direction
        self.speed = speed
        self.length = length
        self.type_name = type_name
        
        # [關鍵修改] 動態計算重生點，確保永遠在畫面外
        if direction == 'L':
            # 向左開的車，重生在右邊界之外 (例如寬度9，重生在 9, 10, 11...)
            self.starting_pos = [grid_cols + i for i in range(5)]
        else:
            # 向右開的車，重生在左邊界之外 (負數)
            self.starting_pos = [-2 - i for i in range(5)]

        self.reset()



    def reset(self, seed=None):
        # 只需要做這件事：隨機選一個重生點
        import random
        self.column = random.choice(self.starting_pos)
    def move(self):
        # 根據速度與方向移動
        if self.direction == 'L':
            self.column -= self.speed
        else:
            self.column += self.speed

    def is_out_of_bounds(self, grid_cols):
        # 判斷是否跑出邊界 (考慮車身長度)
        if self.direction == 'L':
            # 往左走，車頭(column) + 車身長度 都跑出去了才算出去
            return self.column + self.length < -2 
        else:   
            return self.column > grid_cols + 2
    
    def get_front_pos(self):
        return [self.row, self.column]
    
    # [關鍵邏輯] 根據長度動態回傳佔用位置
    def get_occupied_pos(self):
        positions = []
        for i in range(self.length):
            if self.direction == 'L':
                # 向左開：車頭在 column，車身向右延伸 (column+1, column+2...)
                positions.append([self.row, self.column + i])
            else:
                # 向右開：車頭在 column，車身向左延伸 (column-1, column-2...)
                positions.append([self.row, self.column - i])
        return positions

class Bus(Vehicle):
    def __init__(self, row, direction, grid_cols):
        super().__init__(row, direction, grid_cols, speed=1, length=3, type_name="Bus")
    def get_type_name(self): return 'Bus'

class Taxi(Vehicle):
    def __init__(self, row, direction, grid_cols):
        super().__init__(row, direction, grid_cols, speed=1, length=2, type_name="Taxi")
    def get_type_name(self): return 'Taxi'

class Bike(Vehicle):
    def __init__(self, row, direction, grid_cols):
        super().__init__(row, direction, grid_cols, speed=2, length=1, type_name="Bike")
    def get_type_name(self): return 'Bike'

class SportCar(Vehicle):
    def __init__(self, row, direction, grid_cols):
        super().__init__(row, direction, grid_cols, speed=2, length=2, type_name="SportCar")
    def get_type_name(self): return 'SportCar'

class CrossyRoad:

    def __init__(self, grid_rows=10, grid_cols=5, fps=2, road_count=3, rand_seed=None):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.road_count = road_count    # Number of roads in the environment
        self.reset(rand_seed)

        self.fps = fps
        self.last_action = ''
        self._init_pygame()

    def _init_pygame(self):
        pygame.init() # initialize pygame
        pygame.display.init() # Initialize the display module

        # Game clock
        self.clock = pygame.time.Clock()

        # Default font
        self.action_font = pygame.font.SysFont("Calibre",30)
        self.action_info_height = self.action_font.get_height()

        # For rendering
        self.cell_height = 64
        self.cell_width = 64
        self.cell_size = (self.cell_width, self.cell_height)        

        # Define game window size (width, height)
        self.window_size = (self.cell_width * self.grid_cols, self.cell_height * self.grid_rows + self.action_info_height)

        # Initialize game window
        self.window_surface = pygame.display.set_mode(self.window_size) 

        # Load & resize sprites
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

        self.vehicle_sprites = {}
        
        # 對應到我們等一下要在 Vehicle 定義的 type 名稱
        self.vehicle_sprites['Bus'] = self._load_and_split_vehicle("bus.png")    # 預期切成 3 張
        self.vehicle_sprites['Taxi'] = self._load_and_split_vehicle("taxi.png")  # 預期切成 2 張
        self.vehicle_sprites['Bike'] = self._load_and_split_vehicle("bike.png")  # 預期切成 1 張
        self.vehicle_sprites['SportCar'] = self._load_and_split_vehicle("car.png") # 預期切成 2 張

    # Reset the environment
    def reset(self, seed=None):
        self.pedestrian_pos = [self.grid_rows-1, self.grid_cols//2]
        self.goal_row = 0

        random.seed(seed)
        self.road_rows = []
        self.cars = []
        
        # --- [修正] 強制安全島生成邏輯 ---
        current_roads = 0
        
        # 從第 1 列遍歷到倒數第 2 列
        for r in range(1, self.grid_rows - 1):
            # 規則：如果已經連續 2 條路，這行強制變成草地 (安全島)
            if current_roads >= 2:
                is_road = False
                current_roads = 0
            else:
                # 70% 機率生成路
                is_road = random.random() < 0.7
                if is_road:
                    current_roads += 1
                else:
                    current_roads = 0
            
            if is_road:
                direction = random.choice(['L', 'R'])
                self.road_rows.append((r, direction))

                # 隨機選擇車種 (傳入 grid_cols 以確保從畫面外生成)
                VehicleClass = random.choice([Bus, Taxi, Bike, SportCar])
                self.cars.append(VehicleClass(r, direction, grid_cols=self.grid_cols))
        
        # 更新實際的路數量
        self.road_count = len(self.road_rows)
        
        self.car_positions = [] 
        for car in self.cars:
            car.reset()
            for pos in car.get_occupied_pos():
                self.car_positions.append(pos)
    def perform_action(self, pedestrian_action:PedestrianAction) -> bool:
        self.last_action = pedestrian_action

        # Current pos
        x, y = self.pedestrian_pos

        # Move Pedestrian to the next cell
        # If pedesrian trying to bump into the side of a car (or border), they will not move
        if pedestrian_action == PedestrianAction.LEFT:
            if self.pedestrian_pos[1]>0 and ([x, y-1] not in self.car_positions):
                self.pedestrian_pos[1]-=1
        elif pedestrian_action == PedestrianAction.RIGHT:
            if self.pedestrian_pos[1]<self.grid_cols-1 and ([x, y+1] not in self.car_positions):
                self.pedestrian_pos[1]+=1
        elif pedestrian_action == PedestrianAction.UP:
            if self.pedestrian_pos[0]>0 and ([x-1, y] not in self.car_positions):
                self.pedestrian_pos[0]-=1
        elif pedestrian_action == PedestrianAction.DOWN:
            if self.pedestrian_pos[0]<self.grid_rows-1 and ([x+1, y] not in self.car_positions):
                self.pedestrian_pos[0]+=1

        # Return true if Pedestrian reaches Goal
        return self.pedestrian_pos[0] == self.goal_row

    def render(self):
        self._process_events()

        # clear to white background
        self.window_surface.fill((255,255,255))

        # Print current state on console AND Draw sprites
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                
                # --- 定義繪圖座標 ---
                pos = (c * self.cell_width, r * self.cell_height)

                # --- 1. 繪製終點 ---
                if r == self.goal_row:
                    self.window_surface.blit(self.goal_img, pos)
                    # print(GridTile.GOAL, end=' ') # 簡化輸出，不再 print 文字

                # --- 2. 繪製道路與車輛 ---
                elif r in [row for row, _ in self.road_rows]:
                    # 先畫馬路地板
                    self.window_surface.blit(self.road_img, pos)

                    # 如果這格有車，畫車
                    if [r,c] in self.car_positions:
                        # 找出這格是哪一台車
                        current_car = None
                        for car in self.cars:
                            if car.row == r:
                                current_car = car
                                break
                        
                        if current_car:
                            # 1. 計算這格是這台車的第幾個部分 (0=車頭, 1=車身1, 2=車身2...)
                            # 因為車子是連續佔據格子的，我們可以用座標差來算
                            segment_index = abs(c - current_car.column)
                            
                            # 2. 取得對應車種的圖片列表
                            # 例如 sprites['Bus']['L'] 會拿到 [頭圖, 身圖, 尾圖]
                            car_images = self.vehicle_sprites[current_car.type_name][current_car.direction]
                            
                            # 3. 安全檢查：確保 index 不會超過圖片數量 (例如長度3的車要有3張圖)
                            if segment_index < len(car_images):
                                self.window_surface.blit(car_images[segment_index], pos)
                            else:
                                # 如果圖片不夠 (例如拿 Bike 的圖去畫 Bus)，就重複畫最後一張當作備用
                                self.window_surface.blit(car_images[-1], pos)
                # --- 3. 繪製地板 (非道路) ---
                else:
                    self.window_surface.blit(self.floor_img, pos)

                # --- 4. 繪製行人 ---
                if([r,c] == self.pedestrian_pos):
                    self.window_surface.blit(self.pedestrian_img, pos)
                
        # Display last action info
        text_img = self.action_font.render(f'Action: {self.last_action}', True, (0,0,0), (255,255,255))
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos)       

        pygame.display.update()
        self.clock.tick(self.fps)
        
    def _process_events(self):
        # Process user events, key presses
        for event in pygame.event.get():
            # User clicked on X at the top right corner of window
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if(event.type == pygame.KEYDOWN):
                # User hit escape
                if(event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
    def _load_and_split_vehicle(self, filename):
        full_path = path.join(path.dirname(__file__), f"sprites/{filename}")
        full_img = pygame.image.load(full_path)
        
        # 取得圖片寬度，計算可以切成幾塊 (例如 192 / 64 = 3 塊)
        num_chunks = full_img.get_width() // self.cell_width
        
        images_left = []
        images_right = []
        
        for i in range(num_chunks):
            # 1. 切分出 64x64 的區域 (subsurface)
            # 假設原始圖片是向左的，且車頭在最左邊 (index 0)
            rect = pygame.Rect(i * self.cell_width, 0, self.cell_width, self.cell_height)
            chunk = full_img.subsurface(rect)
            
            # 縮放以防萬一 (雖然你的圖應該已經是 64x64 倍數)
            chunk = pygame.transform.scale(chunk, self.cell_size)
            
            images_left.append(chunk)
            
            # 2. 製作向右的版本 (水平翻轉)
            # 注意：向右時，車頭應該還是要用「車頭的圖(翻轉版)」，所以我們單獨翻轉每一塊
            images_right.append(pygame.transform.flip(chunk, True, False))
            
        return {'L': images_left, 'R': images_right}  


# For unit testing
# crossy_road.py 的最後面

if __name__=="__main__":
    crossyRoad = CrossyRoad(road_count=7) # 測試多一點路
    crossyRoad.render()

    import time # 用來控制速度

    while(True):
        # 1. 隨機移動行人 (原本的邏輯)
        rand_action = random.choice(list(PedestrianAction))
        print(f"Pedestrian Action: {rand_action}")
        crossyRoad.perform_action(rand_action)

        # 2. [新增] 移動所有的車子 (模擬環境的行為)
        # 清空舊位置
        crossyRoad.car_positions = []
        
        for car in crossyRoad.cars:
            car.move() # 讓車子動起來！
            
            # 檢查是否跑出邊界，跑出去就重生
            if car.is_out_of_bounds(crossyRoad.grid_cols):
                car.reset()
            
            # 更新位置列表以便渲染
            for pos in car.get_occupied_pos():
                crossyRoad.car_positions.append(pos)

        # 3. 渲染畫面
        crossyRoad.render()
        
        # 4. 稍微暫停一下，不然跑太快看不清楚
        time.sleep(0.5)