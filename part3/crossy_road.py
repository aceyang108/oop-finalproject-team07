import pygame
import random
import sys
import os
from enum import Enum
from abc import ABC, abstractmethod

# ===========================
# 1. Config & Constants
# ===========================
class Config:
    """
    Configuration constants for the game.
    
    Attributes:
        FPS (int): Frames per second.
        CELL_SIZE (int): Pixel size of each grid cell.
        FONT (str): Font name for UI text.
        BASE_DIR (str): Base directory of the script.
        SPRITE_DIR (str): Directory containing image assets.
    """
    FPS = 30
    CELL_SIZE = 64
    FONT = "Calibre"
    BASE_DIR = os.path.dirname(__file__)
    SPRITE_DIR = os.path.join(BASE_DIR, "sprites")

class PedestrianAction(Enum):
    """Enumeration of possible actions for the pedestrian."""
    LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

class GridTile(Enum):
    """Enumeration of grid tile types for console rendering."""
    _FLOOR, _ROAD, PEDESTRIAN, GOAL, CAR, X_CRASH = 0, 1, 2, 3, 4, 5
    
    def __str__(self):
        return self.name[:1]

# ===========================
# 2. Asset Manager
# ===========================
class AssetManager:
    """
    Static class for managing game assets to ensure efficient loading and error handling.
    """
    _images = {}
    _vehicle_sprites = {}

    @staticmethod
    def get_image(filename):
        """
        Loads and caches a single image. Returns a placeholder if the file is missing.

        Args:
            filename (str): Name of the image file in the sprites directory.

        Returns:
            pygame.Surface: The loaded image or a placeholder surface.
        """
        if filename not in AssetManager._images:
            path = os.path.join(Config.SPRITE_DIR, filename)
            try:
                img = pygame.image.load(path)
                AssetManager._images[filename] = pygame.transform.scale(img, (Config.CELL_SIZE, Config.CELL_SIZE))
            except:
                s = pygame.Surface((Config.CELL_SIZE, Config.CELL_SIZE))
                s.fill((200, 200, 200))
                AssetManager._images[filename] = s
        return AssetManager._images[filename]

    @staticmethod
    def get_vehicle_sprites(filename):
        """
        Loads a spritesheet for a vehicle, splits it into frames, and generates directional variants.

        Args:
            filename (str): Name of the spritesheet file.

        Returns:
            dict: A dictionary containing lists of surfaces for 'L' (Left) and 'R' (Right) directions.
        """
        if filename not in AssetManager._vehicle_sprites:
            path = os.path.join(Config.SPRITE_DIR, filename)
            try:
                full_img = pygame.image.load(path)
                w, h = Config.CELL_SIZE, Config.CELL_SIZE
                cols = full_img.get_width() // w
                imgs_l = [pygame.transform.scale(full_img.subsurface((i*w, 0, w, h)), (w, h)) for i in range(cols)]
                imgs_r = [pygame.transform.flip(img, True, False) for img in imgs_l]
                AssetManager._vehicle_sprites[filename] = {'L': imgs_l, 'R': imgs_r}
            except:
                s = pygame.Surface((Config.CELL_SIZE, Config.CELL_SIZE))
                s.fill((255, 0, 0))
                AssetManager._vehicle_sprites[filename] = {'L': [s], 'R': [s]}
        return AssetManager._vehicle_sprites[filename]

# ===========================
# 3. Vehicle Hierarchy
# ===========================
class Vehicle(ABC):
    """
    Abstract base class representing a vehicle in the game.
    """
    def __init__(self, row, direction, grid_cols, speed, length, type_name, img_file):
        """
        Initializes a vehicle instance.

        Args:
            row (int): The grid row index where the vehicle is located.
            direction (str): 'L' for Left or 'R' for Right.
            grid_cols (int): Total number of columns in the grid.
            speed (float): Movement speed of the vehicle.
            length (int): Length of the vehicle in grid cells.
            type_name (str): Name of the vehicle type (e.g., 'Bus').
            img_file (str): Filename of the vehicle's sprite.
        """
        self.row = row
        self.direction = direction
        self.grid_cols = grid_cols
        self.speed = speed
        self.length = length
        self.type_name = type_name
        self.img_file = img_file
        self.reset()

    def reset(self):
        """
        Resets the vehicle's position to a random start point outside the visible grid.
        """
        # 變數名稱 column 必須保留以相容 my_env
        if self.direction == 'L':
            self.column = float(random.choice(range(self.grid_cols, self.grid_cols + 5)))
        else:
            self.column = float(random.choice(range(-5, -1)))

    def move(self):
        """Updates the vehicle's column position based on its speed and direction."""
        self.column += -self.speed if self.direction == 'L' else self.speed

    def is_out_of_bounds(self, grid_cols=None):
        """
        Checks if the vehicle has completely moved off the grid.

        Args:
            grid_cols (int, optional): Override for grid columns. Defaults to self.grid_cols.

        Returns:
            bool: True if the vehicle is out of bounds, False otherwise.
        """
        limit = grid_cols if grid_cols is not None else self.grid_cols
        if self.direction == 'L': return self.column + self.length < -2
        return self.column > limit + 2

    def get_occupied_pos(self):
        """
        Calculates all grid cells currently occupied by this vehicle.

        Returns:
            list: A list of [row, col] coordinates occupied by the vehicle.
        """
        head = int(self.column)
        positions = []
        for i in range(self.length):
            c = head + i if self.direction == 'L' else head - i
            positions.append([self.row, c])
        return positions

# 修正：這裡的參數名稱必須改為 grid_cols，才能接收關鍵字參數
class Bus(Vehicle):
    """Concrete Vehicle class representing a Bus (Slow, Long)."""
    def __init__(self, row, direction, grid_cols): 
        super().__init__(row, direction, grid_cols, 1, 3, "Bus", "bus.png")

class Taxi(Vehicle):
    """Concrete Vehicle class representing a Taxi (Normal speed, Normal length)."""
    def __init__(self, row, direction, grid_cols): 
        super().__init__(row, direction, grid_cols, 1, 2, "Taxi", "taxi.png")

class Bike(Vehicle):
    """Concrete Vehicle class representing a Bike (Fast, Short)."""
    def __init__(self, row, direction, grid_cols): 
        super().__init__(row, direction, grid_cols, 2, 1, "Bike", "bike.png")

class SportCar(Vehicle):
    """Concrete Vehicle class representing a SportCar (Fast, Normal length)."""
    def __init__(self, row, direction, grid_cols): 
        super().__init__(row, direction, grid_cols, 2, 2, "SportCar", "car.png")

# ===========================
# 4. Strategy Pattern
# ===========================
class MapStrategy(ABC):
    """Abstract base class for map generation strategies."""
    @abstractmethod
    def generate(self, rows, cols):
        """
        Generates road layout and initial vehicles.

        Args:
            rows (int): Number of grid rows.
            cols (int): Number of grid columns.

        Returns:
            tuple: (list of road rows, list of Vehicle objects)
        """
        pass

class SafeIslandMap(MapStrategy):
    """Map strategy that includes safe islands between roads."""
    def generate(self, rows, cols):
        road_rows, cars = [], []
        consecutive = 0
        for r in range(1, rows - 1):
            if consecutive < 2 and random.random() < 0.7:
                d = random.choice(['L', 'R'])
                road_rows.append((r, d))
                # 這裡也要傳入 grid_cols=cols 以保持一致性，雖然這裡是位置參數調用
                cars.append(random.choice([Bus, Taxi, Bike, SportCar])(r, d, cols))
                consecutive += 1
            else: consecutive = 0
        return road_rows, cars

class HighwayMap(MapStrategy):
    """Map strategy with high density of fast vehicles and few safe islands."""
    def generate(self, rows, cols):
        road_rows, cars = [], []
        for r in range(1, rows - 1):
            if random.random() < 0.8:
                d = random.choice(['L', 'R'])
                road_rows.append((r, d))
                cars.append(random.choice([SportCar, Bike])(r, d, cols))
        return road_rows, cars

class HeavyTrafficMap(MapStrategy):
    """Map strategy with high density of slow, large vehicles."""
    def generate(self, rows, cols):
        road_rows, cars = [], []
        for r in range(1, rows - 1):
            if random.random() < 0.6:
                d = random.choice(['L', 'R'])
                road_rows.append((r, d))
                cars.append(random.choice([Bus, Taxi])(r, d, cols))
        return road_rows, cars

# ===========================
# 5. Game Engine
# ===========================
class CrossyRoad:
    """
    Main game engine class.
    Manages the grid state, vehicle entities, pedestrian physics, and rendering loop.
    """
    def __init__(self, grid_rows=10, grid_cols=5, fps=30, road_count=3, rand_seed=None):
        """
        Initializes the game engine.

        Args:
            grid_rows (int): Number of rows in the grid.
            grid_cols (int): Number of columns in the grid.
            fps (int): Target frames per second.
            road_count (int): (Deprecated) Approximate number of roads, handled by strategy now.
            rand_seed (int, optional): Seed for random number generator.
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.fps = fps
        self.last_action = ''
        
        pygame.init()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(Config.FONT, 30)
        
        w, h = Config.CELL_SIZE * grid_cols, Config.CELL_SIZE * grid_rows + 30
        self.screen = pygame.display.set_mode((w, h))

        self.reset(rand_seed)

    def reset(self, seed=None, map_strategy=None):
        """
        Resets the game to its initial state with a new map layout.

        Args:
            seed (int, optional): Random seed for reproducibility.
            map_strategy (MapStrategy, optional): Strategy object for map generation. 
                                                  Defaults to SafeIslandMap.
        """
        if seed: random.seed(seed)
        self.pedestrian_pos = [self.grid_rows - 1, self.grid_cols // 2]
        self.goal_row = 0
        
        strategy = map_strategy if map_strategy else SafeIslandMap()
        self.road_rows, self.cars = strategy.generate(self.grid_rows, self.grid_cols)
        self.road_count = len(self.road_rows)
        self.car_positions = []
        self.update_vehicles()

    def update_vehicles(self):
        """
        Moves all vehicles, handles out-of-bounds respawning, and updates collision positions.
        """
        self.car_positions = []
        for i, car in enumerate(self.cars):
            car.move()
            # is_out_of_bounds 支援 my_env 傳入參數
            if car.is_out_of_bounds():
                NewClass = random.choice([Bus, Taxi, Bike, SportCar])
                # 這裡模擬 my_env 的行為，使用 grid_cols 參數
                self.cars[i] = NewClass(car.row, car.direction, grid_cols=self.grid_cols)
            
            self.car_positions.extend(self.cars[i].get_occupied_pos())

    def perform_action(self, action: PedestrianAction) -> bool:
        """
        Executes a pedestrian move action and checks for collisions or goal achievement.

        Args:
            action (PedestrianAction): The direction to move.

        Returns:
            bool: True if the pedestrian reached the goal row, False otherwise.
        """
        self.last_action = action
        r, c = self.pedestrian_pos
        nr, nc = r, c

        if action == PedestrianAction.LEFT: nc -= 1
        elif action == PedestrianAction.RIGHT: nc += 1
        elif action == PedestrianAction.UP: nr -= 1
        elif action == PedestrianAction.DOWN: nr += 1
        
        if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols and [nr, nc] not in self.car_positions:
            self.pedestrian_pos = [nr, nc]
            
        return self.pedestrian_pos[0] == self.goal_row

    def render(self):
        """
        Renders the current game state to the console (text) and the Pygame window (graphics).
        """
        road_indices = [row for row, _ in self.road_rows]
        
        # 1. Console Output
        for r in range(self.grid_rows):
            line = ""
            for c in range(self.grid_cols):
                tile = GridTile._FLOOR
                if [r, c] == self.pedestrian_pos:
                    tile = GridTile.X_CRASH if [r, c] in self.car_positions else GridTile.PEDESTRIAN
                elif r == self.goal_row: tile = GridTile.GOAL
                elif [r, c] in self.car_positions: tile = GridTile.CAR
                elif r in road_indices: tile = GridTile._ROAD
                line += str(tile) + " "
            print(line)
        print()

        # 2. Pygame Output
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); sys.exit()

        self.screen.fill((255, 255, 255))
        
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                pos = (c * Config.CELL_SIZE, r * Config.CELL_SIZE)
                if r == self.goal_row: self.screen.blit(AssetManager.get_image("goal.jpg"), pos)
                elif r in road_indices: self.screen.blit(AssetManager.get_image("road.png"), pos)
                else: self.screen.blit(AssetManager.get_image("floor.png"), pos)

        for car in self.cars:
            sprites = AssetManager.get_vehicle_sprites(car.img_file)[car.direction]
            idx = abs(int(car.column)) % len(sprites)
            px, py = int(car.column * Config.CELL_SIZE), car.row * Config.CELL_SIZE
            self.screen.blit(sprites[0], (px, py))

        pr, pc = self.pedestrian_pos
        self.screen.blit(AssetManager.get_image("pedestrian.png"), (pc * Config.CELL_SIZE, pr * Config.CELL_SIZE))

        txt = self.font.render(f'Action: {self.last_action}', True, (0,0,0), (255,255,255))
        self.screen.blit(txt, (0, self.grid_rows * Config.CELL_SIZE))

        pygame.display.update()
        self.clock.tick(self.fps)

if __name__ == "__main__":
    game = CrossyRoad(road_count=7)
    import time
    while True:
        action = random.choice(list(PedestrianAction))
        print(f"Pedestrian Action: {action}")
        game.perform_action(action)
        game.update_vehicles()
        game.render()
        time.sleep(0.5)