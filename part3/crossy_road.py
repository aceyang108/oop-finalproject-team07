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
    
class Car:
    def __init__(self, row, direction):
        self.row = row
        self.direction = direction  # 'L' for left, 'R' for right

        if direction == 'L':
            self.starting_pos = [5, 6, 7]
        else:
            self.starting_pos = [-1, -2, -3]

        self.column = random.choice(self.starting_pos)

    def reset(self):
        self.column = random.choice(self.starting_pos)

    def move(self):
        if self.direction == 'L':
            self.column -= 1
        else:
            self.column += 1

    def is_out_of_bounds(self, grid_cols):
        if self.direction == 'L':
            return self.column < -1
        else:   
            return self.column > grid_cols
        
    def get_head_pos(self):
        return [self.row, self.column]
    
    def get_occupied_pos(self):
        if self.direction == 'L':
            return [[self.row, self.column], [self.row, self.column + 1]]
        else:
            return [[self.row, self.column], [self.row, self.column - 1]]

class CrossyRoad:

    def __init__(self, grid_rows=10, grid_cols=5, fps=2, road_count=3):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.road_count = road_count
        self.reset()

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

        file_name = path.join(path.dirname(__file__), "sprites/car_left_front.png")
        img = pygame.image.load(file_name)
        self.car_left_front_img = pygame.transform.scale(img, self.cell_size) 

        file_name = path.join(path.dirname(__file__), "sprites/car_left_back.png")
        img = pygame.image.load(file_name)
        self.car_left_back_img = pygame.transform.scale(img, self.cell_size) 
        
        file_name = path.join(path.dirname(__file__), "sprites/car_right_front.png")
        img = pygame.image.load(file_name)
        self.car_right_front_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/car_right_back.png")
        img = pygame.image.load(file_name)
        self.car_right_back_img = pygame.transform.scale(img, self.cell_size)

    def reset(self, seed=None):
        # Initialize Pedestrian's starting position
        self.pedestrian_pos = [self.grid_rows-1, self.grid_cols//2]
        # Initialize the special rows
        self.goal_row = 0
        random.seed(seed)
        self.road_rows = []
        self.cars = []
        selected_rows = set()
        for _ in range(self.road_count):
            row = random.randint(1, self.grid_rows-2)
            while row in selected_rows:
                row = random.randint(1, self.grid_rows-2)
            selected_rows.add(row)
            direction = random.choice(['L', 'R'])
            self.road_rows.append((row, direction))
            self.cars.append(Car(row, direction))
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
        # Print current state on console
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):

                if([r,c] == self.pedestrian_pos):
                    if [r,c] in self.car_positions:
                        print(GridTile.X_CRASH, end=' ')
                    else:
                        print(GridTile.PEDESTRIAN, end=' ')
                elif(r == self.goal_row):
                    print(GridTile.GOAL, end=' ')
                elif r in [row for row, _ in self.road_rows]:
                    if [r,c] in self.car_positions:
                        print(GridTile.CAR, end=' ')
                    else:
                        print(GridTile._ROAD, end=' ')
                else:
                    print(GridTile._FLOOR, end=' ')

            print() # new line
        print() # new line

        self._process_events()

        # clear to white background, otherwise text with varying length will leave behind prior rendered portions
        self.window_surface.fill((255,255,255))

        # Print current state on console
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                
                if r == self.goal_row:
                    # Draw goal
                    pos = (c * self.cell_width, r * self.cell_height)
                    self.window_surface.blit(self.goal_img, pos)
                elif r in [row for row, _ in self.road_rows]:
                    # Draw road
                    pos = (c * self.cell_width, r * self.cell_height)
                    self.window_surface.blit(self.road_img, pos)

                    if [r,c] in self.car_positions:
                        # Draw car
                        direction = [dir for row, dir in self.road_rows if row == r][0]
                        if direction == 'L':
                            if ([r,c-1] in self.car_positions):
                                self.window_surface.blit(self.car_left_back_img, pos)
                            else:
                                self.window_surface.blit(self.car_left_front_img, pos)
                        else:
                            if ([r,c+1] in self.car_positions):
                                self.window_surface.blit(self.car_right_back_img, pos)
                            else:
                                self.window_surface.blit(self.car_right_front_img, pos)
                else:
                    # Draw floor
                    pos = (c * self.cell_width, r * self.cell_height)
                    self.window_surface.blit(self.floor_img, pos)

                if([r,c] == self.pedestrian_pos):
                    # Draw pedestrian
                    pos = (c * self.cell_width, r * self.cell_height)
                    self.window_surface.blit(self.pedestrian_img, pos)
                
        text_img = self.action_font.render(f'Action: {self.last_action}', True, (0,0,0), (255,255,255))
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos)       

        pygame.display.update()
                
        # Limit frames per second
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
                


# For unit testing
if __name__=="__main__":
    crossyRoad = CrossyRoad()
    crossyRoad.render()

    while(True):
        rand_action = random.choice(list(PedestrianAction))
        print(rand_action)

        crossyRoad.perform_action(rand_action)
        crossyRoad.render()