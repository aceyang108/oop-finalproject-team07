import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import crossy_road as cr
import numpy as np

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

    def __init__(self, grid_rows=10, grid_cols=5, render_mode=None):

        self.grid_rows=grid_rows
        self.grid_cols=grid_cols
        self.render_mode = render_mode

        # Initialize the CrossyRoad problem
        self.crossy_road = cr.CrossyRoad(grid_rows=grid_rows, grid_cols=grid_cols, fps=self.metadata['render_fps'], road_count=2, rand_seed=217201)

        # Gym requires defining the action space. The action space is robot's set of possible actions.
        # Training code can call action_space.sample() to randomly select an action. 
        self.action_space = spaces.Discrete(len(cr.PedestrianAction))
        # Gym requires defining the observation space. The observation space consists of the robot's and target's set of possible positions.
        # The observation space is used to validate the observation returned by reset() and step().
        # Use a 1D vector: [robot_row_pos, robot_col_pos, target_row_pos, target_col_pos]
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.grid_rows-1, self.grid_cols-1, self.grid_rows-1, self.grid_cols-1]),
            shape=(4,),
            dtype=np.int32
        )

    # Get the observation state:
    # [pedestrian_row_pos, pedestrian_col_pos, goal_row, road_count, road_rows, road_direction (for each road), car_row_pos, car_col_pos, car_direction (for each car)]
    def _get_obs(self):
        obs = np.concatenate((self.crossy_road.pedestrian_pos, [self.crossy_road.goal_row, self.crossy_road.road_count]))
        for road in self.crossy_road.road_rows:
            obs = np.concatenate((obs, [road[0], road[1]]))
        for car in self.crossy_road.cars:
            obs =  np.concatenate((obs, [car.row, car.column, car.direction]))
        return obs

    # Gym required function (and parameters) to reset the environment
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # gym requires this call to control randomness and reproduce scenarios.

        # Reset the CrossyRoad. Optionally, pass in seed control randomness and reproduce scenarios.
        self.crossy_road.reset(seed=seed)

        # Construct the observation state:
        # [pedestrian_row_pos, pedestrian_col_pos, goal_row, road_count, road_rows, road_direction (for each road), car_row_pos, car_col_pos, car_direction (for each car)]
        obs = self._get_obs()
        
        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        if(self.render_mode=='human'):
            self.render()

        # Return observation and info
        return obs, info

    # Gym required function (and parameters) to perform an action
    def step(self, action):
        # Backup the previous pedestrian position, to calculate distance change later
        old_pos = self.crossy_road.pedestrian_pos.copy()
        # Perform action
        target_reached = self.crossy_road.perform_action(cr.PedestrianAction(action))

        # Determine reward and termination
        reward = 0
        terminated=False

        # Check for goal reached or collision
        if target_reached:
            reward += 100
            terminated=True
            print("Goal Reached!")
        else:
            # Move cars
            self.crossy_road.car_positions = [] # Clear car positions, will be updated after moving cars
            for car in self.crossy_road.cars:
                car.move()
                # Update car positions
                for pos in car.get_occupied_pos():
                    self.crossy_road.car_positions.append(pos)
                # Check for collision
                if car.get_front_pos() == self.crossy_road.pedestrian_pos:
                    reward -= 100
                    terminated = True
                    print("Crashed by a car!")
                # Re-spawn car if out of bounds
                if car.is_out_of_bounds(self.crossy_road.grid_cols):
                    car.reset()
            # Reward for moving closer to goal, penalty for moving away, small penalty for no progress
            old_dist = abs(old_pos[0] - self.crossy_road.goal_row)
            new_dist = abs(self.crossy_road.pedestrian_pos[0] - self.crossy_road.goal_row)
            if new_dist < old_dist:
                reward += 5
            elif new_dist > old_dist:
                reward -= 5
            else:
                reward -= 0.5

        # Construct the observation state: 
        # [pedestrian_row_pos, pedestrian_col_pos, goal_row]
        obs = self._get_obs()

        # Additional info to return. For debugging or whatever.
        info = {}

        # Render environment
        if(self.render_mode=='human'):
            print(cr.PedestrianAction(action))
            self.render()

        # Return observation, reward, terminated, truncated (not used), info
        return obs, reward, terminated, False, info

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
