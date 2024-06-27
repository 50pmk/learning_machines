import cv2
import pandas as pd
from data_files import FIGRURES_DIR, RESULT_DIR
import os 
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)
import numpy as np
import pickle
import csv
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import os

class GymEnv(gym.Env):

    # We don't need metadata, but the superclass wants to see it
    metadata = {"render_modes": ["dummy"]}


    def __init__(self, render_mode=None, rob=IRobobo, max_steps=100, model_name=None):
        super(GymEnv, self).__init__()

        
        assert model_name != None
        assert render_mode is None
        assert rob != False 
        self.rob = rob

        self.model_name = model_name

        # Maximum number of steps per episode
        self.max_steps = max_steps

        # Define action space
        # NOTE: currently action space is a 1D-array from [0,1] which maps onto forward actions in ._move()
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Define observation space 
        # NOTE: currently state space is 4 states: left, middle & right camera mask percentages 
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.package_found = False

        # Initialise for logging
        self.log_irsdata = []
        self.log_rewards = []
        self.log_collision = []
        self.log_actions = []
        self.total_steps = 0

        self.cum_reward = 0
        self.food_count = 0

        # --- LOGGING ---
        self.log_df = pd.DataFrame(columns=['episode reward', 'total food'])

    def _set_camera(self, horizontal_pan=180, vertical_tilt=67):
        print("Setting Camera: ")
        self.rob.set_phone_pan_blocking(horizontal_pan, 100) 
        self.rob.set_phone_tilt_blocking(vertical_tilt, 100) 
        self.rob.set_phone_pan(horizontal_pan, 100) 
        self.rob.set_phone_tilt(vertical_tilt, 100) 
        print("horizontal pan:", self.rob.read_phone_pan())
        print("vertical tilt:", self.rob.read_phone_tilt())
        

    def _process_front_camera(self, bgr_image, mask_color='red', save_images=False):
        # Get the image from the front camera
        # Ensure that the image is retrieved correctly and is in BGR format initially
        if bgr_image is None:
            print("No image received from the camera.")
            return False 
    
        # Convert the BGR image to HSV format
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        if mask_color == 'green':
            # Define HSV range for green color
            if isinstance(self.rob, SimulationRobobo):
                lower_green = np.array([15, 150, 200])
                upper_green = np.array([75, 255, 250]) 
            else:
                lower_green = np.array([45, 70, 70])
                upper_green = np.array([85, 255, 250]) 

            # Threshold the HSV image to get only green colors
            mask = cv2.inRange(hsv_image, lower_green, upper_green)

        elif mask_color == 'red':
            # Define HSV range for red color - NOTE: red needs 2 sets of HSV ranges 
            if isinstance(self.rob, SimulationRobobo):
                # Define HSV range for red color
                lower_red1 = np.array([0, 100, 100])
                upper_red1 = np.array([10, 255, 255])

                lower_red2 = np.array([170, 100, 100])
                upper_red2 = np.array([180, 255, 255])
            else:
                # TODO - define HSV ranges for hardware
                pass 

            # Threshold the HSV image to get only red colors
            mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
            # Combine both masks
            mask = cv2.bitwise_or(mask1, mask2)

        # visualizations
        if save_images:
            cv2.imwrite(str(FIGRURES_DIR / f'sim_camera_{self.total_steps}.png'), bgr_image)
            cv2.imwrite(str(FIGRURES_DIR / f'sim_camera_mask_{mask_color}_{self.total_steps}.png'), mask)

        # Calculate the proportion of color in the image
        color_val = float(np.sum(mask > 0)) / float(mask.size)
        # Determine if color is detected
        if color_val > 0.001:  # Adjust threshold as needed
            color = 1
        else:
            color = 0

        return color 
        # return color_val
    

    def _normalize_irs(self, irs) -> np.array:
        clipped_arr = np.clip(irs, 0, 1400)
        normalized_arr = clipped_arr / 1400.0
        return normalized_arr


    def _move(self, action):
        '''
        action is a 1D-array in range [0,1]
            low action -> turn right 
            high action -> turn left 
        '''
        v_min=0 
        v_max=100
        # Calculate the wheel speeds
        left_speed = v_min + (v_max - v_min) * (1 - action)
        right_speed = v_min + (v_max - v_min) * action
        self.rob.move_blocking(int(left_speed[0]), int(right_speed[0]), 200)

        # # Alternative quicker speeds, trains longer, runs into wall issues
        # if action < 0.5:
        #     left_speed = 100
        #     right_speed = -20 + 100 * action[0] * 2 * 1.2
        # else:
        #     left_speed = -20 + 100 * (1 - action[0]) * 2 * 1.2
        #     right_speed = 100
        # self.rob.move_blocking(left_speed, right_speed, 100)

        
    def _get_obs(self):
        # -- irs component -- 
        obs_irs = self.rob.read_irs()
        obs_irs = self._normalize_irs(obs_irs)
        if obs_irs[4] > 0.8:
            obs_irs = 1
        else: obs_irs = 0

        # -- camera component -- 
        bgr_image = self.rob.get_image_front()

        # make 3 vertical image sections
        width = bgr_image.shape[1] // 3
        left_image, middle_image, right_image  = bgr_image[:, :width, :], bgr_image[:, width:2*width, :], bgr_image[:, 2*width:, :]

        # get camera observations 
        obs_camera = {}
        for color in ['red', 'green']: 
            left = self._process_front_camera(left_image, mask_color=color)
            middle = self._process_front_camera(middle_image, mask_color=color)
            right = self._process_front_camera(right_image, mask_color=color)
    
            # visualization 
            _ = self._process_front_camera(bgr_image, mask_color=color, save_images=True)

            # returns presence or percentage of pixels covered by the color mask 
            obs_camera[color] = np.array([left, middle, right])


        if obs_irs == 0: 
            return np.append(obs_camera['red'], obs_irs)
        else: 
            return np.append(obs_camera['green'], obs_irs)

        # return obs_irs, obs_camera['red'], obs_camera['green']


    def _get_info(self):
        return {'dummy_info': 0}


    def _spin_at_episode_start(self):
        random_amount = np.random.randint(0, 1001)
        self.rob.move_blocking(100, -100, random_amount)
    

    def _get_reward(self, observation, action): 

        l, m, r, has_package = observation

        reward = 0

        if not has_package: # only red states are given to l, m, r

            # if object detected only on left side
            if l == 1 and not m == 1: 
                # if robot turns left 
                if action > 0.5: 
                    reward += 1 

            # if object detected in middle 
            elif m == 1:
                # if robot stays relatively straight 
                if 0.45 < action < 0.55:
                    reward += 10
            
            # if object detected only on right side
            elif r == 1 and not m == 1:
                # if robot turns right 
                if action < 0.5: 
                    reward += 1

            else: 
                # promote spinning to detect objects? -> NOTE: seems to do that on it's own 
                pass 

        elif has_package: # only green states are given to l, m, r

            # give big reward for finding package 
            if not self.package_found: 
                reward += 50 
                print('found package!')
                self.package_found = True 


            # if object detected only on left side
            if l == 1 and not m == 1: 
                # if robot turns left 
                if action > 0.5: 
                    reward += 2 

            # if object detected in middle 
            elif m == 1:
                # if robot stays relatively straight 
                if 0.45 < action < 0.55:
                    reward += 20
            
            # if object detected only on right side
            elif r == 1 and not m == 1:
                # if robot turns right 
                if action < 0.5: 
                    reward += 2

            else: 
                # promote turning to detect base 
                if 0.2 < action < 0.4:
                    reward += 1


        # # if robot has package and hits base, give big reward 
        # if isinstance(self.rob, SimulationRobobo):
        #     # TODO: check if this food count works with the green base?? 
        #     if self.rob.base_detects_food: 
        #         reward += 50
        #         print('delivered package!')

        return reward
    

    def reset(self, seed=None, options=None):
        # This line is probably needed but does nothing
        super().reset(seed=seed)

        if isinstance(self.rob, SimulationRobobo):
            if self.rob.is_running():
                self.rob.stop_simulation()
            
            self.rob.play_simulation()
        else:
            # self.rob.talk("episode starts in five")
            # self.rob.sleep(5)
            pass

        # set camera position at the start 
        self._set_camera(horizontal_pan=180, vertical_tilt=109)

        # TODO: change depending on how we construct state & reward 
        observation = self._get_obs()

        info = self._get_info()

        # --- LOGGING --- 

        episode_logs = {
            'episode reward': [self.cum_reward],
            'total food': [self.food_count]
        }
        new_row = pd.DataFrame(episode_logs)

        self.log_df = pd.concat([self.log_df, new_row], ignore_index=True)
        
        if not os.path.exists(os.path.join(RESULT_DIR, 'log')):
            os.makedirs(os.path.join(RESULT_DIR, 'log'))
        # save at each episode 
        self.log_df.to_csv(str(RESULT_DIR / f'log/{self.model_name}.csv'), index=False)

        # re-initialize
        self.step_count = 0
        self.cum_reward = 0 
        self.package_found = False 

        print()

        return observation, info


    def step(self, action):

        info = self._get_info()

        # Take the action
        self._move(action)
        # self.rob.move_blocking(50, 50, 100) # TODO remove 

        # observation is the bool states of each camera mask: left, middle, right, has_package 
        observation = self._get_obs()

        reward = self._get_reward(observation, action)
            
        print('timestamp: ', self.total_steps)
        print('observation:', observation)
        print('action:', action)
        print('reward:', reward)

        # tracking metrics metrics 
        self.total_steps += 1 
        self.cum_reward += reward
        if isinstance(self.rob, SimulationRobobo):
            self.food_count = self.rob.nr_food_collected()
        self.step_count += 1        

        # Determine if the episode is terminated based on the number of steps
        terminated = self.step_count >= self.max_steps
        # terminate if robot doesn't have package after 50 steps 
        if self.step_count > 49 and observation[-1] == 0:
            terminated = True  


        # --- LOGGING ---

        # Output index 3 has to be False, because it is a deprecated feature
        return observation, reward, terminated, False, info


def task1(rob: IRobobo):
    
    model = DDPG(
        policy = 'MlpPolicy', 
        env = GymEnv(rob=rob, max_steps=100), 
        learning_rate=0.0001, 
        buffer_size=50000, 
        learning_starts=50, # TODO change back 
        batch_size=64, 
        tau=0.005, 
        gamma=0.99, 
        train_freq=1, 
        gradient_steps=1, 
        # action_noise=NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1)), # Change this too if the action space changes shape
        # action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2)), # Change this too if the action space changes shape
        replay_buffer_class=None, 
        replay_buffer_kwargs=None, 
        optimize_memory_usage=False, 
        tensorboard_log=None, 
        policy_kwargs=None, 
        verbose=0, 
        seed=None, 
        device='auto', 
        _init_setup_model=True
    )

    model.learn(total_timesteps=5000, log_interval=10, progress_bar=True)
    
    model.save(str(RESULT_DIR / 'models/run_1'))


def validate_task1(rob: IRobobo):
    model = DDPG.load(str(RESULT_DIR / 'models/run_2c')) 

    env = GymEnv()
    obs = env.reset()[0][1]
    while True:
        action, states = model.predict(obs)
        observation, reward, terminated, _, info = env.step(action)
        obs = observation[1]


def task2(rob: IRobobo, model_name=None):
    if model_name is None:
        raise ValueError("model_name must be provided")

    model = DDPG(
        policy = 'MlpPolicy', 
        env = GymEnv(rob=rob, max_steps=100, model_name=model_name), 
        learning_rate=0.0001, 
        buffer_size=50000, 
        learning_starts=100, 
        batch_size=64, 
        tau=0.005, 
        gamma=0.99, 
        train_freq=1, 
        gradient_steps=1, 
        # action_noise=NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1)), # Change this too if the action space changes shape
        # action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2)), # Change this too if the action space changes shape
        replay_buffer_class=None, 
        replay_buffer_kwargs=None, 
        optimize_memory_usage=False, 
        tensorboard_log=None, 
        policy_kwargs=None, 
        verbose=0, 
        seed=None, 
        device='auto', 
        _init_setup_model=True
    )

    model.learn(total_timesteps=2000, log_interval=10, progress_bar=True) # change back to 5000
    
    file_path = RESULT_DIR / f'models/{model_name}'

    # Check if the file already exists
    if os.path.exists(file_path):
        raise FileExistsError(f"The model '{model_name}' already exists at {file_path}")
    
    model.save(str(RESULT_DIR / f'models/{model_name}'))


def task2_demonstrate(rob: IRobobo, steps=100000, model_name=None):
    if model_name is None:
        raise ValueError("model_name must be provided")

    model = DDPG.load(str(RESULT_DIR / f'models/{model_name}')) 
    env = GymEnv(rob=rob, max_steps=1000, model_name=model_name)
    obs = env.reset()[0]

    for _ in range(steps):
        action, states = model.predict(obs)
        obs, reward, terminated, _, info = env.step(action)


def calibrate(rob: IRobobo):
    # Start the simulation
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        type = 'simulation'
    else: 
        type = 'hardware'


    # logging 
    positions = []
    irs_data = []

    # If sensor sees something: turn right, if not: straight ahead
    for _ in range(100):

        # Somehow the first value for read_irs() in the simulation is always [inf, inf, ...]
        # so the first value is hard_set to 0
        if _ == 0:
            irs = [0,0,0,0,0]
        else:
            # Read ['FrontL', 'FrontR', 'FrontC', 'FrontRR', 'FrontLL']
            irs = rob.read_irs()[2:6] + rob.read_irs()[7:8]
        
        print(irs)
        irs_data.append(irs)

        if isinstance(rob, SimulationRobobo):
            pos = rob.get_position()
            print(pos)
            positions.append(pos)
        # move back 
        rob.move_blocking(-10, -10, 500)

    # Stop simulation
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    # logging 
    run = 2
    df = pd.DataFrame(irs_data, columns=['FrontL', 'FrontR', 'FrontC', 'FrontRR', 'FrontLL'])
    # df.to_csv(str(RESULT_DIR / f'{type}_irs_calibrate_{run}.csv'), index=False)

    # with open(str(RESULT_DIR / f'{type}_pos_calibrate_{run}.csv'), 'w') as file:
    #     for item in positions:
    #         file.write(f"{item}\n")
