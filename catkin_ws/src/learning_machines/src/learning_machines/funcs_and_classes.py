import cv2
from data_files import FIGRURES_DIR, RESULT_DIR
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

class GymEnv(gym.Env):

    # We don't need metadata, but the superclass wants to see it
    metadata = {"render_modes": ["dummy"]}

    def __init__(self, render_mode=None, rob=IRobobo, max_steps=100):
        super(GymEnv, self).__init__()


        assert render_mode is None
        assert rob != False 
        self.rob = rob

        # Maximum number of steps per episode
        self.max_steps = max_steps

        # Define action space
        self.action_space = gym.spaces.Box(low=np.array([0]), high=np.array([5]), dtype=np.float32)

        # Define observation space: assume 8 continuous observations ranging from 0 to 10000
        self.observation_space = gym.spaces.Box(low=0, high=200, shape=(8,), dtype=np.float32)

        # Initialise for logging
        self.log_irsdata = []
        self.log_rewards = []
        self.log_collision = []
        self.log_actions = []
        self.total_steps = 0
        self.total_timesteps_in_learn = 500

    def _move(self, action):
        # self.rob.move_blocking(action[0], action[1], 100)

        action = action[0]

        # if action < 1:
        #     self.rob.move_blocking(0, 100, 100)
        # elif action < 2:
        #     self.rob.move_blocking(50, 100, 100)
        # elif action < 3:
        #     self.rob.move_blocking(100, 100, 100)
        # elif action < 4:
        #     self.rob.move_blocking(100, 50, 100)
        # else:
        #     self.rob.move_blocking(100, 0, 100)

        left = max(2.5 - action, 0) * 100 / 2.5
        right = max(action - 2.5, 0) * 100 / 2.5
        self.rob.move_blocking(left, right, 100)
        
    def _get_obs(self):
        return self.rob.read_irs()

    def _get_info(self):
        return {'dummy_info': 0}

    def _spin_at_episode_start(self):
        random_amount = np.random.randint(0, 1001)
        self.rob.move_blocking(100, -100, random_amount)

    def _calculate_max_distance(self):
        if len(self.visited_positions) < 2:
            return 0.0
        
        # Calculate the pairwise differences
        diffs = self.visited_positions[:, np.newaxis, :] - self.visited_positions[np.newaxis, :, :]
        
        # Compute the Euclidean distances
        distances = np.sqrt(np.sum(diffs**2, axis=-1))
        
        # Find the maximum distance
        max_distance = np.max(distances)
        
        return max_distance

    def reset(self, seed=None, options=None):
        # This line is probably needed but does nothing
        super().reset(seed=seed)

        if isinstance(self.rob, SimulationRobobo):
            if self.rob.is_running():
                self.rob.stop_simulation()
            
            self.rob.play_simulation()
        else:
            self.rob.talk("Put me back")
            self.rob.sleep(5)

        # For maximizing explored distance reward method
        self.visited_positions = np.array([[self.rob.get_position().x, self.rob.get_position().y]])
        self.best_distance = 0

        # self._spin_at_episode_start()

        # Initialize step counter
        self.step_count = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):

        # Take the action
        self._move(action)

        observation = self._get_obs()
        info = self._get_info()

        # Save the new position for keeping track of how much distance the robot covered
        position = np.array([[self.rob.get_position().x, self.rob.get_position().y]])
        self.visited_positions = np.vstack((self.visited_positions, position))

        # # Reward based on maximizing explored distance method
        # new_best_distance = self._calculate_max_distance()
        # reward = new_best_distance - self.best_distance # if improved: value, else: 0
        # reward *= 10
        # self.best_distance = new_best_distance

        # Reward based on wheels difference
        reward = - abs(action[0] - 2.5)

        # # Reward update for collision punishment
        # if max(observation) > 100:
        #     reward -= 0.5

        # Increment step 
        self.step_count += 1
        self.total_steps += 1 # Only for logging
        
        # Determine if the episode is terminated based on the number of steps
        terminated = self.step_count >= self.max_steps

        # TODO early termination if max distance has been found

        # Logging
        self.log_irsdata.append(observation)
        self.log_rewards.append(reward)
        self.log_collision.append(max(observation) > 100)
        self.log_actions.append(action)
        if self.total_steps == self.total_timesteps_in_learn:
            with open(str(RESULT_DIR / 'data/run_2/picklec'), 'wb') as f:
                pickle.dump({
                    'irsdata': self.log_irsdata,
                    'rewards': self.log_rewards,
                    'collision': self.log_collision,
                    'actions': self.log_actions
                }, f)
        # More Logging
            with open(str(RESULT_DIR / 'data/run_2/irsc.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['observation'])
                writer.writerows([[obs] for obs in self.log_irsdata])
            with open(str(RESULT_DIR / 'data/run_2/rewards1.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['reward'])
                writer.writerows([[reward] for reward in self.log_rewards])
            with open(str(RESULT_DIR / 'data/run_2/collisionc.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['collision'])
                writer.writerows([[collision] for collision in self.log_collision])
            with open(str(RESULT_DIR / 'data/run_2/actions1.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['actions'])
                writer.writerows([[action] for action in self.log_actions])
        # /Logging

        # Output index 3 has to be False, because it is a deprecated feature
        return observation, reward, terminated, False, info


def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)


def test_move_and_wheel_reset(rob: IRobobo):
    rob.move_blocking(100, 100, 1000)
    print("before reset: ", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(1)
    print("after reset: ", rob.read_wheels())


def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())


def test_phone_movement(rob: IRobobo):
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())


def test_sim(rob: SimulationRobobo):
    print(rob.get_sim_time())
    print(rob.is_running())
    rob.stop_simulation()
    print(rob.get_sim_time())
    print(rob.is_running())
    rob.play_simulation()
    print(rob.get_sim_time())
    print(rob.get_position())


def test_hardware(rob: HardwareRobobo):
    print("Phone battery level: ", rob.read_phone_battery())
    print("Robot battery level: ", rob.read_robot_battery())


def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    test_emotions(rob)
    test_sensors(rob)
    test_move_and_wheel_reset(rob)
    if isinstance(rob, SimulationRobobo):
        test_sim(rob)

    if isinstance(rob, HardwareRobobo):
        test_hardware(rob)

    test_phone_movement(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


def test(rob: IRobobo):
    # Start the simulation
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    for _ in range(100):

        rob.move_blocking(100, 0, 250)

    # Stop simulation
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


def task0(rob: IRobobo):
    # Start the simulation
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    # If sensor sees something: turn right, if not: straight ahead
    for _ in range(100):

        # Somehow the first value for read_irs() in the simulation is always [inf, inf, ...]
        # so the first value is hard_set to 0
        if _ == 0:
            irs = [0,0,0]
        else:
            # Read FrontL, FrontR, FrontC infrared sensor
            irs = rob.read_irs()[2:5]

        print(irs)

        # Turn right or move straight
        if max(irs) > 100:
            rob.move_blocking(50, -100, 250)
        else:
            rob.move_blocking(50, 50, 100)

    # Stop simulation
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


def task1(rob: IRobobo):

    model = DDPG(
        policy = 'MlpPolicy', 
        env = GymEnv(rob=rob, max_steps=100), 
        learning_rate=0.001, 
        buffer_size=50000, 
        learning_starts=100, 
        batch_size=256, 
        tau=0.005, 
        gamma=0.99, 
        train_freq=1, 
        gradient_steps=1, 
        action_noise=NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1)), # Change this too if the action space changes shape
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

    model.learn(total_timesteps=500, log_interval=10, progress_bar=True)
    
    model.save(str(RESULT_DIR / 'models/run_2c'))
