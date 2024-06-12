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
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

class GymEnv(gym.Env):

    # We don't need metadata, but the superclass wants to see it
    metadata = {"render_modes": ["dummy"]}

    def __init__(self, render_mode=None, rob=IRobobo):

        assert render_mode is None
        assert rob != False 
        self.rob = rob

        # Define action space: two continuous values for left and right wheel speeds
        self.action_space = gym.spaces.Box(low=np.array([-100, -100]), high=np.array([100, 100]), dtype=np.float32)

        # Define observation space: assume 4 continuous observations ranging from 0 to 10000
        self.observation_space = gym.spaces.Box(low=0, high=20000, shape=(8,), dtype=np.float32)

    def _move(self, action):
        self.rob.move_blocking(action[0], action[1], 100)
        # TODO calculate rob distance

    def _get_obs(self):
        return self.rob.read_irs()

    def _get_info(self):
        return {'dummy_info': 0}

    def _spin_at_episode_start(self):
        left = 0 # should take random value between no spinning and 360 degrees spinning
        self.rob.move_blocking(left, 0, 100)

    def _reward(self):
        pass

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

        # Variable for tracking location relative to starting location
        self.location = {'x': 0, 'y': 0, 'degrees_rotated': 0}

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):

        # Take the action
        self._move(action)

        observation = self._get_obs()
        info = self._get_info()

        # TODO
        reward = 0 + np.sqrt(abs(self.location['x'] + abs(self.location['y'])))
        if max(observation) > 200:
            reward -= 10

        # TODO
        terminated = 0
        # TODO early termination if max distance has been found

        print(observation)
        print(reward)

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
        env = GymEnv(rob=rob), 
        learning_rate=0.001, 
        buffer_size=1000000, 
        learning_starts=100, 
        batch_size=256, 
        tau=0.005, 
        gamma=0.99, 
        train_freq=1, 
        gradient_steps=1, 
        action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2)), 
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

    model.learn(total_timesteps=10000, log_interval=2)

    model.save(str(RESULT_DIR / 'models/ddpg_test'))
