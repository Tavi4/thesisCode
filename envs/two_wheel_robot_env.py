import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from scripts.functions import (
    MAX_STEER,
    get_observation,
    get_distance_to_target,
    is_success,
    reset_robot,
)


class TwoWheelRobotEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(
            "models/two_wheel_robot.xml"
        )
        self.data = mujoco.MjData(self.model)

        # Simulation settings
        self.frame_skip = 10
        self.max_episode_steps = 500

        self.current_step = 0
        self.previous_distance = None

        # action = [drive, steering]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # observation =
        # [x, y, yaw, vx, vy, yaw_rate, steer_left, steer_right]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

    def _apply_action(self, action):

        #Map the 2 RL actions to the 4 MuJoCo actuators.
        action = np.clip(action, -1.0, 1.0)

        drive = action[0]
        steering = action[1] * MAX_STEER

        # Synchronous wheel drive
        self.data.ctrl[0] = drive
        self.data.ctrl[1] = drive

        # Synchronous steering
        self.data.ctrl[2] = steering
        self.data.ctrl[3] = steering

    def reset(self, seed=None, options=None):

        #Start a new episode.
        super().reset(seed=seed)

        reset_robot(self.model, self.data)

        self.current_step = 0

        self.previous_distance = get_distance_to_target(
            self.model,
            self.data,
        )

        observation = get_observation(
            self.model,
            self.data,
        )

        info = {
            "distance_to_target": self.previous_distance
        }

        return observation, info

    def step(self, action):

        #Apply one RL action and advance the simulation.
        self._apply_action(action)

        # Keep the action active for several physics steps
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        observation = get_observation(
            self.model,
            self.data,
        )

        current_distance = get_distance_to_target(
            self.model,
            self.data,
        )

        # Positive reward when moving closer to the target
        reward = self.previous_distance - current_distance

        terminated = is_success(
            self.model,
            self.data,
        )

        # small penalty every step
        reward -= 0.01

        # Success bonus
        if terminated:
            reward += 100

        truncated = (
            self.current_step >= self.max_episode_steps
        )

        self.previous_distance = current_distance

        info = {
            "distance_to_target": current_distance
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )