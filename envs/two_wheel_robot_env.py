import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from scripts.functions import (
    get_observation,
    compute_reward,
    is_success,
    reset_robot,
)

MAX_STEER = np.deg2rad(25)


class TwoWheelRobotEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.frame_skip = 10

        self.max_episode_steps = 500
        self.current_step = 0
        
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(
            "models/two_wheel_robot.xml"
        )

        self.data = mujoco.MjData(self.model)

        # RL action:
        # action[0] = drive
        # action[1] = steering
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # Current observation:
        # [x, y, yaw, vx, vy, yaw_rate, steer_left, steer_right]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

    def _apply_action(self, action):
        action = np.clip(action, -1.0, 1.0)

        drive = action[0]
        steering = action[1] * MAX_STEER

        # Synchronous drive
        self.data.ctrl[0] = drive
        self.data.ctrl[1] = drive

        # Synchronous steering
        self.data.ctrl[2] = steering
        self.data.ctrl[3] = steering

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        reset_robot(self.model, self.data)

        self.current_step = 0

        observation = get_observation(
            self.model,
            self.data,
        )

        info = {
            "distance_to_target": float(
                -compute_reward(self.model, self.data)
            )
        }

        return observation, info

    def step(self, action):
        self._apply_action(action)

        for _ in range(self.frame_skip):
            mujoco.mj_step(
                self.model,
                self.data,
            )

        self.current_step += 1

        observation = get_observation(
            self.model,
            self.data,
        )

        reward = compute_reward(
            self.model,
            self.data,
        )

        terminated = is_success(
            self.model,
            self.data,
        )

        truncated = self.current_step >= self.max_episode_steps

        info = {
            "distance_to_target": float(-reward)
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )