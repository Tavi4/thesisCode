import time
import mujoco.viewer

from envs.two_wheel_robot_env import TwoWheelRobotEnv


env = TwoWheelRobotEnv()
observation, info = env.reset()


with mujoco.viewer.launch_passive(env.model, env.data) as viewer:


    while viewer.is_running():

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        viewer.sync()

        time.sleep(
            env.model.opt.timestep * env.frame_skip
        )

        if terminated or truncated:
            observation, info = env.reset()