import time

import mujoco.viewer
from stable_baselines3 import PPO

from envs.two_wheel_robot_env import TwoWheelRobotEnv


# Create the same environment used for training
env = TwoWheelRobotEnv()

# Load the trained PPO model
model = PPO.load(
    "ppo_two_wheel_robot",
    env=env,
    device="cpu",
)

# Start a new episode
obs, info = env.reset()

print("Initial distance:", info["distance_to_target"])


# Open MuJoCo Viewer using the model and data from the environment
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    step = 0

    while viewer.is_running():

        # The trained PPO policy chooses the action
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        # Apply the action through the Gymnasium environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Update the Viewer
        viewer.sync()

        # Display some useful information
        if step % 10 == 0:
            print(
                f"Step {step:03d} | "
                f"drive={action[0]:.3f} | "
                f"steering={action[1]:.3f} | "
                f"distance={info['distance_to_target']:.3f} | "
                f"reward={reward:.3f}"
            )

        # Slow simulation down so we can watch it
        time.sleep(
            env.model.opt.timestep * env.frame_skip
        )

        # Stop when the episode ends
        if terminated or truncated:

            print(
                "Episode ended | "
                f"success={terminated} | "
                f"truncated={truncated} | "
                f"distance={info['distance_to_target']:.3f}"
            )

            break

        step += 1


env.close()