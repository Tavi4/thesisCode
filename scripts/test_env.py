import numpy as np

from gymnasium.utils.env_checker import check_env

from envs.two_wheel_robot_env import TwoWheelRobotEnv

env = TwoWheelRobotEnv()

# Verify that the environment follows the Gymnasium API
check_env(env)

print("Gymnasium check passed.")

# Start a new episode
obs, info = env.reset()

print("Initial observation:", obs)
print("Initial distance:", info["distance_to_target"])


# Execute random actions to test the environment
for step in range(1000):

    # Generate a random valid action
    action = env.action_space.sample()

    # Advance one RL step
    obs, reward, terminated, truncated, info = env.step(action)

    # Reset when an episode ends
    if terminated or truncated:

        print(
            f"Episode ended at step {step}: "
            f"terminated={terminated}, "
            f"truncated={truncated}, "
            f"distance={info['distance_to_target']:.3f}"
        )

        obs, info = env.reset()

print("Environment test passed.")