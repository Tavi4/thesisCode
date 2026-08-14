import numpy as np

from envs.two_wheel_robot_env import TwoWheelRobotEnv


env = TwoWheelRobotEnv()

obs, info = env.reset()

print("Initial observation:", obs)
print("Initial distance:", info["distance_to_target"])

for step in range(2000):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if not np.all(np.isfinite(obs)):
        raise RuntimeError(f"Non-finite observation at step {step}")

    if not np.isfinite(reward):
        raise RuntimeError(f"Non-finite reward at step {step}")

    if terminated or truncated:
        print(
            "Episode ended:",
            "step =", step,
            "terminated =", terminated,
            "truncated =", truncated,
            "distance =", info["distance_to_target"],
        )

        obs, info = env.reset()

print("Random-action smoke test passed.")