from stable_baselines3 import PPO

from envs.two_wheel_robot_env import TwoWheelRobotEnv


# Create the same environment used during training
env = TwoWheelRobotEnv()

# Load the trained PPO model
model = PPO.load(
    "ppo_two_wheel_robot",
    env=env,
)

# Start a new episode
obs, info = env.reset()

print("Initial distance:", info["distance_to_target"])


for step in range(100):

    # Let the trained model choose an action
    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    # Apply the predicted action to the environment
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step {step:03d} | "
        f"drive={action[0]:.3f} "
        f"steer={action[1]:.3f} | "
        f"distance={info['distance_to_target']:.3f}"
    )

    if terminated or truncated:
        print(
            "Episode ended:",
            f"terminated={terminated},",
            f"truncated={truncated}"
        )
        break


env.close()

print("Trained model test passed.")