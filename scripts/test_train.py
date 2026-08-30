from stable_baselines3 import PPO

from envs.two_wheel_robot_env import TwoWheelRobotEnv


env = TwoWheelRobotEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
)

model.learn(
    total_timesteps=100_000
)

model.save("ppo_two_wheel_robot")

env.close()

print("Training finished.")