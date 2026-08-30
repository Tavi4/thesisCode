from stable_baselines3 import PPO

from envs.two_wheel_robot_env import TwoWheelRobotEnv


env = TwoWheelRobotEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
    tensorboard_log="./tensorboard/",
)

model.learn(
    total_timesteps=300_000
)

model.save("ppo_two_wheel_robot")

env.close()

print("Training finished.")