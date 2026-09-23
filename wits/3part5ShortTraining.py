from stable_baselines3 import PPO

from envs.two_wheel_robot_env import TwoWheelRobotEnv


env = TwoWheelRobotEnv()


print("\nStarting short training...")


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
)

model.learn(
    total_timesteps=10000
)


model.save("models/wits/short_trained_model")


print("\nTraining finished.")
print("Model saved to: models/wits/short_trained_model.zip")


env.close()