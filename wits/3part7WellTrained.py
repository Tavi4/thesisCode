import time
import mujoco.viewer

from stable_baselines3 import PPO

from envs.two_wheel_robot_env import TwoWheelRobotEnv


env = TwoWheelRobotEnv()

model = PPO.load(
    "ppo_two_wheel_robot.zip"
)

observation, info = env.reset()

initial_distance = info["distance_to_target"]


with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    print("\nWell-trained model demo")
    print(
        "Initial distance:",
        f"{initial_distance:.2f} m"
    )

    episode_finished = False

    while viewer.is_running():

        if not episode_finished:

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                episode_finished = True

                final_distance = info["distance_to_target"]

                print("\nEpisode finished.")
                print(
                    "Initial distance:",
                    f"{initial_distance:.2f} m"
                )
                print(
                    "Final distance:  ",
                    f"{final_distance:.2f} m"
                )

        viewer.sync()

        time.sleep(
            env.model.opt.timestep * env.frame_skip
        )