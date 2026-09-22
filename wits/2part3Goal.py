import time
import mujoco
import mujoco.viewer

from envs.two_wheel_robot_env import TwoWheelRobotEnv


env = TwoWheelRobotEnv()
observation, info = env.reset()

initial_distance = info["distance_to_target"]


with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    viewer.sync()

    print("\nInitial observation:")
    print(observation)

    print("\nInitial distance to target:")
    print(f"{initial_distance:.2f} m")

    input("\nPress Enter")

    action = [0.8, -0.7]
    
    print("\nAction:")
    print("drive    =", action[0])
    print("steering =", action[1])

    for _ in range(250):
        # 5 secunde 
        observation, reward, terminated, truncated, info = env.step(action)

        viewer.sync()

        time.sleep(
            env.model.opt.timestep * env.frame_skip
        )

        if terminated or truncated:
            break

    final_distance = info["distance_to_target"]
    progress = initial_distance - final_distance

    print("\n--- RESULT ---")
    print("Initial distance:", f"{initial_distance:.2f} m")
    print("Final distance:  ", f"{final_distance:.2f} m")
    print("Moved closer by: ", f"{progress:.2f} m")

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)