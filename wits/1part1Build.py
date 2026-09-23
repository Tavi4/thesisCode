import time

import mujoco
import mujoco.viewer


models = {
    "1": "models/wits/01_body.xml",
    "2": "models/wits/02_body_wheels.xml",
    "3": "models/wits/03_joints_actuators.xml",
}


print("\nChoose robot stage:")
print("1. Body")
print("2. Body + wheels")
print("3. Joints + actuators")

choice = input("\nSelect stage [1-3]: ").strip()

model_path = models.get(choice)

if model_path is None:
    raise ValueError("Invalid choice.")

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:

    if choice == "3":
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True

    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)

        viewer.sync()

        time_until_next_step = model.opt.timestep - (
            time.time() - step_start
        )

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)