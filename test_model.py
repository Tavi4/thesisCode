import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    "models/two_wheel_robot.xml"
)

data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # Drive
        data.ctrl[0] = 0.6
        data.ctrl[1] = 0.6

        # Steering
        target = np.deg2rad(15)

        data.ctrl[2] = target
        data.ctrl[3] = target

        mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(model.opt.timestep)