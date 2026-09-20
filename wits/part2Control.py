import time
import mujoco
import mujoco.viewer

import scripts.functions as functions


model = mujoco.MjModel.from_xml_path(
    "models/two_wheel_robot.xml"
)

data = mujoco.MjData(model)


print("\nChoose control mode:")
print("1. No input")
print("2. Default movement")
print("3. Custom drive + steering")

choice = input("\nSelect mode [1-3]: ").strip()


if choice == "1":
    drive = 0.0
    steering = 0.0

elif choice == "2":
    drive = 0.6
    steering = 15.0

elif choice == "3":
    print("\nDrive must be between -1 and 1.")
    print("Steering must be between -25 and 25 degrees.\n")

    drive = float(
        input("Drive [-1, 1]: ")
    )

    steering = float(
        input("Steering [-25, 25] deg: ")
    )

else:
    raise SystemExit("Invalid option.")


print("\nControls:")
print("drive    =", drive)
print("steering =", steering, "deg")


with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        functions.apply_direct_controls(
            data,
            drive,
            drive,
            steering,
            steering,
        )

        mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(model.opt.timestep)