import time
import mujoco
import mujoco.viewer

import scripts.functions as functions

model = mujoco.MjModel.from_xml_path(
    "models/two_wheel_robot.xml"
)

data = mujoco.MjData(model)

print("\nChoose test mode:")
print("1. No input")
print("2. Default movement")
print("3. Custom controls")

choice = input("\nSelect mode [1-3]: ").strip()


if choice == "1":
    drive_left = 0.0
    drive_right = 0.0
    steer_left = 0.0
    steer_right = 0.0

elif choice == "2":
    drive_left = 0.6
    drive_right = 0.3

    steer_left = 15.0
    steer_right = 10.0

elif choice == "3":
    print("\nValues for drive must be between -1 and 1.")
    print("Steering is entered in degrees, between -25 and 25.\n")

    drive_left = float(
        input("Left wheel drive [-1, 1]: ")
    )

    drive_right = float(
        input("Right wheel drive [-1, 1]: ")
    )

    steer_left = float(
        input("Left steering angle [-25, 25] deg: ")
    )

    steer_right = float(
        input("Right steering angle [-25, 25] deg: ")
    )

else:
    print("Invalid option.")
    raise SystemExit


print("\nControls:")
print("drive_left  =", drive_left)
print("drive_right =", drive_right)
print("steer_left  =", steer_left, "deg")
print("steer_right =", steer_right, "deg")


with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        functions.apply_direct_controls(
            data,
            drive_left,
            drive_right,
            steer_left,
            steer_right,
        )

        mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(model.opt.timestep)