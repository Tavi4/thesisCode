import numpy as np
import mujoco


MAX_STEER = np.deg2rad(30)
SUCCESS_DISTANCE = 0.10


def apply_direct_controls(
    data,
    drive_left,
    drive_right,
    steer_left,
    steer_right,
):
    #Applying independent controls for manual testing.
    data.ctrl[0] = np.clip(drive_left, -1.0, 1.0)
    data.ctrl[1] = np.clip(drive_right, -1.0, 1.0)

    data.ctrl[2] = np.clip(
        np.deg2rad(steer_left),
        -MAX_STEER,
        MAX_STEER,
    )

    data.ctrl[3] = np.clip(
        np.deg2rad(steer_right),
        -MAX_STEER,
        MAX_STEER,
    )


def get_observation(model, data):
    #Return the 8 values observed by the RL agent.
    x_id = model.joint("robot_x").id
    y_id = model.joint("robot_y").id
    yaw_id = model.joint("robot_yaw").id

    steer_left_id = model.joint("steer_left_joint").id
    steer_right_id = model.joint("steer_right_joint").id

    observation = np.array([
        # Position and orientation
        data.qpos[model.jnt_qposadr[x_id]],
        data.qpos[model.jnt_qposadr[y_id]],
        data.qpos[model.jnt_qposadr[yaw_id]],

        # Linear and angular velocity
        data.qvel[model.jnt_dofadr[x_id]],
        data.qvel[model.jnt_dofadr[y_id]],
        data.qvel[model.jnt_dofadr[yaw_id]],

        # Steering angles
        data.qpos[model.jnt_qposadr[steer_left_id]],
        data.qpos[model.jnt_qposadr[steer_right_id]],
    ], dtype=np.float32)

    return observation


def get_robot_position(model, data):

    #Return robot position on the XY plane.
    x_id = model.joint("robot_x").id
    y_id = model.joint("robot_y").id

    return np.array([
        data.qpos[model.jnt_qposadr[x_id]],
        data.qpos[model.jnt_qposadr[y_id]],
    ], dtype=np.float32)


def get_target_position(model, data):

    #Return target position on the XY plane.
    target_id = model.body("target").id

    return data.xpos[target_id][:2].copy()


def get_distance_to_target(model, data):

    #Return Euclidean distance between robot and target.
    robot_position = get_robot_position(model, data)
    target_position = get_target_position(model, data)

    return float(
        np.linalg.norm(target_position - robot_position)
    )


def is_success(model, data):
    
    #Check if the robot reached the target.
    return (
        get_distance_to_target(model, data)
        < SUCCESS_DISTANCE
    )


def reset_robot(model, data):

    #Reset MuJoCo simulation to its initial state.
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)