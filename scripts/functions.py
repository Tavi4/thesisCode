import numpy as np
import mujoco


MAX_STEER = np.deg2rad(25)

SUCCESS_DISTANCE = 0.10

def apply_direct_controls(data, drive_left, drive_right, steer_left, steer_right):
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
    # Joint IDs
    x_id = model.joint("robot_x").id
    y_id = model.joint("robot_y").id
    yaw_id = model.joint("robot_yaw").id

    steer_left_id = model.joint("steer_left_joint").id
    steer_right_id = model.joint("steer_right_joint").id

    # qpos addresses
    x_qpos = model.jnt_qposadr[x_id]
    y_qpos = model.jnt_qposadr[y_id]
    yaw_qpos = model.jnt_qposadr[yaw_id]

    steer_left_qpos = model.jnt_qposadr[steer_left_id]
    steer_right_qpos = model.jnt_qposadr[steer_right_id]

    # qvel addresses
    x_qvel = model.jnt_dofadr[x_id]
    y_qvel = model.jnt_dofadr[y_id]
    yaw_qvel = model.jnt_dofadr[yaw_id]

    observation = np.array([
        data.qpos[x_qpos],
        data.qpos[y_qpos],
        data.qpos[yaw_qpos],

        data.qvel[x_qvel],
        data.qvel[y_qvel],
        data.qvel[yaw_qvel],

        data.qpos[steer_left_qpos],
        data.qpos[steer_right_qpos],
    ], dtype=np.float32)

    return observation

def get_target_position(model, data):
    target_id = model.body("target").id
    return data.xpos[target_id][:2].copy()


def get_robot_position(model, data):
    x_id = model.joint("robot_x").id
    y_id = model.joint("robot_y").id

    x_qpos = model.jnt_qposadr[x_id]
    y_qpos = model.jnt_qposadr[y_id]

    return np.array([
        data.qpos[x_qpos],
        data.qpos[y_qpos],
    ], dtype=np.float32)


def get_distance_to_target(model, data):
    robot_pos = get_robot_position(model, data)
    target_pos = get_target_position(model, data)

    return np.linalg.norm(target_pos - robot_pos)


def compute_reward(model, data):
    distance = get_distance_to_target(model, data)

    reward = -distance

    return float(reward)

SUCCESS_DISTANCE = 0.10


def is_success(model, data):
    distance = get_distance_to_target(model, data)

    return distance < SUCCESS_DISTANCE


def reset_robot(model, data):
    mujoco.mj_resetData(model, data)

    # Poziția inițială a robotului
    x_id = model.joint("robot_x").id
    y_id = model.joint("robot_y").id
    z_id = model.joint("robot_z").id
    yaw_id = model.joint("robot_yaw").id

    data.qpos[model.jnt_qposadr[x_id]] = 0.0
    data.qpos[model.jnt_qposadr[y_id]] = 0.0
    data.qpos[model.jnt_qposadr[z_id]] = 0.0
    data.qpos[model.jnt_qposadr[yaw_id]] = 0.0

    mujoco.mj_forward(model, data)