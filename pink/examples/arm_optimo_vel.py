#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 Stéphane Caron

"""Universal Robots UR5 arm tracking a moving target."""

from typing import Any
import numpy as np
import qpsolvers
import csv
import time
import meshcat_shapes
import pink
import pinocchio as pin
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask, DampingTask
from pink.utils import custom_configuration_vector
from pink.visualization import start_meshcat_visualizer

import matplotlib.pyplot as plt

try:
    from loop_rate_limiters import RateLimiter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples use loop rate limiters, "
        "try `[conda|pip] install loop-rate-limiters`"
    ) from exc

try:
    from robot_descriptions.loaders.pinocchio import load_robot_description
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples need robot_descriptions, "
        "try `[conda|pip] install robot_descriptions`"
    ) from exc


EE_NAME = "link8"
URDF_PATH = "/home/jdx/Documents/1.0LatentAct/pink/data/roboligent_optimo_description/roboligent_optimo.urdf"
PACKAGE_DIRS = ["/home/jdx/Documents/1.0LatentAct/pink/data/roboligent_optimo_description"]
ROOT_JOINT = None
PATH_TO_POSES = "/home/jdx/Documents/1.0LatentAct/pink/data/odometry_data_grab.csv"

# Transformation from sensor frame to object frame
# TODO: Update these values with actual calibration data
SENSOR_TO_OBJECT_ROTATION = np.array([[0, 0.939693, 0.342020], [-1, 0, 0], [0, -0.342020, 0.939693]]) # 3x3 rotation matrix
SENSOR_TO_OBJECT_TRANSLATION = np.array([0.098085, -17.5, 0.017882])  # translation vector

def read_poses(path, R_sensor_to_object=None, t_sensor_to_object=None):
    """Read poses from CSV file and transform from sensor frame to object frame.
    
    CSV format: timestamp,seq,frame_id,child_frame_id,pos_x,pos_y,pos_z,orient_x,orient_y,orient_z,orient_w,...
    
    Args:
        path: Path to CSV file
        R_sensor_to_object: 3x3 rotation matrix from sensor to object frame (default: identity)
        t_sensor_to_object: 3x1 translation vector from sensor to object frame (default: zeros)
    """
    if R_sensor_to_object is None:
        R_sensor_to_object = SENSOR_TO_OBJECT_ROTATION
    if t_sensor_to_object is None:
        t_sensor_to_object = SENSOR_TO_OBJECT_TRANSLATION
    poses = []
    try:
        with open(path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Extract position from odometry data
                position = np.array([
                    float(row['pos_x']),
                    float(row['pos_y']),
                    float(row['pos_z'])
                ])
                
                # Extract quaternion and convert to rotation matrix
                quat = np.array([
                    float(row['orient_z']),
                    float(row['orient_y']),
                    float(row['orient_x']),
                    float(row['orient_w'])
                ])
                # Use pinocchio to convert quaternion to rotation matrix
                # Pinocchio uses (w, x, y, z) order
                rotation_sensor = pin.Quaternion(quat[3], quat[0], quat[1], quat[2]).toRotationMatrix()
                
                # position_object = R_sensor_to_object @ position + t_sensor_to_object
                # rotation_object = R_sensor_to_object @ rotation_sensor
                
                poses.append({
                    "position": position,
                    "rotation": rotation_sensor
                })
    except FileNotFoundError:
        print(f"Error: CSV file not found at {path}")
        return []
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []
    
    return poses


if __name__ == "__main__":
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=URDF_PATH,
        package_dirs=PACKAGE_DIRS,
        root_joint=ROOT_JOINT,
    )
    viz = start_meshcat_visualizer(robot)
    viewer = viz.viewer
    meshcat_shapes.frame(viewer["end_effector_target"], opacity=0.2)
    meshcat_shapes.frame(viewer["end_effector"], opacity=1.0)

    end_effector_task = FrameTask(
        EE_NAME,
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=1.0,  # tuned for this setup
    )

    posture_task = PostureTask(
        cost=1e-1,  # [cost] / [rad]
    )
    damping_task = DampingTask(
        cost=1.0,  # [cost] * [s] / [rad]
    )

    tasks = [end_effector_task, posture_task]


    q_ref = custom_configuration_vector( # original
        robot,
        joint1=0.0,
        joint2=3.8,
        joint3=0.0,
        joint4=-1.8,
        joint5=0.0,
        joint6=-0.5,
        joint7=0.0,
    )

    # q_ref = custom_configuration_vector( # first initla pose
    #     robot,
    #     joint1=0.0,
    #     joint2=3.14,
    #     joint3=0.0,
    #     joint4=-2.0,
    #     joint5=0.0,
    #     joint6=0.36,
    #     joint7=0.0,
    # )

    # q_ref = custom_configuration_vector( # second initla pose
    #     robot,
    #     joint1=-0.122821550858719,
    #     joint2=2.46169484598639,
    #     joint3=-0.037354025758061,
    #     joint4=-1.01633248485748,
    #     joint5=0.806033596526222,
    #     joint6=-0.0297647572468588,
    #     joint7=0.785874218297791,
    # )

    
    configuration = pink.Configuration(robot.model, robot.data, q_ref)
    for task in tasks:
        task.set_target_from_configuration(configuration)
    tasks.append(damping_task)
    viz.display(configuration.q)

    # Select QP solver
    solver = qpsolvers.available_solvers[0]
    if "daqp" in qpsolvers.available_solvers:
        solver = "daqp"

    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    t = 0.0  # [s]

    ee_poses = read_poses(PATH_TO_POSES)
    print(f"Loaded {len(ee_poses)} poses from rosbag")
    
    # Get initial end effector pose from q_ref configuration
    initial_ee_transform = configuration.get_transform_frame_to_world(EE_NAME)
    initial_position = initial_ee_transform.translation.copy()
    initial_rotation = initial_ee_transform.rotation.copy()
    
    print(f"Initial end effector position: {initial_position}")
    print(f"First CSV pose (relative): {ee_poses[0]['position']}")
    
    # Get the first CSV pose as the reference (to make trajectory start at current position)
    first_csv_position = ee_poses[0]["position"]
    first_csv_rotation = ee_poses[0]["rotation"]
    
    # Continuous loop: solve IK and track targets in real-time
    idx = 0
    t = 0.0
    solved_qs = []
    for pose in ee_poses:
        start_time = time.time()
        end_effector_target = end_effector_task.transform_target_to_world
        
        relative_position = ee_poses[idx]["position"] - first_csv_position
        end_effector_target.translation = initial_position + relative_position
        
        relative_rotation = first_csv_rotation.T @ ee_poses[idx]["rotation"]
        end_effector_target.rotation = initial_rotation @ relative_rotation
        
        # Solve IK to compute joint velocities
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)
        solved_qs.append(configuration.q)
        # Update robot visualization
        viz.display(configuration.q)
        
        # Update frame visualizations
        viewer["end_effector_target"].set_transform(end_effector_target.np)
        viewer["end_effector"].set_transform(
            configuration.get_transform_frame_to_world(
                end_effector_task.frame
            ).np
        )
        idx += 1
        rate.sleep()
    while True:
        # Update task targets
        if idx >= len(ee_poses):
            idx = 0
        end_effector_target = end_effector_task.transform_target_to_world
        
        relative_position = ee_poses[idx]["position"] - first_csv_position
        end_effector_target.translation = initial_position + relative_position
        
        relative_rotation = first_csv_rotation.T @ ee_poses[idx]["rotation"]
        end_effector_target.rotation = initial_rotation @ relative_rotation

        # Update visualization frames
        viewer["end_effector_target"].set_transform(end_effector_target.np)
        viewer["end_effector"].set_transform(
            configuration.get_transform_frame_to_world(
                end_effector_task.frame
            ).np
        )
        
        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=solver)
        configuration.integrate_inplace(velocity, dt)
        viz.display(configuration.q)
        idx += 1
        rate.sleep()

    fig, ax = plt.subplots()
    ax.plot(solved_qs)
    plt.show()
    # save the solved_qs to a csv file
    with open('solved_qs_grabfrom.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'])
        for i, q in enumerate(solved_qs):
            writer.writerow([i, q[0], q[1], q[2], q[3], q[4], q[5], q[6]]) 