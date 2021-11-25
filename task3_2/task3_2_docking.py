import subprocess, math, time, sys, os, numpy as np
import threading

import matplotlib.pyplot as plt
import pybullet as bullet_simulation
import pybullet_data

# setup paths and load the core
abs_path = os.path.dirname(os.path.realpath(__file__))
root_path = abs_path + '/..'
core_path = root_path + '/core'
sys.path.append(core_path)
from Pybullet_Simulation import Simulation

# specific settings for this task

taskId = 3.2

try:
    if sys.argv[1] == 'nogui':
        gui = False
    else:
        gui = True
except:
    gui = True

pybulletConfigs = {
    "simulation": bullet_simulation,
    "pybullet_extra_data": pybullet_data,
    "gui": gui,
    "panels": False,
    "realTime": False,
    "controlFrequency": 1000,
    "updateFrequency": 250,
    "gravity": -9.81,
    "gravityCompensation": 1.,
    "floor": True,
    "cameraSettings": (1.2, 90, -22.8, (-0.12, -0.01, 0.99))
}
robotConfigs = {
    "robotPath": core_path + "/nextagea_description/urdf/NextageaOpen.urdf",
    "robotPIDConfigs": core_path + "/PD_gains.yaml",
    "robotStartPos": [0, 0, 0.85],
    "robotStartOrientation": [0, 0, 0, 1],
    "fixedBase": True,
    "colored": False
}

sim = Simulation(pybulletConfigs, robotConfigs)

##### Please leave this function unchanged, feel free to modify others #####
def getReadyForTask():
    global finalTargetPos
    global taleId, cubeId, targetId, obstacle
    finalTargetPos = np.array([0.35,0.38,1.0])
    # compile target urdf
    urdf_compiler_path = core_path + "/urdf_compiler.py"
    subprocess.call([urdf_compiler_path,
                     "-o", abs_path+"/lib/task_urdfs/task3_2_target_compiled.urdf",
                     abs_path+"/lib/task_urdfs/task3_2_target.urdf"])

    sim.p.resetJointState(bodyUniqueId=1, jointIndex=12, targetValue=-0.4)
    sim.p.resetJointState(bodyUniqueId=1, jointIndex=6, targetValue=-0.4)

    # load the table in front of the robot
    tableId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/table/table_taller.urdf",
        basePosition          = [0.8, 0, 0],
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/2]),
        useFixedBase          = True,
        globalScaling         = 1.4
    )
    cubeId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/cubes/task3_2_dumb_bell.urdf",
        basePosition          = [0.5, 0, 1.1],
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,0]),
        useFixedBase          = False,
        globalScaling         = 1.4
    )
    sim.p.resetVisualShapeData(cubeId, -1, rgbaColor=[1,1,0,1])

    targetId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/task3_2_target_compiled.urdf",
        basePosition          = finalTargetPos,
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/4]),
        useFixedBase          = True,
        globalScaling         = 1
    )
    obstacle = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/cubes/task3_2_obstacle.urdf",
        basePosition          = [0.43,0.275,0.9],
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/4]),
        useFixedBase          = True,
        globalScaling         = 1
    )

    return tableId, cubeId, targetId


def solution():
    endEffector_L = 'LHAND'
    endEffector_R = 'RHAND'

    trajectoryL = list(np.linspace(np.array([0.50497461,  0.23, 0.18214933]), np.array([0.48,  0.09, 0.21]), 4))
    trajectoryR = list(np.linspace(np.array([0.50497461, -0.23, 0.18214933]), np.array([0.48, -0.09, 0.21]), 4))
    for i, j in zip(trajectoryL, trajectoryR):
        sim.move_two_joints_with_PD(endEffector_L, endEffector_R, i, j, orientation=[0, 0, 0])

    sim.moveJoint("LARM_JOINT5", np.deg2rad(-180), 0.0, 0.0017) # lower speed, more steps
    sim.moveJoint("RARM_JOINT5", np.deg2rad(180), 0.0, 0.0017)

    print(1, endEffector_L, sim.getJointPosition(endEffector_L))
    print(2, endEffector_R, sim.getJointPosition(endEffector_R))

    trajectoryL = list(np.linspace(np.array([0.48,  0.09, 0.18971861]),  np.array([0.48, 0.09, 0.42]), 5))
    trajectoryR = list(np.linspace(np.array([0.48, -0.09, 0.18963386]), np.array([0.48, -0.09, 0.42]), 5))
    for i, j in zip(trajectoryL, trajectoryR):
        sim.move_two_joints_with_PD(endEffector_L, endEffector_R, i, j, orientation=[0, 0, 0])

    sim.moveJoint("CHEST_JOINT0", np.deg2rad(42), 0.0, 0.0010)
    print(11, endEffector_L, sim.getJointPosition(endEffector_L))
    print(22, endEffector_R, sim.getJointPosition(endEffector_R))

    # leave dumb bell
    trajectoryL = list(np.linspace(np.array([0.25514343, 0.42992666, 0.40]), np.array([0.24514343, 0.42992666, 0.08]), 8))
    trajectoryR = list(np.linspace(np.array([0.44914699, 0.22027466, 0.40]), np.array([0.45914699, 0.22027466, 0.08]), 8))
    for i, j in zip(trajectoryL, trajectoryR):
        sim.move_two_joints_with_PD(endEffector_L, endEffector_R, i, j, orientation=[0, 0, 0])

    print(111, endEffector_L, sim.getJointPosition(endEffector_L))
    print(222, endEffector_R, sim.getJointPosition(endEffector_R))

    actual = sim.p.getBasePositionAndOrientation(cubeId)
    expected = sim.p.getBasePositionAndOrientation(targetId)
    print('Actual Position: {}, Expected: {}'.format(actual[0], expected[0]))
    # print('Position Error: {}'.format(np.asarray(expected[0])-np.asarray(actual[0])))
    print('Error: {}'.format(np.linalg.norm(np.asarray(expected[0]) - np.asarray(actual[0]))))
    # print('Actual Orientation: {}, Expected: {}'.format(sim.p.getEulerFromQuaternion(actual[1]), sim.p.getEulerFromQuaternion(expected[1])))
    # print('Orientation Error: {}'.format(np.asarray(sim.p.getEulerFromQuaternion(expected[1]))-np.asarray(sim.p.getEulerFromQuaternion(actual[1]))))
    #  convert to rads







    # 2nd best
    # Actual Position: (0.35868788963724574, 0.35678059077403135, 1.022964196283556), Expected: (0.35, 0.38, 1.0)
    # Position Error: [-0.00868789  0.02321941 - 0.0229642]
    # Actual Orientation: (0.008779203102985557, -0.1286328045455976, 0.7661142366983472), Expected: (-0.0, -0.0, 0.7853981633974485)
    # Orientation Error: [-0.0087792   0.1286328   0.01928393]
    # sim.moveJoint("CHEST_JOINT0", np.deg2rad(42), 0.0, 0.0010)
    # print(1, endEffector_L, sim.getJointPosition(endEffector_L))
    # print(2, endEffector_R, sim.getJointPosition(endEffector_R))
    #
    # # leave dumb bell
    # trajectoryL = list(np.linspace(np.array([0.25514343, 0.42992666, 0.40]), np.array([0.25514343, 0.42992666, 0.08]), 8))
    # trajectoryR = list(np.linspace(np.array([0.44914699, 0.22027466, 0.40]), np.array([0.44914699, 0.22027466, 0.08]), 8))
    # for i, j in zip(trajectoryL, trajectoryR):
    #     sim.move_two_joints_with_PD(endEffector_L, endEffector_R, i, j, orientation=[0, 0, 0])


    # basis
    # move waist
    # sim.moveJoint("CHEST_JOINT0", np.deg2rad(42), 0.0, 0.0010)
    # print(1, endEffector_L, sim.getJointPosition(endEffector_L))
    # print(2, endEffector_R, sim.getJointPosition(endEffector_R))
    #
    # # leave dumb bell
    # trajectoryL = list(np.linspace(np.array([0.2230598, 0.44803498, 0.32672087]), np.array([0.2230598, 0.44803498, 0.08]), 8))
    # trajectoryR = list(np.linspace(np.array([0.43216633, 0.25286098, 0.32442961]), np.array([0.43216633, 0.26286098, 0.08]), 8))
    # for i, j in zip(trajectoryL, trajectoryR):
    #     sim.move_two_joints_with_PD(endEffector_L, endEffector_R, i, j, orientation=[0, 0, 0])




    # trajectoryL = list(np.linspace(np.array([0.19, 0.43, 0.07]), np.array([0.15, 0.48, 0.07]), 6))
    # trajectoryR = list(np.linspace(np.array([0.40, 0.25, 0.08]), np.array([0.30, 0.16, 0.08]), 6))
    # for i, j in zip(trajectoryL, trajectoryR):
    #     sim.move_2_joints_with_PD(endEffector_L, endEffector_R, i, j, orientation=[0, 0, 0]) # [0,0,1]

#######
# endEffector_L = 'LHAND'
# endEffector_R = 'RHAND'
#
# trajectoryL = list(np.linspace(np.array([[0.50485951, 0.23009971, 0.18199865]]), np.array([0.46, 0.09, 0.21]), 5))
# trajectoryR = list(np.linspace(np.array([[0.50488567, -0.230092, 0.18197175]]), np.array([0.46, -0.05, 0.21]), 5))
# for i, j in zip(trajectoryL, trajectoryR):
#     sim.move_two_joints_with_PD(endEffector_L, endEffector_R, i, j, orientation=[0, 0, 0])
#
# sim.moveJoint("LARM_JOINT5", np.deg2rad(-100), 0.0, 0.0014)  # lower speed, more steps
# sim.moveJoint("RARM_JOINT5", np.deg2rad(100), 0.0, 0.0017)
#
# trajectoryL = list(np.linspace(np.array([0.46, 0.09, 0.21]), np.array([0.46, 0.09, 0.43]), 5))
# trajectoryR = list(np.linspace(np.array([0.46, -0.05, 0.21]), np.array([0.46, -0.1, 0.43]), 5))
# for i, j in zip(trajectoryL, trajectoryR):
#     sim.move_two_joints_with_PD(endEffector_L, endEffector_R, i, j, orientation=[0, 0, 0])
#
# # move waist
# sim.moveJoint("CHEST_JOINT0", np.deg2rad(40), 0.0, 0.0013)
# print(endEffector_L, sim.getJointPosition(endEffector_L))
# print(endEffector_R, sim.getJointPosition(endEffector_R))
#
# # leave dumb bell
# trajectoryL = list(np.linspace(np.array([0.21, 0.43, 0.33]), np.array([0.23, 0.43, 0.07]), 6))
# trajectoryR = list(np.linspace(np.array([0.41, 0.24, 0.33]), np.array([0.44, 0.25, 0.09]), 6))
# for i, j in zip(trajectoryL, trajectoryR):
#     sim.move_two_joints_with_PD(endEffector_L, endEffector_R, i, j, orientation=[0, 0, 0])
#
# print(11, endEffector_L, sim.getJointPosition(endEffector_L))
# print(22, endEffector_R, sim.getJointPosition(endEffector_R))
#
# actual = sim.p.getBasePositionAndOrientation(cubeId)
# expected = sim.p.getBasePositionAndOrientation(targetId)
# print('Position', actual[0], expected[0])
# print('Orientation', sim.p.getEulerFromQuaternion(actual[1]), sim.p.getEulerFromQuaternion(expected[1]))

tableId, cubeId, targetId = getReadyForTask()
solution()