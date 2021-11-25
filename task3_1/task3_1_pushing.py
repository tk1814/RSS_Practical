import subprocess, math, time, sys, os, numpy as np
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

taskId = 3.1

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
    "cameraSettings": (1.07, 90.0, -52.8, (0.07, 0.01, 0.76))
}
robotConfigs = {
    "robotPath": core_path + "/nextagea_description/urdf/NextageaOpen.urdf",
    "robotPIDConfigs": core_path + "/PD_gains.yaml",
    "robotStartPos": [0, 0, 0.85],
    "robotStartOrientation": [0, 0, 0, 1],
    "fixedBase": True,
    "colored": False
}
# ref = [0, 0, 1] , refVect=ref
sim = Simulation(pybulletConfigs, robotConfigs)

##### Please leave this function unchanged, feel free to modify others #####
def getReadyForTask():
    global finalTargetPos
    # compile urdfs
    finalTargetPos = np.array([0.7, 0.00, 0.91])
    urdf_compiler_path = core_path + "/urdf_compiler.py"
    subprocess.call([urdf_compiler_path,
                     "-o", abs_path+"/lib/task_urdfs/task3_1_target_compiled.urdf",
                     abs_path+"/lib/task_urdfs/task3_1_target.urdf"])

    sim.p.resetJointState(bodyUniqueId=1, jointIndex=12, targetValue=-0.4)
    sim.p.resetJointState(bodyUniqueId=1, jointIndex=6, targetValue=-0.4)
    # load the table in front of the robot
    tableId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/table/table_taller.urdf",
        basePosition        = [0.8, 0, 0],
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, math.pi/2]),
        useFixedBase        = True,
        globalScaling       = 1.4
    )
    cubeId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/cubes/cube_small.urdf",
        basePosition        = [0.33, 0, 1.0],
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase        = False,
        globalScaling       = 1.4
    )
    sim.p.resetVisualShapeData(cubeId, -1, rgbaColor=[1, 1, 0, 1])

    targetId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/task3_1_target_compiled.urdf",
        basePosition        = finalTargetPos,
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, math.pi]),
        useFixedBase        = True,
        globalScaling       = 1
    )

    return tableId, cubeId, targetId


def solution():
    endEffector = "LHAND"
    waypoints = np.array([[0.50497461, 0.23, 0.18214933], [0.25, 0.30, 0.15], [0.1, -0.02, 0.14], [0.49, 0.01, 0.065], [0.60, 0.0, 0.18214933]])
    trajectory = sim.cubic_interpolation(waypoints)

    for i in range(len(trajectory[0])):
        sim.move_with_PD(endEffector, trajectory[:, i], orientation=[1, 0, 0])

    actual = sim.p.getBasePositionAndOrientation(cubeId)
    expected = sim.p.getBasePositionAndOrientation(targetId)
    print('Actual Position: {}, Expected: {}'.format(actual[0], expected[0]))
    # print('Position Error: {}'.format(np.asarray(expected[0])-np.asarray(actual[0])))
    print('Error: {}'.format(np.linalg.norm(np.asarray(expected[0]) - np.asarray(actual[0]))))




    # print('Error: {}'.format(np.linalg.norm(np.asarray(expected[0]), np.asarray(actual[0]))))


    # print('Actual Orientation: {}, Expected: {}'.format(sim.p.getEulerFromQuaternion(actual[1]), sim.p.getEulerFromQuaternion(expected[1])))
    # print('Orientation Error: {}'.format(np.asarray(sim.p.getEulerFromQuaternion(expected[1]))-np.asarray(sim.p.getEulerFromQuaternion(actual[1]))))


tableId, cubeId, targetId = getReadyForTask()
solution()