import scipy.interpolate
from scipy.spatial.transform import Rotation as npRotation
from scipy.special import comb
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import time
import yaml

from Pybullet_Simulation_base import Simulation_base

class Simulation(Simulation_base):
    """A Bullet simulation involving Nextage robot"""

    def __init__(self, pybulletConfigs, robotConfigs, refVect=None):
        """Constructor
        Creates a simulation instance with Nextage robot.
        For the keyword arguments, please see in the Pybullet_Simulation_base.py
        """
        super().__init__(pybulletConfigs, robotConfigs)
        if refVect:
            self.refVector = np.array(refVect)
        else:
            self.refVector = np.array([1, 0, 0])

    ########## Task 1: Kinematics ##########
    # Task 1.1 Forward Kinematics
    jointRotationAxis = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.zeros(3),  # Fixed joint
        'CHEST_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT0': np.array([0, 0, 1]),
        'LARM_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT2': np.array([0, 1, 0]),
        'LARM_JOINT3': np.array([1, 0, 0]),
        'LARM_JOINT4': np.array([0, 1, 0]),
        'LARM_JOINT5': np.array([0, 0, 1]),
        'RARM_JOINT0': np.array([0, 0, 1]),
        'RARM_JOINT1': np.array([0, 1, 0]),
        'RARM_JOINT2': np.array([0, 1, 0]),
        'RARM_JOINT3': np.array([1, 0, 0]),
        'RARM_JOINT4': np.array([0, 1, 0]),
        'RARM_JOINT5': np.array([0, 0, 1]),
        'RHAND'      : np.array([1, 0, 0]),
        'LHAND'      : np.array([1, 0, 0])
    }

    frameTranslationFromParent = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.zeros(3),  # Fixed joint
        'CHEST_JOINT0': np.array([0, 0, 0.267]),
        'HEAD_JOINT0': np.array([0, 0, 0.302]),
        'HEAD_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT0': np.array([0.04, 0.135, 0.1015]),
        'LARM_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT2': np.array([0, 0.095, -0.25]),
        'LARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'LARM_JOINT4': np.array([0.1495, 0, 0]),
        'LARM_JOINT5': np.array([0, 0, -0.1335]),
        'RARM_JOINT0': np.array([0.04, -0.135, 0.1015]),
        'RARM_JOINT1': np.array([0, 0, 0.066]),
        'RARM_JOINT2': np.array([0, -0.095, -0.25]),
        'RARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'RARM_JOINT4': np.array([0.1495, 0, 0]),
        'RARM_JOINT5': np.array([0, 0, -0.1335]),
        'RHAND': np.array([0, 0, 0]), #-0.08])
        'LHAND': np.array([0, 0, 0])  #-0.08])
    }

    def getJointRotationalMatrix(self, jointName=None, theta=None):
        """
            Returns the 3x3 rotation matrix for a joint from the axis-angle representation,
            where the axis is given by the revolution axis of the joint and the angle is theta.
        """
        if jointName == None:
            raise Exception("[getJointRotationalMatrix] \
                Must provide a joint in order to compute the rotational matrix!")
        axis = self.jointRotationAxis[jointName]

        # z axis
        if all(axis == [0, 0, 1]):
            return np.matrix([[math.cos(theta), -math.sin(theta), 0],
                              [math.sin(theta), math.cos(theta), 0],
                              [0, 0, 1]])
        # y axis
        elif all(axis == [0, 1, 0]):
            return np.matrix([[math.cos(theta), 0, math.sin(theta)],
                              [0, 1, 0],
                              [-math.sin(theta), 0, math.cos(theta)]])
        # x axis
        elif all(axis == [1, 0, 0]):
            return np.matrix([[1, 0, 0],
                              [0, math.cos(theta), -math.sin(theta)],
                              [0, math.sin(theta), math.cos(theta)]])
        else:
            return np.identity(3)

    allJoints = ['CHEST_JOINT0', 'base_to_dummy', 'base_to_waist', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2',
                 'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2',
                 'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5']

    def getTransformationMatrices(self):
        """
            Returns the homogeneous transformation matrices for each joint as a dictionary of matrices.
        """
        transformationMatrices = {
            'base_to_dummy': np.identity(4), # np.zeros([4, 4]),  # Virtual joint
            'base_to_waist': np.identity(4), # np.zeros([4, 4]),  # Fixed joint # set to 0 rotation so it will be fixed to base
            'CHEST_JOINT0': np.array([]),
            'HEAD_JOINT0': np.array([]),
            'HEAD_JOINT1': np.array([]),
            'LARM_JOINT0': np.array([]),
            'LARM_JOINT1': np.array([]),
            'LARM_JOINT2': np.array([]),
            'LARM_JOINT3': np.array([]),
            'LARM_JOINT4': np.array([]),
            'LARM_JOINT5': np.array([]),
            'RARM_JOINT0': np.array([]),
            'RARM_JOINT1': np.array([]),
            'RARM_JOINT2': np.array([]),
            'RARM_JOINT3': np.array([]),
            'RARM_JOINT4': np.array([]),
            'RARM_JOINT5': np.array([]),
            'RHAND': np.identity(4), # every joint that does not move returns an identity matrix
            'LHAND': np.identity(4),
        }
        # Compute homogeneous transformation matrices (4,4)
        for jointName in transformationMatrices:

            # We hardcoded static joints to identity matrices for base and endEffectors
            if not any(joint in jointName for joint in ['base', 'RHAND', 'LHAND']):
                # Get Rotational Matrix R (3x3) and Translation Vector (3x1)
                jointRotationalMatrix = self.getJointRotationalMatrix(jointName, self.getJointPos(jointName))
                translationVector = np.matrix(self.frameTranslationFromParent[jointName]).T

                # stack Rotational Matrix and Translation Vector (3,4)
                rotation_translation = np.hstack((jointRotationalMatrix, translationVector))
                # add the Augmentation Component (4,1) at the bottom
                transformationMatrices[jointName] = np.vstack([rotation_translation, [0, 0, 0, 1]])

        return transformationMatrices

    def getJointLocationAndOrientation(self, jointName):
        """
            Returns the position and rotation matrix of each joint using Forward Kinematics 
            according to the topology of the Nextage robot.
        """
        # Remember to multiply the transformation matrices following the kinematic chain for each arm.
        transformationMatrices = self.getTransformationMatrices()
        chestJointMatrix = transformationMatrices.get('CHEST_JOINT0')  # base_to_waist

        # chestJointMatrix = transformationMatrices.get('base_to_waist')
        # chestJointMatrix = chestJointMatrix @ transformationMatrices.get('CHEST_JOINT0')

        # If joint is eff then compute kinematic chain else chest_joint @ arm_joint
        if jointName == 'LHAND':
            limbItr = 'LARM_JOINT5'
        elif jointName == 'RHAND':
            limbItr = 'RARM_JOINT5'
        else:
            limbItr = jointName

        # don't include LHAND, RHAND, CHEST, dummy, waist transformation matrices in the kinematic chain (so use from Joint0 to 5)
        # if limbItr is chest, dummy, waist then return chest transformation, jac won't call for dummy, waist, head
        if any(limbItr in joint for joint in self.robotLimbs):
            # calculate kinematic chain
            for i in range(0, int(limbItr[-1]) + 1):
                chestJointMatrix = chestJointMatrix @ transformationMatrices[limbItr[0:10] + str(i)]

        # translation vector
        pos = np.matrix([chestJointMatrix[0, 3], chestJointMatrix[1, 3], chestJointMatrix[2, 3]])
        # rotational matrix
        rotmat = np.matrix(chestJointMatrix[0:3, 0:3])

        eulerAngles = npRotation.from_rotvec(rotmat).as_euler('xyz')
        # return the pose of a joint (= position/pos/translation + orientation/rotmat/rotation)
        return pos, eulerAngles #rotmat

    def getJointPosition(self, jointName):
        """Get the position of a joint in the world frame, leave this unchanged please."""
        return self.getJointLocationAndOrientation(jointName)[0]

    def getJointOrientation(self, jointName, ref=None):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        if ref is None:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.refVector).squeeze()
        else:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ ref).squeeze()

    def getJointAxis(self, jointName):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.jointRotationAxis[jointName]).squeeze()

    def jacobianMatrix(self, endEffector):
        """Calculate the Jacobian Matrix for the Nextage Robot."""
        # You can implement the cross product yourself or use calculateJacobian(). for all 15 joints (3,15)
        allJoints = ['base_to_dummy', 'base_to_waist', 'LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2',
         'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5', 'RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2',
         'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5']

        jaco = []
        # initialise position of relative vector (from chest to eff)
        initialJoint = 'CHEST_JOINT0'
        # a1 x (pEff - p1)  joint position pEff
        jacobianPos = np.cross(self.getJointAxis(initialJoint), (
                self.getJointPosition(endEffector) - self.getJointPosition(initialJoint)))  # (1,3)
        jacobianOri = np.cross(self.getJointAxis(initialJoint), self.getJointAxis(endEffector)).reshape(1, 3)
        jaco.append(np.hstack([jacobianPos, jacobianOri]))

        # compute Jacobian for the kinematic chain
        # for each joint calculate [ai x (pEff - pi)]
        for nextJoint in allJoints:
            if (endEffector[0] == nextJoint[0] == 'L') or (endEffector[0] == nextJoint[0] == 'R'):
                nextJacobianPos = np.cross(self.getJointAxis(nextJoint), (
                        self.getJointPosition(endEffector) - self.getJointPosition(nextJoint)))
                nextJacobianOri = np.cross(self.getJointAxis(nextJoint), self.getJointAxis(endEffector)).reshape(1, 3)
            else:
                nextJacobianPos = [0, 0, 0]  # (3,1)
                nextJacobianOri = [0, 0, 0]
            jaco.append(np.hstack([nextJacobianPos, nextJacobianOri]))

        ###### Task 1 - calculation of jacobian using calculateJacobian() ###### use only for task 1
        # if endEffector == 'LHAND':
        #     jointName = 'LARM_JOINT5'
        # elif endEffector == 'RHAND':
        #     jointName = 'RARM_JOINT5'
        # else:
        #     jointName = endEffector
        # mpos = []
        # for joint in self.allJoints:
        #     mpos.append(self.getJointPos(joint))
        # com_trn = self.getLinkCoM(jointName) - self.getLinkFramePos(jointName)
        # j_geo, j_rot = self.p.calculateJacobian(self.robot, self.jointIds[jointName], com_trn, mpos, [0.0] * len(mpos),
        #                                         [0.0] * len(mpos), )
        # pybullet_jacobian = np.vstack([j_geo, j_rot])
        ######

        my_jacobian = np.vstack(jaco).T
        return my_jacobian # (6,15)

    def inverseKinematics(self, step_index, step_positions, step_orientations, endEffector, targetPosition,
                          orientation=None, frame=None):
        """Your IK solver \\
        Arguments: \\
            endEffector: the jointName the end-effector \\
            targetPosition: final destination the the end-effector \\
        Keywork Arguments: \\
            orientation: the desired orientation of the end-effector
                         together with its parent link \\
            speed: how fast the end-effector should move (m/s) \\
            orientation: the desired orientation \\
            compensationRatio: naive gravity compensation ratio \\
            debugLine: optional \\
            verbose: optional \\
        Return: \\
            List of x_refs
        """
        # get current endEffector position
        effPos = self.getJointPosition(endEffector)            # (1,3)
        deltaStepPos = step_positions[step_index, :] - effPos  # (1,3) = (y_target - y)

        if orientation is None:
            deltaStepRot = [0, 0, 0]
        else:
            deltaStepRot = step_orientations[step_index, :] - self.getJointAxis(endEffector)# self.getJointOrientation(endEffector)
        deltaStep = np.hstack([deltaStepPos, np.matrix([deltaStepRot])])

        J = self.jacobianMatrix(endEffector) # (6,15)
        JpInverse = np.linalg.pinv(J)        # (15,6)
        deltaTheta = JpInverse @ deltaStep.T # (15,6)x(6,1) = (15,1) = pinv(J) * (y_target - y)

        return deltaTheta

    def move_without_PD(self, endEffector, targetPosition, speed=0.01, orientation=None, threshold=1e-3, maxIter=3000,
                        debug=False, verbose=False):
        """
        Move joints using Inverse Kinematics solver (without using PD control).
        This method should update joint states directly.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        # iterate through joints and update joint states based on IK solver

        # get init end-effector pos using forward kinematics
        effPos = self.getJointPosition(endEffector)  # (1,3)
        distance = np.linalg.norm(targetPosition - effPos)
        IK_steps = int(distance / speed)
        if IK_steps < 100:
            IK_steps = 100
        step_positions = np.linspace(effPos, targetPosition, IK_steps)

        pltTime = []
        pltDisArray = []
        pltDistanceMatrix = np.empty((0, 3), int)
        for step_index in range(IK_steps):

            threshold = 1e-4
            e = np.linalg.norm(targetPosition - self.getJointPosition(endEffector)) / IK_steps
            if e < threshold:
                break

            # IK returns (15,1) vector deltaTheta:
            deltaTheta = self.inverseKinematics(step_index, step_positions, [], endEffector, targetPosition, None)

            # every time theta updates for all 15 joints, robot state updates
            for jointName, deltaTheta_step in zip(self.allJoints, deltaTheta):
                self.jointTargetPos[jointName] = self.getJointPos(jointName) + deltaTheta_step

            # update robot states
            self.tick_without_PD()

            # variables to plot graphs
            pltTime.append(step_index)
            pltDistance = targetPosition - self.getJointPosition(endEffector)
            pltDistanceMatrix = np.append(pltDistanceMatrix, pltDistance, axis=0)
            pltDisArray = np.append(pltDisArray, np.linalg.norm(targetPosition - self.getJointPosition(endEffector)))

        return pltTime, pltDistanceMatrix, pltDisArray

    def tick_without_PD(self):
        """Ticks one step of simulation without PD control. """
        # Iterate through all joints and update joint states.
        # For each joint, you can use the shared variable self.jointTargetPos.

        for jointName in self.allJoints:
            self.p.resetJointState(self.robot, self.jointIds[jointName], self.jointTargetPos[jointName])

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)


    ########## Task 2: Dynamics ##########
    # Task 2.1 PD Controller
    def calculateTorque(self, x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd):
        """ This method implements the closed-loop control \\
        Arguments: \\
            x_ref - the target position \\
            x_real - current position \\
            dx_ref - target velocity \\
            dx_real - current velocity \\
            integral - integral term (set to 0 for PD control) \\
            kp - proportional gain \\
            kd - derivative gain \\
            ki - integral gain \\
        Returns: \\
            u(t) - the manipulation signal
        """
        ut = kp * (x_ref - x_real) - kd * dx_real
        return ut

    ###### Task 2.2 Joint Manipulation: Only to tune PD gains and plot graphs
    def moveJointToy(self, joint, targetPosition, targetVelocity, verbose=False):
        """ This method moves a joint with your PD controller. \\
        Arguments: \\
            joint - the name of the joint \\
            targetPos - target joint position \\
            targetVel - target joint velocity
        """
        def toy_tick(x_ref, x_real, dx_ref, dx_real, integral):
            # loads your PID gains
            jointController = self.jointControllers[joint]
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Start your code here: ###
            # Calculate the torque with the above method you've made
            torque = self.calculateTorque(x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd)
            pltTorque.append(torque)
            ### To here ###

            # send the manipulation signal to the joint
            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )
            # calculate the physics and update the world
            self.p.stepSimulation()
            time.sleep(self.dt)

        targetPosition, targetVelocity = float(targetPosition), float(targetVelocity)
        # disable joint velocity controller before apply a torque
        self.disableVelocityController(joint)
        # logging for the graph
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = [], [], [], [], [], []

        steps = 500
        x_old = self.getJointPos(joint)

        for step in range(steps):
            x_real = self.getJointPos(joint)
            dx_real = (x_real-x_old) / self.dt
            e = targetPosition - x_real
            print(e)
            toy_tick(targetPosition, x_real, targetVelocity, dx_real, 0)
            pltPosition.append(x_real)
            pltTarget.append(targetPosition)
            pltVelocity.append(dx_real)
            x_old = x_real

        pltTime = np.arange(steps) * self.dt
        return pltTime, pltTarget, pltTorque, pltTime, pltPosition, pltVelocity

    ##### Task 3: Move one joint using tick()
    def moveJoint(self, joint, targetPosition, targetVelocity, speed=0.0015, verbose=False):
        """ This method moves a joint with your PD controller. \\
        Arguments: \\
            joint - the name of the joint \\
            targetPos - target joint position \\
            targetVel - target joint velocity
        """
        time.sleep(self.dt)
        targetPosition, targetVelocity = float(targetPosition), float(targetVelocity)

        # disable joint velocity controller before apply a torque
        self.disableVelocityController(joint)
        num_steps = np.linalg.norm(targetPosition - self.getJointPos(joint)) / speed
        targetPositionSteps = np.linspace(self.getJointPos(joint), targetPosition, int(num_steps))

        self.jointPositionOld[joint] = self.getJointPos(joint)
        for targetPosition in targetPositionSteps:
            self.jointTargetPos[joint] = targetPosition
            self.tick()

    pltTorque = []
    control_cycles = 4.0
    ###### Task 3.1 Pushing
    def move_with_PD(self, endEffector, targetPosition, speed=0.01, orientation=None, threshold=1e-3, maxIter=3000,
                     debug=False, verbose=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        # Iterate through joints and use states from IK solver as reference states in PD controller.
        # Perform iterations to track reference states using PD controller until reaching
        # max iterations or position threshold.
        distance = np.linalg.norm(targetPosition - self.getJointPosition(endEffector))
        speed = 0.01
        IK_steps = int(distance / speed)
        if IK_steps < 100:
            IK_steps = 100
        IK_steps = 100 # <297

        # get init end-effector pos using forward kinematics
        effPos = self.getJointPosition(endEffector)
        step_positions = np.linspace(effPos, targetPosition, IK_steps)
        step_orientations = []
        if orientation is not None:
            step_orientations = np.linspace(self.getJointOrientation(endEffector), orientation, IK_steps)

        pltTime = []
        pltDistance = []
        for step_index in range(IK_steps):

            threshold = 1e-3
            e = np.linalg.norm(targetPosition - self.getJointPosition(endEffector)) / IK_steps
            # print(targetPosition, self.getJointPosition(endEffector))

            # variables to plot graphs
            pltTime.append(step_index)
            pltDistance.append(np.linalg.norm(targetPosition - self.getJointPosition(endEffector)))

            if e < threshold:
                break
            deltaTheta = self.inverseKinematics(step_index, step_positions, step_orientations, endEffector, targetPosition, orientation)

            # every time theta updates for all 15 joints, robot state updates
            for jointName, deltaTheta_step in zip(self.allJoints, deltaTheta):
                # before updating the angle, get the old pos
                self.jointPositionOld[jointName] = self.getJointPos(jointName)
                # set the target pos
                self.jointTargetPos[jointName] = self.getJointPos(jointName) + deltaTheta_step

            self.control_cycles = self.controlFrequency / self.updateFrequency
            for i in range(int(self.control_cycles)):
                self.tick()

        return pltTime, pltDistance
        # Hint: here you can add extra steps if you want to allow your PD
        # controller to converge to the final target position after performing
        # all IK iterations (optional).

    ###### Task 3.2 Grasping & Docking
    def move_two_joints_with_PD(self, endEffector1, endEffector2, targetPosition1, targetPosition2, speed=0.01,
                                orientation=None, threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        # Iterate through joints and use states from IK solver as reference states in PD controller.
        # Perform iterations to track reference states using PD controller until reaching
        # max iterations or position threshold.

        distance1 = np.linalg.norm(targetPosition1 - self.getJointPosition(endEffector1))
        distance2 = np.linalg.norm(targetPosition2 - self.getJointPosition(endEffector2))

        IK_steps = int(distance1 / speed)
        if IK_steps < 100:
            IK_steps = 100

        # get init end-effector pos using forward kinematics
        step_positions1 = np.linspace(self.getJointPosition(endEffector1), targetPosition1, IK_steps)
        step_positions2 = np.linspace(self.getJointPosition(endEffector2), targetPosition2, IK_steps)

        step_orientations1 = []
        step_orientations2 = []
        if orientation is not None:             # getJointAxis
            step_orientations1 = np.linspace(self.getJointOrientation(endEffector1), orientation, IK_steps)
            step_orientations2 = np.linspace(self.getJointOrientation(endEffector2), orientation, IK_steps)

        for step_index in range(IK_steps):

            threshold = 1e-3
            e1 = np.linalg.norm(targetPosition1 - self.getJointPosition(endEffector1)) / IK_steps
            e2 = np.linalg.norm(targetPosition2 - self.getJointPosition(endEffector2)) / IK_steps
            # print(targetPosition, self.getJointPosition(endEffector))

            if e1 < threshold or e2 < threshold:
                break
            deltaTheta1 = self.inverseKinematics(step_index, step_positions1, step_orientations1, endEffector1, targetPosition1, orientation)
            deltaTheta2 = self.inverseKinematics(step_index, step_positions2, step_orientations2, endEffector2, targetPosition2, orientation)

            deltaTheta = np.vstack([deltaTheta1[:9], deltaTheta2[9:]])

            # every time theta updates for all 15 joints, robot state updates
            for jointName, deltaTheta_step in zip(self.allJoints, deltaTheta):
                # before updating the angle, get the old pos
                self.jointPositionOld[jointName] = self.getJointPos(jointName)
                # set the target pos
                self.jointTargetPos[jointName] = self.getJointPos(jointName) + deltaTheta_step

            self.control_cycles = self.controlFrequency / self.updateFrequency
            for i in range(int(self.control_cycles)):
                self.tick()

        return [], []

    def tick(self):
        """Ticks one step of simulation using PD control."""
        # Iterate through all joints and update joint states using PD control.
        for joint in self.joints:
            # skip dummy joints (world to base joint)
            jointController = self.jointControllers[joint]
            if jointController == 'SKIP_THIS_JOINT':
                continue

            # disable joint velocity controller before apply a torque
            self.disableVelocityController(joint)

            # loads your PID gains
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Implement your code from here ... ###
            x_old = self.jointPositionOld[joint]
            x_real = self.getJointPos(joint)
            dx_real = (x_real - x_old) / (self.dt * self.control_cycles)
            x_ref = self.jointTargetPos[joint]

            e = x_ref - x_real
            # print(e)

            # calculate the torque according to the desired target
            torque = self.calculateTorque(x_ref, x_real, 0.0, dx_real, 0.0, kp, ki, kd)
            self.jointPositionOld[joint] = x_real
            ### ... to here ###

            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )

            # Gravity compensation
            # A naive gravity compensation is provided for you
            # If you have embeded a better compensation, feel free to modify
            compensation = self.jointGravCompensation[joint]
            self.p.applyExternalForce(
                objectUniqueId=self.robot,
                linkIndex=self.jointIds[joint],
                forceObj=[0, 0, -compensation],
                posObj=self.getLinkCoM(joint),
                flags=self.p.WORLD_FRAME
            )
            # Gravity compensation ends here

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)

    ########## Task 3: Robot Manipulation ##########
    def cubic_interpolation(self, waypoints):
        """
        Given a set of control points, return the
        cubic spline defined by the control points,
        sampled nTimes=50 along the curve.
        """
        tt = np.linspace(0, 1, len(waypoints))
        x = CubicSpline(tt, waypoints[:, 0])  # x waypoints
        y = CubicSpline(tt, waypoints[:, 1])  # y waypoints
        z = CubicSpline(tt, waypoints[:, 2])  # z waypoints

        t = np.linspace(0, 1)
        return np.vstack([x(t), y(t), z(t)])

 ### END
