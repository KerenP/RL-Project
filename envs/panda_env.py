import time

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random


MAX_EPISODE_LEN = 5

class PandaEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    # For main run GUI mode, for learn_ppo run DIRECT mode
    def __init__(self, gui=False):
        self.step_counter = 0
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        # Define action space, first 4 parameters define action type. last 3 are coordinates in xyz
        self.action_space = spaces.Box(np.array([-1]*7), np.array([1]*7))
        # state_robot(3) + state_fingers(2) + state_object(3)
        self.observation_space = spaces.Box(np.array([-1]*8), np.array([1]*8))

    def normalize(self, location):
        x_norm = location[0] * 0.25 + 0.75
        y_norm = location[1] * 0.2
        z_norm = (location[2] + 1) * 0.3
        return x_norm, y_norm, z_norm

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])

        dx, dy, dz = self.normalize(action[4:])
        current_pose = p.getLinkState(self.pandaUid, 11)
        current_position = current_pose[0]
        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        mov_type = np.argmax(action[0:4])
        if mov_type == 0:  # get above object
            new_position = [dx, dy, current_position[2]]
            fingers = 1
        elif mov_type == 1:  # close fingers
            new_position = current_position
            fingers = -1
        elif mov_type == 2:  # lift object
            new_position = [current_position[0], current_position[1], 0.5]
            fingers = -1
        elif mov_type == 3:  # get down
            new_position = [current_position[0], current_position[1], dz]
            fingers = 1

        jointPoses = p.calculateInverseKinematics(self.pandaUid, 11, new_position, orientation)[0:7]

        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9, 10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])

        for i in range(100):
            p.stepSimulation()

        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        finger1 = p.getJointState(self.pandaUid, 9)
        finger2 = p.getJointState(self.pandaUid, 10)
        state_fingers = (finger1[0], finger2[0])

        distance = np.array(state_robot) - np.array(state_object)

        l1_norm = np.linalg.norm(distance, 1)

        finger_tension = finger1[3] + finger2[3]
        finger_reward = 0
        if finger_tension < -1:
            finger_reward = math.log(-finger_tension, 10)*(1-l1_norm)/10

        height_reward = min((state_object[2]-0.05)*10, 0)

        reward = -l1_norm+finger_reward+height_reward

        done = False

        if state_object[2] > 0.3:
            reward = 10
            done = True

        self.step_counter += 1

        if self.step_counter > MAX_EPISODE_LEN:
            done = True

        info = {'object_position': state_object}
        self.observation = state_robot + state_fingers + state_object
        return np.array(self.observation).astype(np.float32), reward, done, info

    def reset(self):
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        urdfRootPath=pybullet_data.getDataPath()
        p.setGravity(0,0,-10)

        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

        rest_poses = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]

        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
        for i in range(7):
            p.resetJointState(self.pandaUid,i, rest_poses[i])
        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid,10, 0.08)
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])

        trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])

        state_object = (0.75, 0, 0.05)
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        self.observation = state_robot + state_fingers + state_object
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        return np.array(self.observation).astype(np.float32)

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        ...
