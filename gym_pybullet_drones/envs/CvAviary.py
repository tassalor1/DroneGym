import numpy as np
import pybullet as p
import torch
import cv2
from gymnasium import spaces
import os
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

from gym_pybullet_drones.midasModel.run import run  

class DepthDistance:
    def __init__(self, img, best_bbox=None):
        self.best_bbox = best_bbox
        self.img = img

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    def slice_img(self, img):
        height, width = img.shape[:2]

        if self.best_bbox is None:
            # No bounding box available, crop to center
            crop_size = 450
            x_center = width // 2
            y_center = height // 2

            x_min = max(0, x_center - crop_size // 2)
            y_min = max(0, y_center - crop_size // 2)
            x_max = min(width, x_center + crop_size // 2)
            y_max = min(height, y_center + crop_size // 2)
            print("Centre of image for MiDaS depth estimation")
        else:
            # Use the existing bounding box logic
            best_bbox1 = list(map(int, self.best_bbox))
            x_min = max(0, min(best_bbox1[0], width))
            y_min = max(0, min(best_bbox1[1], height))
            x_max = max(0, min(best_bbox1[2], width))
            y_max = max(0, min(best_bbox1[3], height))
            print("Using best bounding box for MiDaS depth estimation")

        return img[y_min:y_max, x_min:x_max]

    def run_model(self):
        default_models = {
            'dpt_swin2_tiny_256': 'midasModel/weights/dpt_swin2_tiny_256.pt',
        }
        sliced_img = self.slice_img(self.img)

        # Set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        median_depth, prediction = run(image=sliced_img,
                                       model_path=default_models['dpt_swin2_tiny_256'],
                                       model_type='dpt_swin2_tiny_256',
                                       )

        return median_depth, prediction

class CVAviary(BaseRLAviary):
    """A custom aviary environment with computer vision capabilities."""

    def __init__(self,
                 drone_model=DroneModel.CF2X,
                 num_drones=1,
                 neighbourhood_radius=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics=Physics.PYB,
                 pyb_freq=240,
                 ctrl_freq=48,
                 gui=False,
                 record=False,
                 obs=False,
                 act: ActionType=ActionType.RPM,
                 ):
        
        # camera parameters
        self.IMG_WIDTH = 640
        self.IMG_HEIGHT = 480
        self.IMG_RES = np.array([self.IMG_WIDTH, self.IMG_HEIGHT])
        self.CAM_FOV = 90
        self.CAM_NEAR = 0.1
        self.CAM_FAR = 1000

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         )
        yolov5_path = os.path.join("/Users/connoranthow/dev/p_dev/DroneGym/gym_pybullet_drones", "yolov5")

        self.yolo_model = torch.hub.load(yolov5_path, 'custom',
                                         path='gym_pybullet_drones/yolov5/weights/best.pt', source='local')

        # Determine the camera link index
        self.CAMERA_LINK_INDEX = self._getCameraLinkIndex()

    def _getCameraLinkIndex(self):
        # Get the index of the 'camera_link' in the drone model
        drone_id = self.DRONE_IDS[0]
        num_joints = p.getNumJoints(drone_id, physicsClientId=self.CLIENT)
        for i in range(num_joints):
            joint_info = p.getJointInfo(drone_id, i, physicsClientId=self.CLIENT)
            joint_name = joint_info[12].decode('UTF-8')
            if joint_name == 'camera_link':
                return i
        raise ValueError("Camera link not found in the drone model.")

    def _actionSpace(self):
        # Define the action space (e.g., velocity control)
        max_velocity = 3  # Max velocity in m/s
        return spaces.Box(low=-max_velocity, high=max_velocity, shape=(3,), dtype=np.float32)

    def _observationSpace(self):
        # Define the observation space
        max_detections = 1  # Processing only the best detection
        obs_dict = {
            'rgb': spaces.Box(low=0, high=255, shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3), dtype=np.uint8),
            'detections': spaces.Box(low=-np.inf, high=np.inf, shape=(max_detections, 7), dtype=np.float32),
        }
        return spaces.Dict(obs_dict)

    def _computeObs(self):
        obs = {}
        # Capture the image
        rgb_array = self._getDroneImage(0)  # Assuming single drone
        obs['rgb'] = rgb_array

        input_image = self.preprocess_image(rgb_array)

        # Run the YOLO model
        results = self.yolo_model(input_image)
        detections = results.xyxy[0].cpu().numpy()

        # Initialize the detections array
        max_detections = self.observation_space['detections'].shape[0]
        padded_detections = np.zeros((max_detections, 7), dtype=np.float32)

        # Process the best detection
        if len(detections) > 0:
            # Sort detections by confidence
            detections = detections[detections[:, 4].argsort()[::-1]]  
            best_detection = detections[0]
            bbox = best_detection[:4]  # x1, y1, x2, y2

            # Compute the estimated distance using DepthDistance
            depth_distance = DepthDistance(img=rgb_array, best_bbox=bbox)
            median_depth, prediction = depth_distance.run_model()

            # Extend the detection to include the median depth
            detection_with_depth = np.concatenate((best_detection, [median_depth]))

            # Store the detection
            padded_detections[0] = detection_with_depth
        else:
            # No detection, leave zeros
            pass

        obs['detections'] = padded_detections

        # Store the last observation for reward computation
        self.last_obs = obs

        return obs

    def _preprocessAction(self, action):
        # Convert the action to motor RPMs or velocity commands
        # Implement velocity control for the drone
        desired_velocity = action  
        rpm = self._velocityToRPM(desired_velocity)
        return rpm

    def _velocityToRPM(self, desired_velocity):
        # Placeholder function to convert desired velocity to RPMs
        # Implement the actual conversion based on your drone model
        # For simplicity, we'll return hover RPMs
        return np.array([self.HOVER_RPM] * 4)

    def _computeReward(self):
        # Implement the reward function
        reward = 0
        # Get the detection with depth
        detection_with_depth = self.last_obs['detections'][0]
        if detection_with_depth[4] > 0.5:
            # Compute relative position
            relative_pos = self._bboxToRelativePosition(detection_with_depth)
            # Compute the distance to the target
            distance = np.linalg.norm(relative_pos)
            # Reward is negative distance
            reward = -distance
        else:
            # No detection, small penalty
            reward = -1
        return reward

    def _computeTerminated(self):
        # Define termination conditions
        terminated = False
        if self.step_counter / self.CTRL_FREQ > 60:  # 60 seconds episode
            terminated = True
        return terminated

    def _getDroneImage(self, nth_drone):
        # Capture an image from the drone's camera
        camera_link_state = p.getLinkState(self.DRONE_IDS[nth_drone],
                                           self.CAMERA_LINK_INDEX,
                                           computeForwardKinematics=True,
                                           physicsClientId=self.CLIENT)
        camera_pos = camera_link_state[0]
        camera_orient = camera_link_state[1]

        # Compute the view matrix
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_pos,
            distance=0.1,
            yaw=0,
            pitch=-90,
            roll=0,
            upAxisIndex=2
        )

        # Compute the projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.CAM_FOV,
            aspect=self.IMG_WIDTH / self.IMG_HEIGHT,
            nearVal=self.CAM_NEAR,
            farVal=self.CAM_FAR
        )

        # Capture the image
        _, _, rgb_img, _, _ = p.getCameraImage(
            width=self.IMG_WIDTH,
            height=self.IMG_HEIGHT,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.CLIENT
        )

        # Process the image
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape((self.IMG_HEIGHT, self.IMG_WIDTH, 4))[:, :, :3]
        return rgb_array

    def preprocess_image(self, img):
        # Preprocess the image for the YOLO model
        img_resized = cv2.resize(img, (640, 640))
        img_normalized = img_resized / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))  # Channels first
        img_tensor = torch.from_numpy(img_transposed).float().unsqueeze(0)  # Add batch dimension
        return img_tensor

    def _bboxToRelativePosition(self, detection_with_depth):
        bbox = detection_with_depth[:4]
        median_depth = detection_with_depth[6]

        # Compute angles based on pixel coordinates
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        x_angle = (x_center - self.IMG_WIDTH / 2) * (self.CAM_FOV / self.IMG_WIDTH)
        y_angle = (y_center - self.IMG_HEIGHT / 2) * (self.CAM_FOV / self.IMG_HEIGHT)

        estimated_distance = median_depth

        # Compute relative position in the camera frame
        relative_x = estimated_distance * np.tan(np.radians(x_angle))
        relative_y = estimated_distance * np.tan(np.radians(y_angle))
        relative_z = estimated_distance

        return np.array([relative_x, relative_y, relative_z])
