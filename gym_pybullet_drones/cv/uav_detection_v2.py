import rclpy 
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.node import Node 

from sensor_msgs.msg import Image 
from px4_msgs.msg import TrajectorySetpoint
from std_msgs.msg import Float32

from cv_bridge import CvBridge 
import cv2

import torch
import numpy as np

from simple_pid import PID
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from midasModel.run import run

def process_depth_for_display(prediction, bits=1):
    if not np.isfinite(prediction).all():
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")
    
    depth_min = prediction.min()
    depth_max = prediction.max()
    max_val = (2**(8*bits)) - 1
    
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (prediction - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(prediction.shape, dtype=prediction.dtype)
    
    out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)
    # cv2.putText(out, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    
    return out.astype("uint8" if bits == 1 else "uint16")

def create_combined_image(img, depth_img, target_width=1300, target_height=440):
    # Define the width for each half of the combined image
    half_width = target_width // 2

    # Resize both images to fit half of the target width while maintaining aspect ratio
    img_resized = cv2.resize(img, (half_width, target_height))
    depth_resized = cv2.resize(depth_img, (half_width, target_height))

    # concat horizontally
    combined_img = np.hstack((img_resized, depth_resized))

    return combined_img

class DepthDistance:
    def __init__(self, img, best_bbox = None) -> None:
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
            print(" Centre of image midas depth")
        else:
            # Use the existing bounding box logic
            best_bbox1 = list(map(int, self.best_bbox))
            x_min = max(0, min(best_bbox1[0], width))
            y_min = max(0, min(best_bbox1[1], height))
            x_max = max(0, min(best_bbox1[2], width))
            y_max = max(0, min(best_bbox1[3], height))
            print(" best box midas depth")
        
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
    
class CVProcessor:
    def __init__(self, pid_x, pid_y):
        self.pid_x = pid_x
        self.pid_y = pid_y
        self.vertical_offset_ratio = 0.10
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/weights/best.pt')

    def process_image(self, current_frame, current_yaw):
        median_depth = None

        height, width, _ = current_frame.shape
        mask = self.create_mask(height, width)
        masked_image = cv2.bitwise_and(current_frame, current_frame, mask=mask)
        results = self.model(masked_image)
        img = np.copy(results.render()[0])
 
        recon_centroid_x, recon_centroid_y, highest_conf, best_bbox = self.get_centroid(results, height, width)

        # depth model
        # if best_bbox is not None:
        depth_model = DepthDistance(img=img, best_bbox=best_bbox)
        median_depth, prediction = depth_model.run_model()
            
        self.draw_annotations(img, height, width, recon_centroid_x, 
                              recon_centroid_y, median_depth)
        
        if recon_centroid_x is not None and recon_centroid_y is not None:

            error_x = recon_centroid_x - width / 2
            error_y = recon_centroid_y - height / 2

            velocity_x = self.pid_x(error_x)
            velocity_y = self.pid_y(error_y)

            return img, velocity_x, velocity_y, highest_conf, prediction
        
        return img, None, None, None, prediction

    def create_mask(self, height, width):
        mask = np.ones((height, width), dtype=np.uint8) * 255
        propeller_mask_height = int(height * 0.20)
        propeller_mask_width = int(width * 0.20)
        vertical_offset = int(height * self.vertical_offset_ratio)

        mask[vertical_offset:vertical_offset + propeller_mask_height, -propeller_mask_width:] = 0
        mask[vertical_offset:vertical_offset + propeller_mask_height, :propeller_mask_width] = 0
        return mask

    def get_centroid(self, results, height, width):
        highest_conf = 0.75
        best_centroid_x, best_centroid_y = None, None
        best_bbox = None
        for bbox in results.xyxy[0].cpu().numpy():
            if len(bbox) >= 6:  # Ensure there are enough values to unpack
                x_min, y_min, x_max, y_max, conf, cls = bbox
                if 0 <= x_min < width and 0 <= x_max < width and 0 <= y_min < height and 0 <= y_max < height:
                    if conf > highest_conf:
                        highest_conf = conf
                        best_centroid_x = int((x_min + x_max) / 2)
                        best_centroid_y = int((y_min + y_max) / 2)
                        best_bbox = (x_min, y_min, x_max, y_max)
                else:
                    print(f"Bounding Box Out of Image Bounds: {bbox}") 
        return best_centroid_x, best_centroid_y, highest_conf, best_bbox

    def draw_annotations(self, img, height, width, 
                        recon_centroid_x, recon_centroid_y, median_depth=None):
        
        if recon_centroid_x:
            cv2.circle(img, (recon_centroid_x, recon_centroid_y), 5, (0, 0, 255), -1)
            cv2.line(img, (int(width / 2), int(height / 2)), (recon_centroid_x, recon_centroid_y), (0, 255, 0), 2)
            cv2.putText(img, 'Recon Drone Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)

        if median_depth is not None:
            text = f'Median Depth: {median_depth:.2f}'
        else:
            text = 'Median Depth: Not available'

        cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image,
            'camera',
            self.listener_callback,
            1)

        self.intecpetor_trajectory = self.create_subscription(
            TrajectorySetpoint,
            '/px4_2/fmu/in/trajectory_setpoint',
            self.get_inteceptor_trajectory,
            qos_profile)

        self.inteceptor_velocity = self.create_publisher(
            TrajectorySetpoint,
            'px4_2/fmu/in/trajectory_setpoint',
            qos_profile)

        self.model_confidence = self.create_publisher(
            Float32,
            'model_confidence',
            1)

        self.current_yaw = 0.0
        self.br = CvBridge()

        # Initialise PID controllers
        pid_x = PID(1.0, 0.1, 0.05, setpoint=0)
        pid_y = PID(1.0, 0.1, 0.05, setpoint=0)
        pid_x.output_limits = (-1, 1)
        pid_y.output_limits = (-1, 1)

        self.cv_processor = CVProcessor(pid_x, pid_y)

    def get_inteceptor_trajectory(self, msg):
        self.current_yaw = msg.yaw

    def listener_callback(self, data):
        current_frame = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")
        img, velocity_x, velocity_y, highest_conf, prediction = self.cv_processor.process_image(current_frame, self.current_yaw)

        # Process depth image for imshow
        depth_img = process_depth_for_display(prediction)

        # Create the combined image
        combined_img = create_combined_image(img, depth_img)

        # Display the combined image
        cv2.imshow('Detected Frame', combined_img)
        cv2.waitKey(1)

        if velocity_x is not None and velocity_y is not None:
            twist = TrajectorySetpoint()
            twist.velocity[0] = velocity_x
            twist.velocity[1] = velocity_y
            self.inteceptor_velocity.publish(twist)

        if highest_conf is not None:
            print(f"Highest Confidence: {highest_conf}")
            conf_data = Float32()
            conf_data.data = float(highest_conf)
            self.model_confidence.publish(conf_data)
        else:
            conf_data = Float32()
            conf_data.data = 0.0
            self.model_confidence.publish(conf_data)


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
# TODO: 
#       1.Detect yellow square with cv2 and input the slcied img
#       2.set up pipeline that obtains drones actual distance/ midas depth/ timestamp - lets also save midas images for review 
#       3.setup some sort of analytics for midas and yolo model - i.e. plots showing detection/ inference speed/ precision/ compute used
