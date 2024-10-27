import torch
import numpy as np
import cv2

class CVProcessor:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/weights/best.pt')

    def process_image(self, current_frame):
        height, width, _ = current_frame.shape
        masked_image = cv2.bitwise_and(current_frame, current_frame)
        results = self.model(masked_image)
        img = np.copy(results.render()[0])

        recon_centroid_x, recon_centroid_y, highest_conf, best_bbox = self.get_centroid(results, height, width)

        self.draw_annotations(img, height, width, recon_centroid_x, recon_centroid_y)

        return img, highest_conf, best_bbox
        
    def get_centroid(self, results, height, width):
        highest_conf = 0
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

    def draw_annotations(self, img, height, width, recon_centroid_x, recon_centroid_y):
        cv2.circle(img, (recon_centroid_x, recon_centroid_y), 5, (0, 0, 255), -1)
        cv2.line(img, (int(width / 2), int(height / 2)), (recon_centroid_x, recon_centroid_y), (0, 255, 0), 2)
        cv2.putText(img, 'Shahed Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
