"""Compute depth maps for images in the input folder.
"""
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import torch
import cv2
import numpy as np
from midasModel.midas.model_loader import default_models, load_model
from gym_pybullet_drones.midasModel.cv_processor import CVProcessor



def read_image(image):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    return image

first_execution = True
def process(device, model, model_type, image, input_size, target_size, optimize, use_camera):
    """
    Run the inference and interpolate.

    Args:
        device (torch.device): the torch device used
        model: the model used for inference
        model_type: the type of the model
        image: the image fed into the neural network
        input_size: the size (width, height) of the neural network input (for OpenVINO)
        target_size: the size (width, height) the neural network output is interpolated to
        optimize: optimize the model to half-floats on CUDA?
        use_camera: is the camera used?

    Returns:
        the prediction
    """
    global first_execution

    if "openvino" in model_type:
        if first_execution or not use_camera:
            print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
            first_execution = False

        sample = [np.reshape(image, (1, 3, *input_size))]
        prediction = model(sample)[model.output(0)][0]
        prediction = cv2.resize(prediction, dsize=target_size,
                                interpolation=cv2.INTER_CUBIC)
    else:
        sample = torch.from_numpy(image).to(device).unsqueeze(0)

        if optimize and device == torch.device("cuda"):
            if first_execution:
                print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                      "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                      "  half-floats.")
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        if first_execution or not use_camera:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    return prediction

def show_depth(prediction, depth, bits=1):
    """Show depth map using OpenCV.

    Args:
        depth (array): depth
    """

    bits = 1

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
    text = f'Depth: {depth:.2f}'
    text_x = 50
    text_y = 100
    cv2.putText(out, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Depth Map', out.astype("uint8" if bits == 1 else "uint16"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return

alpha = 0.2
previous_depth = 0.0
def apply_ema_filter(current_depth, alpha=0.2):
    """Apply an exponential moving average filter."""
    global previous_depth
    filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = filtered_depth
    return filtered_depth

def depth_to_distance(depth_value, depth_scale=1.0):
    """Convert depth value to distance."""
    return 1.0 / (depth_value * depth_scale)

def run(image, model_path, model_type="dpt_swin2_tiny_256", optimize=False, height=None,
        square=False, trial=False):
    """Run MonoDepthNN to compute depth maps.

    Args:
        image (str): image
        model_path (str): path to saved model
        model_type (str): the model type
        optimize (bool): optimize the model to half-floats on CUDA?
        height (int): inference encoder image height
        square (bool): resize to a square resolution?
    """
    colour = None
    width = None
    height = None
    global previous_depth
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Model path: {model_path}")
        print(f"Model type: {model_type}")
        raise  

    if image is not None:
        # input
        original_image_rgb = read_image(image=image)  # in [0, 1]
        image_t = transform({"image": original_image_rgb})["image"]

        # compute
        with torch.no_grad():
            prediction = process(device, model, model_type, image_t, (net_w, net_h), original_image_rgb.shape[1::-1],
                                 optimize, False)

        # predict distance
        if prediction is not None:
            height, width = original_image_rgb.shape[1::-1]
            shape = (height, width, 3) if colour else (height, width)
            data = np.reshape(prediction, shape)
            depth_map = np.flipud(data)
            valid_depth_values = depth_map[(depth_map > 0) & np.isfinite(depth_map)]
            if len(valid_depth_values) == 0:
                print("No valid depth values found in the depth map")
            else:
                # Compute the median depth from the valid values
                median_depth = np.median(valid_depth_values)
                smoothed_depth = apply_ema_filter(median_depth)
                distance = depth_to_distance(smoothed_depth)
                if trial:
                    show_depth(prediction, distance, bits=2)
                return median_depth
            


    length = 8.1               
    width = 16
    height = 3.1
if __name__ == "__main__":

    # TODO compute actual distance on bounding box
    # https://medium.com/artificialis/getting-started-with-depth-estimation-using-midas-and-python-d0119bfe1159


    cv = CVProcessor()
    image_path = 'midasModel/input/934.jpg' 
    current_frame = cv2.imread(image_path)
    img, highest_conf, best_bbox = cv.process_image(current_frame=current_frame)

    default_models = {
        'dpt_swin2_tiny_256': 'midasModel/weights/dpt_swin2_tiny_256.pt',
        'dpt_levit_224': 'midasModel/weights/dpt_levit_224.pt'
    }

    # Set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    image = 'midasModel/1.JPEG'
    run(image=current_frame,
        model_path=default_models['dpt_swin2_tiny_256'],
        model_type='dpt_swin2_tiny_256',
        trial=True
)
