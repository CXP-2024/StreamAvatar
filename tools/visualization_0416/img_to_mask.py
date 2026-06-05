"""
input: image_path
output: save a masked image and resized image
"""
import os
import sys
import urllib.request
import numpy as np
import torch
import cv2
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
from utils.face_detector import FaceDetector
from pathlib import Path

def generate_crop_bounding_box(h, w, center, size=512):
    """
    Crop a region of a specified size from the given center point,
    filling the area outside the image boundary with zeros.

    :param image: The input image in NumPy array form, shape (H, W, C)
    :param center: The center point (y, x) to start cropping from
    :param size: The size of the cropped region (default is 512)
    :return: The cropped region with padding, shape (size, size, C)
    """
    half_size = size // 2  # Half the size for the cropping region

    # Calculate the top-left and bottom-right coordinates of the cropping region
    y1 = max(center[0] - half_size, 0)  # Ensure the y1 index is not less than 0
    x1 = max(center[1] - half_size, 0)  # Ensure the x1 index is not less than 0
    y2 = min(center[0] + half_size, h)  # Ensure the y2 index does not exceed the image height
    x2 = min(center[1] + half_size, w)  # Ensure the x2 index does not exceed the image width
    return [x1, y1, x2, y2]

def crop_from_bbox(image, center, bbox, size=512):
    """
    Crop a region of a specified size from the given center point,
    filling the area outside the image boundary with zeros.

    :param image: The input image in NumPy array form, shape (H, W, C)
    :param center: The center point (y, x) to start cropping from
    :param size: The size of the cropped region (default is 512)
    :return: The cropped region with padding, shape (size, size, C)
    """
    h, w = image.shape[:2]  # Get the height and width of the image
    x1, y1, x2, y2 = bbox
    half_size = size // 2  # Half the size for the cropping region
    # Create a zero-filled array for padding
    cropped = np.zeros((size, size, image.shape[2]), dtype=image.dtype)

    # Copy the valid region from the original image to the cropped region
    cropped[(y1 - (center[0] - half_size)):(y2 - (center[0] - half_size)),
            (x1 - (center[1] - half_size)):(x2 - (center[1] - half_size))] = image[y1:y2, x1:x2]

    return cropped

face_detector = None
model_path = "./utils/face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

def initialize_face_detector():
    global face_detector
    if face_detector is None:
        face_detector = FaceDetector(
            mediapipe_model_asset_path=model_path,
            face_detection_confidence=0.5,
            num_faces=1,
        )
initialize_face_detector()

def augmentation(images, transform, state=None):
    if state is not None:
        torch.set_rng_state(state)
    if isinstance(images, list):
        transformed = [transforms.functional.to_tensor(img) for img in images]
        return transform(torch.stack(transformed, dim=0))
    return transform(transforms.functional.to_tensor(images))

def scale_bbox(bbox, h, w, scale=1.8):
    sw = (bbox[2] - bbox[0]) / 2
    sh = (bbox[3] - bbox[1]) / 2
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    sw *= scale
    sh *= scale
    scaled = [cx - sw, cy - sh, cx + sw, cy + sh]
    scaled[0] = np.clip(scaled[0], 0, w)
    scaled[2] = np.clip(scaled[2], 0, w)
    scaled[1] = np.clip(scaled[1], 0, h)
    scaled[3] = np.clip(scaled[3], 0, h)
    return scaled

def get_mask(bbox, hd, wd, scale=1.0, return_pil=True):
    if min(bbox) < 0:
        raise Exception("Invalid mask")
    bbox = scale_bbox(bbox, hd, wd, scale=scale)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    mask = np.zeros((hd, wd, 3), dtype=np.uint8)
    mask[y0:y1, x0:x1, :] = 255
    if return_pil:
        return Image.fromarray(mask)
    return mask

def generate_masked_image(
        image_path="./test_case/test_img.png",
        save_path="./test_case/test_img.png",
        crop=False,
        union_bbox_scale=1.3):
    cfg = OmegaConf.load("./configs/audio_head_animator.yaml")
    pixel_transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize([0.5], [0.5]),
    ])
    resize_transform = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)

    img = Image.open(image_path).convert("RGB")
    state = torch.get_rng_state()

    # Get face detection results first
    det_res = face_detector.get_face_xy_rotation_and_keypoints(
        np.array(img), cfg.data.mouth_bbox_scale, cfg.data.eye_bbox_scale
    )

    person_id = 0
    mouth_bbox = np.array(det_res[6][person_id])
    eye_bbox = det_res[7][person_id]
    face_contour = np.array(det_res[8][person_id])
    left_eye_bbox = eye_bbox["left_eye"]
    right_eye_bbox = eye_bbox["right_eye"]

    # If crop is True, crop the face region first
    if crop:
        # Get the face bounding box and calculate center
        face_bbox = det_res[5][person_id]  # Get the face bounding box from det_res[5]
        # face_bbox is [(x1, y1), (x2, y2)]
        x1, y1 = face_bbox[0]
        x2, y2 = face_bbox[1]
        center = [(y1 + y2) // 2, (x1 + x2) // 2]

        # Calculate the size for cropping
        width = x2 - x1
        height = y2 - y1
        max_size = int(max(width, height) * union_bbox_scale)

        # Get the image dimensions
        hd, wd = img.size[1], img.size[0]

        # Generate the crop bounding box
        crop_bbox = generate_crop_bounding_box(hd, wd, center, max_size)

        # Crop the image
        img_array = np.array(img)
        cropped_img = crop_from_bbox(img_array, center, crop_bbox, size=max_size)
        img = Image.fromarray(cropped_img)

        # Update the face detection results for the cropped image
        det_res = face_detector.get_face_xy_rotation_and_keypoints(
            cropped_img, cfg.data.mouth_bbox_scale, cfg.data.eye_bbox_scale
        )
        mouth_bbox = np.array(det_res[6][person_id])
        eye_bbox = det_res[7][person_id]
        face_contour = np.array(det_res[8][person_id])
        left_eye_bbox = eye_bbox["left_eye"]
        right_eye_bbox = eye_bbox["right_eye"]

    pixel_values_ref = augmentation([img], pixel_transform, state)
    pixel_values_ref = (pixel_values_ref + 1) / 2
    new_hd, new_wd = img.size[1], img.size[0]

    mouth_mask = resize_transform(get_mask(mouth_bbox, new_hd, new_wd, scale=1.0))
    left_eye_mask = resize_transform(get_mask(left_eye_bbox, new_hd, new_wd, scale=1.0))
    right_eye_mask = resize_transform(get_mask(right_eye_bbox, new_hd, new_wd, scale=1.0))
    face_contour = resize_transform(Image.fromarray(face_contour))

    eye_mask = np.bitwise_or(np.array(left_eye_mask), np.array(right_eye_mask))
    combined_mask = np.bitwise_or(eye_mask, np.array(mouth_mask))

    combined_mask_tensor = torch.from_numpy(combined_mask / 255.0).permute(2, 0, 1).unsqueeze(0)
    face_contour_tensor = torch.from_numpy(np.array(face_contour) / 255.0).permute(2, 0, 1).unsqueeze(0)

    masked_ref = pixel_values_ref * combined_mask_tensor + face_contour_tensor * (1 - combined_mask_tensor)
    masked_ref = masked_ref.clamp(0, 1)
    masked_ref_np = (masked_ref.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    base, _ = os.path.splitext(save_path)
    resized_img = (pixel_values_ref.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(resized_img).save(f"{base}_resize.png")
    Image.fromarray(masked_ref_np).save(f"{base}_masked.png")

if __name__ == '__main__':
    import fire
    fire.Fire(generate_masked_image)
    # python img_to_mask.py --image_path /mnt/weka/haiyang_workspace/ckpts/good_train_case/image_example/KristiNoem2-Scene-001.png --save_path /mnt/weka/haiyang_workspace/ckpts/good_train_case/image_example/KristiNoem2-Scene-001.png --crop True --union_bbox_scale 1.6
    # python img_to_latent.py --mask_image_path ./test_case/ChrisVanHollen0-Scene-003_masked.png --save_npz_path ./test_case/ChrisVanHollen0-Scene-003_resize.npz
    # python latent_two_video.py --npz_path ./test_case/ChrisVanHollen0-Scene-003_resize.npz --save_dir ./test_case/
