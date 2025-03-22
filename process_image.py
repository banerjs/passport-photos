import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

from PIL import Image
from deepface import DeepFace
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def detect_faces(input_path):
    # Detect faces using DeepFace API
    faces = DeepFace.extract_faces(img_path=input_path, detector_backend='opencv', enforce_detection=True)
    assert len(faces) == 1
    face = faces[0]
    return face

def crop_image(image, face):
    # Create a mask for Cropping
    mask = np.zeros(image.shape[:2], np.uint8)

    fx, fy, fw, fh = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
    left_eye, right_eye = face['facial_area']['left_eye'], face['facial_area']['right_eye']

    eye_height = (left_eye[1] + right_eye[1]) // 2
    face_center_x = fx + fw // 2
    resolution = fh / 1.25

    mask_height = int(2 * resolution)  # Total height of the mask is 2 inches
    mask_width = int(2 * resolution)   # Total width of the mask is 2 inches

    # Ideally this value should be ~0.87. But adjust according to the image
    mask_top = max(0, eye_height - int(1.05 * resolution))  # Eyes are 1.25 from bottom
    mask_bottom = min(image.shape[0], mask_top + mask_height)

    mask_left = max(0, face_center_x - mask_width // 2)
    mask_right = min(image.shape[1], mask_left + mask_width)

    mask[mask_top:mask_bottom, mask_left:mask_right] = 1

    # Crop the image to the mask area
    image = image[mask_top:mask_bottom, mask_left:mask_right]

    plt.imsave("debug.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGBA))

    print("Image cropped to face")
    return image

def remove_background(image):
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    sam.to(device="cpu")

    used_save = os.path.exists("save.pkl")

    if used_save:
        with open("save.pkl", "rb") as fd:
            max_mask = pickle.load(fd)
    else:
        generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")
        print("Segmentation model loaded")

        masks = generator.generate(image)

        # Assume that the mask of max area is the background
        max_mask = None
        max_area = -np.inf
        for idx, mask in enumerate(masks):
            # print(idx, mask['bbox'], mask['area'])
            # plt.imsave(f"debug{idx}.png", mask['segmentation'])
            if mask['area'] > max_area:
                max_area = mask['area']
                max_mask = mask

    if not used_save:
        with open("save.pkl", "wb") as fd:
            pickle.dump(max_mask, fd)

    # Set the background to white
    if max_mask['bbox'][0] != 0 and max_mask['bbox'][1] != 0:
        segmentation = 1 - max_mask['segmentation']
    else:
        segmentation = max_mask['segmentation']

    image = image * (1 - segmentation[:, :, None])
    image = np.where(segmentation[:, :, None], 255, image)

    print("Background removed")
    return image.astype(np.uint8)

def tile_image(image, x, y):
    tiled_image = np.tile(image, (y, x, 1))
    print(f"Image tiled {x} times in x direction and {y} times in y direction")
    return tiled_image

def resize_image(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    print(f"Image resized to {width}x{height}")
    return resized_image

def process_image(input_path, output_path, width=6, height=4, dpi=400):
    size = 2 # inches

    try:
        # Load the image using OpenCV
        img_cv = cv2.imread(input_path)

        # Pipeline
        face = detect_faces(input_path)
        img_cv_cropped = crop_image(img_cv, face)
        img_cv_cropped = remove_background(img_cv_cropped)
        img_cv_tiled = tile_image(img_cv_cropped, width // size, height // size)
        img_cv_resized = resize_image(img_cv_tiled, width * dpi, height * dpi)

        # Convert the resized image to RGBA using PIL
        img_pil = Image.fromarray(cv2.cvtColor(img_cv_resized, cv2.COLOR_BGR2RGBA))
        img_pil.save(output_path)
        print(f"Processed image saved as {output_path}")
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_image.py <input_path> <output_path>")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        process_image(input_path, output_path)
