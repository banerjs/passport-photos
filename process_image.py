import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

from PIL import Image
from deepface import DeepFace
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def detect_faces(image, input_path, select_face=False):
    print("Detecting faces in", input_path)
    # Detect faces using DeepFace API
    faces = DeepFace.extract_faces(img_path=input_path, detector_backend='opencv', enforce_detection=True)
    print(f"Found {len(faces)} faces in image. Using face with highest confidence")
    face = None
    max_confidence = -np.inf
    for idx, candidate_face in enumerate(faces):
        crop_image(image, candidate_face, idx)
        if select_face:
            choice = input(f"Select face {idx}? (y/n): ")
            if choice == 'y':
                face = candidate_face
                max_confidence = face['confidence']
            else:
                continue
        elif candidate_face['confidence'] > max_confidence:
            face = candidate_face
            max_confidence = face['confidence']

    assert face is not None, "No faces found!"
    return face

def crop_image(image, face, idx=None):
    # Create a mask for Cropping
    mask = np.zeros(image.shape[:2], np.uint8)

    fx, fy, fw, fh = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
    left_eye, right_eye = face['facial_area']['left_eye'], face['facial_area']['right_eye']

    # Optional padding to include if the face bounding box is significantly cropping out the chin / top of the head
    ypad = int(0.1 * fh)
    xpad = int(0.1 * fw)
    fy = fy - ypad
    fx = fx - xpad
    fh = int(fh + (2 * ypad))
    fw = int(fw + (2 * xpad))

    eye_height = (left_eye[1] + right_eye[1]) // 2
    face_center_x = fx + fw // 2
    resolution = fh / 1.25

    mask_height = int(2 * resolution)  # Total height of the mask is 2 inches
    mask_width = int(2 * resolution)   # Total width of the mask is 2 inches

    # Ideally this value should be ~0.87. But adjust according to the image
    mask_top = max(0, eye_height - int(0.95 * resolution))  # Eyes are 1.25 from bottom
    mask_bottom = min(image.shape[0], mask_top + mask_height)

    mask_left = max(0, face_center_x - mask_width // 2)
    mask_right = min(image.shape[1], mask_left + mask_width)

    mask[mask_top:mask_bottom, mask_left:mask_right] = 1

    # # Crop the image to the mask area
    image = image[mask_top:mask_bottom, mask_left:mask_right]
    plt.imsave(f"face{idx if idx is not None else ''}.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGBA))

    print("Image cropped to face")
    return image

def remove_background(image, resegment=False):
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    sam.to(device="cpu")

    used_save = os.path.exists("segmentation.pkl")
    if used_save and not resegment:
        with open("segmentation.pkl", "rb") as fd:
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

    if not used_save or resegment:
        with open("segmentation.pkl", "wb") as fd:
            pickle.dump(max_mask, fd)

    # Set the background to white
    if max_mask['bbox'][0] != 0 and max_mask['bbox'][1] != 0:
        segmentation = 1 - max_mask['segmentation']
    else:
        segmentation = max_mask['segmentation']

    image = image * (1 - segmentation[:, :, None])
    image = np.where(segmentation[:, :, None], 255, image)

    print("Background removed")
    plt.imsave("background_removed.png", cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGBA))
    return image.astype(np.uint8)

def tile_image(image, x, y):
    tiled_image = np.tile(image, (y, x, 1))
    print(f"Image tiled {x} times in x direction and {y} times in y direction")
    return tiled_image

def resize_image(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    print(f"Image resized to {width}x{height}")
    return resized_image

def process_image(
        input_path: str,
        output_path: str,
        *,
        width: int = 2,
        height: int = 2,
        tiled_width: int = 6,
        tiled_height: int = 4,
        dpi: int = 400,
        select_face: bool = False,
        keep_untiled: bool = True,
        keep_background: bool = False,
        resegment: bool = False,
    ):
    try:
        # Load the image using OpenCV
        img_cv = cv2.imread(input_path)

        # Pipeline
        face = detect_faces(img_cv, input_path, select_face)
        img_cv_cropped = crop_image(img_cv, face)
        if not keep_background:
            img_cv_cropped = remove_background(img_cv_cropped, resegment)

        if not keep_untiled:
            img_cv_tiled = tile_image(img_cv_cropped, tiled_width // width, tiled_height // height)
            img_cv_resized = resize_image(img_cv_tiled, tiled_width * dpi, tiled_height * dpi)
        else:
            img_cv_resized = resize_image(img_cv_cropped, width * dpi, height * dpi)

        # Convert the resized image to RGBA using PIL
        img_pil = Image.fromarray(cv2.cvtColor(img_cv_resized, cv2.COLOR_BGR2RGBA))
        img_pil.save(output_path)
        print(f"Processed image saved as {output_path}")
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--select-face", action="store_true")
    parser.add_argument("--keep-untiled", action="store_true")
    parser.add_argument("--keep-background", action="store_true")
    parser.add_argument("--resegment", action="store_true")
    args = parser.parse_args()

    process_image(
        args.input_path,
        args.output_path,
        select_face=args.select_face,
        keep_untiled=args.keep_untiled,
        keep_background=args.keep_background,
        resegment=args.resegment,
    )
