from PIL import Image
import cv2
import numpy as np
import sys
from deepface import DeepFace

import segmentation_models as sm

import matplotlib.pyplot as plt

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

    mask_top = max(0, eye_height - int(0.87 * resolution))  # Eyes are 1.25 from bottom
    mask_bottom = min(image.shape[0], mask_top + mask_height)

    mask_left = max(0, face_center_x - mask_width // 2)
    mask_right = min(image.shape[1], mask_left + mask_width)

    mask[mask_top:mask_bottom, mask_left:mask_right] = 1

    # Crop the image to the mask area
    image = image[mask_top:mask_bottom, mask_left:mask_right]
    return image

def apply_semantic_segmentation(image):
    # Load pre-trained MobileNetV2 model + higher level layers
    model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False)

    # Preprocess the image
    image_resized = cv2.resize(image, (224, 224))
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # Perform semantic segmentation
    predictions = model.predict(image_array)
    predictions = tf.image.resize(predictions, (image.shape[0], image.shape[1]))
    num_classes = predictions.shape[-1]
    predictions = tf.argmax(predictions, axis=-1)
    predictions = predictions[0]

    # Display the segmentation labels
    plt.imshow(predictions/num_classes)
    plt.title("Semantic Segmentation Labels")
    plt.show()

    # Set background to white
    image[predictions == 0] = [255, 255, 255]
    return image

def process_image(input_path, output_path):
    try:
        # Load the image using OpenCV
        img_cv = cv2.imread(input_path)

        # Pipeline
        face = detect_faces(input_path)
        img_cv_cropped = crop_image(img_cv, face)
        apply_semantic_segmentation(img_cv_cropped)

        # Convert the cropped image to RGBA using PIL
        img_pil = Image.fromarray(cv2.cvtColor(img_cv_cropped, cv2.COLOR_BGR2RGBA))
        datas = img_pil.getdata()

        new_data = []
        for item in datas:
            if item[3] == 0:  # If the pixel is transparent
                new_data.append((255, 255, 255, 255))  # Change to white
            else:
                new_data.append(item)  # Keep the original pixel

        img_pil.putdata(new_data)
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
