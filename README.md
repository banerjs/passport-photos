# Passport Photos Processing

This project processes passport photos by detecting faces, removing backgrounds, and tiling the images. The processed images are saved with a specified resolution.

## Requirements

- Python 3.x
- OpenCV
- Pillow
- DeepFace
- Segment Anything Model (SAM)
- TensorFlow
- Matplotlib

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/banerjs/passport-photos.git
    cd passport-photos
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To process an image, run the following command:

```sh
python process_image.py <input_path> <output_path>
```

### Parameters

- `<input_path>`: Path to the input image file.
- `<output_path>`: Path to save the processed image file.

### Example

```sh
python process_image.py input.jpg output.png
```

## Configuration

The script uses a pre-trained model for semantic segmentation. Ensure that the model checkpoint file `sam_vit_b_01ec64.pth` is available in the project directory.

## License

This project is licensed under the MIT License.
