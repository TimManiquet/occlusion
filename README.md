# Occlusion of Images
This repository contains three Python scripts to apply occlusion to a set of images in different ways. The three ways of occluding images are: by adding *blobs* on top of images, by adding a *partial viewing* mask with apertures on top of the images, or by *deleting* parts of the images. Each of these manipulations, occludes objects in the images such that only a _certain proportion_ of them will remain visible.

## Input images
To ensure that the occluded  proportion is calculated correctly, make sure the input images contain an transparent mask (alpha channel) where the object is delineated from the background. These images should be 4-channel, grayscale images. Example of such images can be found in the `input images` folder.

## Requirements
- Python 3
- OpenCV (cv2)
- NumPy
- pathlib
- glob
- tqdm

## Usage
1. Clone or download this repository.
2. Place the images to occlude in the input_images folder. 
3. Run the `occlude` function in the script, specifying the path to the `input_images` folder and the desired parameters for occlusion:

- `easy`: The percentage of the object occluded in the low level of occlusion. Defaults to 20.
- `hard`: The percentage of the object occluded in the high level of occlusion. Defaults to 60.
- `many_small`: If True, apply many small occluders, otherwise apply few large occluders. Defaults to True.
- `few_large`: If True, apply few large occluders, otherwise apply many small occluders. Defaults to True.
- `col`: The grayscale color of the occluding window. Defaults to 0 (black).
- `control`: If True, show the original and occluded images side by side for comparison. Defaults to False.
- `blobs`: If True, add random blobs as occluders. Defaults to True.
- `deletion`: If True, occlude part of the image by deleting pixels. Defaults to True.
- `partialviewing`: If True, occlude part of the image by setting pixels to the occluding color. Defaults to True.

4. The resulting occluded images will be saved in the `output` folder.


## Example

```
from scripts.blobs import blobs
from scripts.partial_viewing import partial_viewing
from scripts.deletion import deletion

blobs("input_images", easy=10, hard=50, many_small=False, col=255)

deletion("input_images", easy=40, hard=70, many_small=False, col=255)

partial_viewing("input_images", easy=20, hard=50, many_small=False, col=255)
```

This example applies occlusion to the images in the `input_images` folder, using a low occlusion level of 10%, a high occlusion level of 50%, few large occluders, and a white occluding window. The resulting occluded images will be saved in the `output_images` folder.
