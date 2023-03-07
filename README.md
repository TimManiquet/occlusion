# Occlusion of Images
This repository contains three Python scripts to apply occlusion to a set of images in different ways. The three ways of occluding images are: by adding **blobs** on top of images, by adding a **partial viewing** mask with apertures on top of the images, or by **deleting** parts of the images. Each of these manipulations, occludes objects in the images such that only a _certain proportion_ of them will remain visible.

## Input images
To ensure that the occluded  proportion is calculated correctly, make sure the input images contain an transparent mask (alpha channel) where the object is delineated from the background. These images should be 4-channel, grayscale images. Example of such images can be found in the `input images` folder.

## Requirements
- Python 3
- OpenCV (cv2)
- NumPy
- pathlib
- glob
- os

## Usage
1. Clone or download this repository.
2. Place the images to occlude in the `img_dir` folder. 
3. Run the following in your command line:

`python occlude_images_folder.py /path/to/input_images --hard 80 --apply_blobs --many_small --out_dir /path/to/outputs`

4. The resulting occluded images will be saved in the `out_dir` folder.

## Arguments

- `img_dir` (required): Path to the input images directory.
- `--easy` (optional, default 20): Percentage of easy occlusion to apply.
- `--hard` (optional, default 60): Percentage of hard occlusion to apply.
- `--apply_blobs` (optional): Whether to apply blob occlusion.
- `--apply_deletion` (optional): Whether to apply deletion occlusion.
- `--apply_partialviewing` (optional): Whether to apply partial viewing occlusion.
- `--many_small` (optional): Whether to occlude many small areas.
- `--few_large` (optional): Whether to occlude few large areas.
- `--col` (optional, default 0): The grayscale color of the occluding window. Defaults to 0 (black).
- `--out_dir` (optional, default ./outputs): Path to the output directory.
- `--seed` (optional, default 42): Seed for the random number generator.

## Validation

The script includes validation to ensure that at least one of 'blobs', 'deletion', or 'partialviewing' is set to `True`, and that at least one of 'many_small' or 'few_large' is set to `True`. Additionally, the percentage of object occluded in the easy condition must be lower than the hard condition.

If any of these validations fail, the script will raise a `ValueError`.

## Example
The program can also be run as a python script as shown below

```
from scripts.occlusion_funcs import blobs
from scripts.occlusion_funcs import partial_viewing
from scripts.occlusion_funcs import deletion

# If you want to apply only one manipulation type
blobs(input_dir, easy=10, hard=50, many_small=True, col=255)

deletion(input_dir, easy=40, hard=70, few_large=True, col=255)

partial_viewing(input_dir, easy=20, hard=50, many_small=False, few_large=True, seed=10, out_dir='/outfolder')

# We can also use the more general occlude function that is called in occlude_images_folder.py
from scripts.occlusion_funcs import occlude

occlude(
    input_dir,
    easy=20,
    hard=60,
    apply_blobs=True,
    apply_deletion=True,
    apply_partialviewing=True,
    many_small=True,
    few_large=True,
    col=0,
    out_root="./outputs",
    seed=42,
)

```

This example applies different level and types of occlusion to the images in the `input_dir` folder. The resulting occluded images will be saved in the `output_images` folder. An example output object from this command is found below:

<img
   src="https://github.com/TimManiquet/occlusion/blob/main/outputs/blobs/fewlarge/control/fewlarge_blobs_control_banana.png" 
   alt="Alt text" 
   title="Few large blob occluders"
   style="display: inline-block; margin: 0 auto; max-width: 300px">


## Installation

To install this package using pip, first ensure that you have Git installed on your system. Then, create a new virtual environment and activate it. Finally, run the following command:

```pip install git+https://github.com/TimManiquet/occlusion.git@v0.1```
