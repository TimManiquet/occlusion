# This scripts occludes images

# Steps:
# imports objects, and runs 3 manipulations
# on two levels of difficulty (easy/hard)
# and 2 ways of occluding (many small occluders/
# few large occluders)

import cv2 as cv
import numpy as np
import pathlib
import glob
import os
from numpy.random import randint

## GLOBAL VARIABLES
# Parameters for many small
MS_SIZE_OCCLUDER = [10, 40]
MS_NUM_OCCLUDERS_LOW = 15
MS_NUM_OCCLUDERS_HIGH = 50

# Parameters for few large
FL_SIZE_OCCLUDER = [70, 300]
FL_NUM_OCCLUDERS_LOW = 5
FL_NUM_OCCLUDERS_HIGH = 5

def blobs(
        img_dir,
        easy=20,
        hard=60,
        many_small=False,
        few_large=False,
        col=0,
        out_root="./outputs",
        seed=42,
    ):
    
    occlude(
        img_dir,
        easy=easy,
        hard=hard,
        apply_blobs=True,
        apply_deletion=False,
        apply_partialviewing=False,
        many_small=many_small,
        few_large=few_large,
        col=col,
        out_root=out_root,
        seed=seed,
    )

    return

def deletion(
        img_dir,
        easy=20,
        hard=60,
        many_small=False,
        few_large=False,
        col=0,
        out_root="./outputs",
        seed=42,
    ):
    
    occlude(
        img_dir,
        easy=easy,
        hard=hard,
        apply_blobs=False,
        apply_deletion=True,
        apply_partialviewing=False,
        many_small=many_small,
        few_large=few_large,
        col=col,
        out_root=out_root,
        seed=seed,
    )

    return

def partial_viewing(
        img_dir,
        easy=20,
        hard=60,
        many_small=False,
        few_large=False,
        col=0,
        out_root="./outputs",
        seed=42,
    ):
    
    occlude(
        img_dir,
        easy=easy,
        hard=hard,
        apply_blobs=False,
        apply_deletion=False,
        apply_partialviewing=True,
        many_small=many_small,
        few_large=few_large,
        col=col,
        out_root=out_root,
        seed=seed,
    )

    return


def get_occluded_ratio(img, occluder_mask):
    """
    Calculate the proportion of an object in an image that is occluded by an occluder mask.

    Args:
        img (numpy.ndarray): The input image containing the object to be occluded.
        occluder_mask (numpy.ndarray): The mask used to occlude the object.

    Returns:
        float: The proportion of the object in the input image that is occluded by the occluder mask, as a percentage.
    """
    # creating object masks & measuring pixel size
    object_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    # creating a white mask where the object is located
    object_mask[img[:, :, 3] == 255] = 255

    # measuring the pixel size of the object
    size_object = object_mask[object_mask == 255].shape[0]

    # calculate the intersection between masks
    intersection = cv.bitwise_and(object_mask, occluder_mask)

    # create the intersection
    intersection_size = intersection[intersection == 255].shape[0]

    # measure the size of intersection
    proportion_occluded = (intersection_size / size_object) * 100

    return proportion_occluded


def get_manipulation_coordinates(img, n_occluders, size_occluder):
    """
    Randomly generate the coordinates and radii for occluders to be applied to an image.

    Args:
        img (numpy.ndarray): The input image to which occluders will be applied.
        n_occluders (int): The number of occluders to be applied.
        size_occluder (Tuple[int, int]): The range of sizes for the occluders, given as a tuple (min, max).

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing two numpy arrays: the (x, y) coordinates of the occluders and their corresponding radii.
    """
    # create the coordinates for occluders
    points_1 = np.array(randint(0, img.shape[0], n_occluders))
    points_2 = np.array(randint(0, img.shape[1], n_occluders))
    points = np.column_stack((points_1, points_2))

    # randomly generate the manipulation radiuses within the determined range
    radii = randint(size_occluder[0], size_occluder[1], n_occluders)

    return points, radii


def deletion_(img, points, radii, col):
    """
    Apply occlusion by deleting part of the image.

    Args:
        img (numpy.ndarray): The image to be occluded.
        points (numpy.ndarray): The coordinates of the centers of the occluders. It should be a 2D array of shape (n_occluders, 2).
        radii (numpy.ndarray): The radii of the occluders. It should be a 1D array of length n_occluders.
        col (tuple): The RGB color of the occluder. It should be a tuple of length 3.

    Returns:
        tuple: A tuple of two elements. The first element is the occluded image, and the second element is the occluder mask.

    """
    # preparing the background for occluder mask
    # create the black figure that will hold the occluder mask
    occluder_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    for point, radius in zip(points, radii):
        occluded = cv.circle(img, point, radius, int(col), -1)
        # here I draw the occluder mask
        occluder_mask = cv.circle(
            occluder_mask, point, radius, color=(255, 255, 255), thickness=-1
        )

    return occluded, occluder_mask


def blobs_(img, points, radii, col):
    """
    Draw circular occluders on an input image and generate a corresponding
    occluder mask.

    Args:
        img (numpy.ndarray): The input image on which to draw occluders.
        points (list of tuples): A list of (x, y) tuples specifying the center points of each occluder.
        radii (list of floats): A list of radii for each occluder.
        col (tuple of ints): A tuple specifying the color of the occluders in (B, G, R) format.

    Returns:
        tuple of numpy.ndarrays: A tuple containing two NumPy arrays: the input image with occluders drawn on it, and a corresponding binary occluder mask.

    """
    # preparing the background for occluder mask
    # create the black figure that will hold the occluder mask
    occluder_mask = np.zeros_like(img)

    # drawing occluders on the image & on their mask
    for point, radius in zip(points, radii):
        occluded = cv.circle(img, point, radius, int(col), -1)
        # here I draw the occluder mask
        occluder_mask = cv.circle(
            occluder_mask, point, radius, color=(255, 255, 255), thickness=-1
        )

    return occluded, occluder_mask


def partial_viewing_(img, points, radii, col):
    """
    Simulate partially occluded viewing of an input image.

    Args:
        img (numpy.ndarray): The input image to occlude.
        points (list of tuples): A list of (x, y) tuples specifying the center points of each occluder.
        radii (list of floats): A list of radii for each occluder.
        col (tuple of ints): A tuple specifying the color to use for the occluded regions in (B, G, R) format.

    Returns:
        tuple of numpy.ndarrays: A tuple containing two NumPy arrays: the input image partially occluded according to the specified points and radii, and a corresponding binary occluder mask.

    """
    # preparing the background for occluder mask
    # create the black figure that will hold the occluder mask
    occluder = np.zeros_like(img)

    for point, radius in zip(points, radii):
        cv.circle(occluder, point, radius, (255, 255, 255), -1)

    occluded = cv.bitwise_and(img, img, mask=occluder)

    # turn the occluder into the desired color
    occluded[occluder == 0] = col

    # create the occluder mask
    # occluder_inv = cv.bitwise_not(occluder)
    blank = np.ones_like(img)
    blank[:] = 255
    occluder_mask = cv.bitwise_and(blank, blank, mask=occluder)
    return occluded, occluder_mask


def apply_manipulation(
    img_paths,
    out_root,
    size_occluders="manysmall",
    level_occluder="high",
    occlusion_level=60,
    manip_func=None,
    col=0,
    seed=42,
):
    """
    Apply an occlusion manipulation to a set of images and save the results.

    Args:
        img_paths (list): A list of file paths to the images to be manipulated.
        out_root (str): The root directory where the manipulated images will be saved.
        size_occluders (str, optional): The size of the occluders to be used. Either "manysmall" or "fewlarge". Defaults to "manysmall".
        level_occluder (str, optional): The level of occluder density to be used. Either "low", "high", or "control". Defaults to "high".
        occlusion_level (float, optional): The desired percentage of occlusion in the manipulated image. Defaults to 60.
        manip_func (function, optional): The manipulation function to be applied to the images. Defaults to None.
        col (tuple, optional): The grayscale color of the occluder. Defaults to 0 (black).
        seed (int, optional): The random seed to be used for selecting occluder positions. Defaults to 42.

    Returns:
        None

    """
    # Set random seed
    np.random.seed(seed)

    # Retrieve manipulation name (and remove underscore)
    manip = manip_func.__name__
    manip = manip[:-1] if manip[-1] == '_' else manip

    # create the write path
    out_dir = os.path.join(out_root, manip, size_occluders, level_occluder)
    os.makedirs(out_dir, exist_ok=True)

    # Determine occluder parameters based on inputs
    occluder_size, n_occluders_low, n_occluders_high = (
        (MS_SIZE_OCCLUDER, MS_NUM_OCCLUDERS_LOW, MS_NUM_OCCLUDERS_HIGH)
        if size_occluders == "manysmall"
        else (FL_SIZE_OCCLUDER, FL_NUM_OCCLUDERS_LOW, FL_NUM_OCCLUDERS_HIGH)
    )

    n_occluders = (
        n_occluders_low if level_occluder == "low" else n_occluders_high
    )

    # Apply manipulation to all image in img_paths
    for img_path in img_paths:
        # get the file
        file = pathlib.Path(img_path)

        # read the image
        im = cv.imread(str(file), -1)

        # Set the (temp) occluded proportion
        proportion_occluded = 0

        # as long as the difference between the required occlusion proportion and the actual occlusion proportion is more than 1%
        while abs(occlusion_level - proportion_occluded) > 1:
            # print(occlusion_level, proportion_occluded)
            # make a 2D copy of the image
            gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

            # Get manipulations coordinates
            points, radii = get_manipulation_coordinates(
                gray, n_occluders, occluder_size
            )
            
            if len(radii) < 1:
                n_occluders += 1
                continue
            
            # drawing occluders on the image & on their mask
            occluded, occluder_mask = manip_func(gray, points, radii, col)

            # Get occluded proportion
            proportion_occluded = get_occluded_ratio(im, occluder_mask)

            # if the actual occlusion is less than 1% away from goal, write the obtained image
            if abs(occlusion_level - proportion_occluded) < 1:
                fname = os.path.join(
                    out_dir,
                    f"{size_occluders}_{manip}_{level_occluder}_{file.name}",
                )
                if level_occluder == "control":
                    mask = (cv.imread(str(file), -1))[:, :, 3] == 255
                    occluded[mask] = im[mask][:, 2]
                cv.imwrite(fname, occluded)
                # print('Object occluded successfully')
                break

            # if occlusion is too low, increase the number of occluders
            elif occlusion_level > proportion_occluded:
                n_occluders += 1
                # print('Occlusion too low: {} instead of {}.'.format(proportion_occluded, occlusion_level))
                continue

            # if occlusion is too high, decrease the number of occluders
            elif occlusion_level < proportion_occluded:
                n_occluders -= 1
                # print('Occlusion too high: {} instead of {}.'.format(proportion_occluded, occlusion_level))
                continue
    return


def occlude(
    img_dir,
    easy=20,
    hard=60,
    apply_blobs=False,
    apply_deletion=False,
    apply_partialviewing=False,
    many_small=False,
    few_large=False,
    col=0,
    out_root="./outputs",
    seed=42,
):
    """
    Apply occlusion to a set of images by adding blobs, occluding or deleting part of the image.

    Args:
        img_dir (str): The path to the directory containing the images to occlude.
        easy (int, optional): The percentage of the object *occluded* in the low level of occlusion. Defaults to 20.
        hard (int, optional): The percentage of the object *occluded* in the high level of occlusion. Defaults to 60.
        apply_blobs (bool, optional): If True, apply occlusion by adding blobs. Defaults to True.
        apply_deletion (bool, optional): If True, apply occlusion by deleting part of the image. Defaults to True.
        apply_partialviewing (bool, optional): If True, apply occlusion by partially viewing the image. Defaults to True.
        many_small (bool, optional): If True, apply many small occluders. Defaults to True.
        few_large (bool, optional): If True, apply few large occluders. Defaults to True.
        col (int, optional): The grayscale color of the occluding window. Defaults to 0 (black).
        out_root (str, optional): The path to the output directory where occluded images will be saved. Defaults to "./outputs".
        seed (int, optional): The random seed to use. Defaults to 42.

    Returns:
        None: Saves the occluded images in the output directory.
    """
    if not (apply_blobs or apply_deletion or apply_partialviewing):
        raise ValueError("At least one of 'apply_blobs', 'apply_deletion', or 'apply_partialviewing' must be True.")
    
    if not (many_small or few_large):
        raise ValueError("At least one of 'many_small' or 'few_large' must be True.")

    if not easy < hard:
        raise ValueError(f"The percentage of object occluded in the easy condition must be lower than the hard condition {(easy, hard)}.")

    # Determine the occlusion sizes to use based on the function arguments
    occl_sizes = [
        occl_size
        for occl_size in ["manysmall", "fewlarge"]
        if (many_small and occl_size == "manysmall")
        or (few_large and occl_size == "fewlarge")
    ]

    # Get a list of all image file paths in the specified directory
    img_paths = glob.glob(os.path.join(img_dir, "*"))
    img_paths = [img_path for img_path in img_paths if (".png" in img_path) or (".bmp" in img_path) or (".jpg" in img_path)]
    img_paths.sort()
    
    if len(img_paths) < 1:
        raise ValueError(f"No supported iamge (png, bmp, jpg) found in {img_dir}")
        
    # Determine the background color for the deletion function
    color_bg = tuple((cv.imread(img_paths[0], -1))[0, 0, 0:3])

    # Create a dictionary of manipulation functions and their parameters
    manipulation_params = {
        "deletion": {
            "levels": ["low", "high"],
            "color": color_bg[0],
            "func": deletion_,
        }
        if apply_deletion
        else None,
        "partialviewing": {
            "levels": ["low", "high", "control"],
            "color": col,
            "func": partial_viewing_,
        }
        if apply_partialviewing
        else None,
        "blobs": {
            "levels": ["low", "high", "control"],
            "color": col,
            "func": blobs_,
        }
        if apply_blobs
        else None,
    }

    # Iterate over each manipulation with its corresponding parameters
    for manipulation, params in manipulation_params.items():
        if params is None:
            continue
        # Iterate over each occlusion size
        for occl_size in occl_sizes:
            # Iterate over each level of occlusion (low, high)
            for level in params["levels"]:
                # Determine the percentage of the object to occlude for the given level of occlusion
                occlusion_level = easy if level == "low" else hard
                # Apply the manipulation function to each image and save the resulting image
                print(
                    f"STEP: Manipulation: {manipulation}, occlusion size: {occl_size}, level: {level} ...",
                    end="",
                )
                # try:
                #     apply_manipulation(
                #         img_paths,
                #         out_root,
                #         occl_size,
                #         level,
                #         occlusion_level,
                #         params["func"],
                #         params["color"],
                #         seed=seed,
                #     )
                #     print("DONE!")
                # except Exception as e:
                #     print("ERROR:", e)
                apply_manipulation(
                    img_paths,
                    out_root,
                    occl_size,
                    level,
                    occlusion_level,
                    params["func"],
                    params["color"],
                    seed=seed,
                )
                print("DONE!")
