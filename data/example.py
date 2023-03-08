#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scripts.occlusion_funcs import blobs
from scripts.occlusion_funcs import partial_viewing
from scripts.occlusion_funcs import deletion

input_dir = './input_images'

# If you want to apply only one manipulation type
blobs(input_dir, easy=10, hard=50, many_small=True)
deletion(input_dir, easy=40, hard=70, few_large=True, col=255)
partial_viewing(input_dir, easy=20, hard=50, many_small=False, few_large=True, seed=42)

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
