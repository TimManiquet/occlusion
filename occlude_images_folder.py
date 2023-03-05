#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 17:47:35 2023

@author: costantino_ai
"""

import argparse
from scripts.occlusion_funcs import occlude

# Example call from command line:
# python occlude_images_folder.py /path/to/input_images --hard 80 --blobs --out_dir /path/to/outputs

def parse_args():
    parser = argparse.ArgumentParser(description='Script to occlude images in a folder.')
    parser.add_argument('img_dir', type=str, help='Path to the input images directory.')
    parser.add_argument('--easy', type=int, default=20, help='Percentage of easy occlusion to apply.')
    parser.add_argument('--hard', type=int, default=60, help='Percentage of hard occlusion to apply.')
    parser.add_argument('--control', action='store_true', help='Whether to use control conditions.')
    parser.add_argument('--blobs', action='store_true', help='Whether to apply blob occlusion.')
    parser.add_argument('--deletion', action='store_true', help='Whether to apply deletion occlusion.')
    parser.add_argument('--partialviewing', action='store_true', help='Whether to apply partial viewing occlusion.')
    parser.add_argument('--many_small', action='store_true', help='Whether to occlude many small areas.')
    parser.add_argument('--few_large', action='store_true', help='Whether to occlude few large areas.')
    parser.add_argument('--col', type=int, default=0, help='Color channel to occlude (0 for grayscale, 1 for red, 2 for green, 3 for blue).')
    parser.add_argument('--out_dir', type=str, default='./outputs', help='Path to the output directory.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    occlude(
        args.img_dir,
        easy=args.easy,
        hard=args.hard,
        control=args.control,
        apply_blobs=args.blobs,
        apply_deletion=args.deletion,
        apply_partialviewing=args.partialviewing,
        many_small=args.many_small,
        few_large=args.few_large,
        col=args.col,
        out_root=args.out_dir,
        seed=args.seed,
    )
