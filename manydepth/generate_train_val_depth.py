# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


import os

import argparse
from glob import glob
from tqdm import tqdm
import os.path as osp
import numpy as np
import PIL.Image as pil

from utils import readlines
from kitti_utils import generate_depth_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        )
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    txt_set_path = glob(osp.join(split_folder, '*.txt'))
    for txt_path in txt_set_path:
        set_name = os.path.basename(txt_path).split('.')[0]

        lines = readlines(txt_path)

        print("Exporting ground truth depths for {}".format(opt.split))

        gt_depths = []

        def f(i):
            line = lines[i]

            folder, frame_id, cam_str = line.split()
            cam = 2 if cam_str == 'l' else 3
            frame_id = int(frame_id)

            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, cam, True)

            output_path = os.path.join(
                split_folder, set_name, folder, f'depth_cam_0{cam}', f'{frame_id:010d}.npz')
            os.makedirs(osp.dirname(output_path), exist_ok=True)

            np.savez_compressed(output_path, gt_depth)

        from multiprocessing import Pool
        from tqdm import tqdm
        with Pool() as p:
            r = list(tqdm(p.imap(f, range(len(lines))), total=len(lines)))
