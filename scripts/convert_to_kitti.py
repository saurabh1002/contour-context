import os
import argparse
import numpy as np
from tqdm import tqdm

from kiss_icp.datasets.generic import GenericDataset

argparser = argparse.ArgumentParser()
argparser.add_argument("path", type=str)
args = argparser.parse_args()

data_dir = os.path.abspath(args.path)

dataset = GenericDataset(data_dir)
scan_files = dataset.scan_files

kitti_dir = os.path.abspath(os.path.join(data_dir, os.pardir, f"{os.path.basename(data_dir)}_kitti"))
os.makedirs(kitti_dir, exist_ok=True)
for i, pcl in tqdm(enumerate(dataset), total=len(dataset)):
    filename = os.path.basename(scan_files[i]).split(".")
    pcl_kitti = np.concatenate((np.asarray(pcl, np.float32), np.ones((pcl.shape[0], 1), np.float32)), axis = 1)
    pcl_kitti.tofile(os.path.join(kitti_dir, f"{filename[0]}.bin"))