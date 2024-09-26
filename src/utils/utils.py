import os
import time
import torch
import numpy as np

def makedirs(save_path):
    os.makedirs(save_path, exist_ok=True)


def to_cuda_vars(vars_dict):
    new_dict = {}
    for k, v in vars_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.float().cuda()
    return new_dict


def format_time(time):
    hour, minute, second = time // 3600, (time % 3600) // 60, time % 3600 % 60
    string = str(hour).zfill(2) + ":" + str(minute).zfill(2) + ":" + str(second).zfill(2)
    return string


def get_printability_array(self, printability_file, side):
    printability_list = []
    with open(printability_file) as f:
        for line in f:
            printability_list.append(line.split(","))

    printability_array = []
    for printability_triplet in printability_list:
        printability_imgs = []
        red, green, blue = printability_triplet
        printability_imgs.append(np.full((side, side), red))
        printability_imgs.append(np.full((side, side), green))
        printability_imgs.append(np.full((side, side), blue))
        printability_array.append(printability_imgs)

    printability_array = np.asarray(printability_array)
    printability_array = np.float32(printability_array)
    pa = torch.from_numpy(printability_array)
    return pa

def get_nps_score(patch, file):