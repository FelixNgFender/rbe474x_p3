import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as T
import torch.nn.functional as F
from torch import nn

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

class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """
    
    def __init__(self, printability_file, img_size):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, img_size),
                                               requires_grad=False)
    
    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array.cuda() + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]  # test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)
    
    def get_printability_array(self, printability_file, side):
        printability_list = []
        
        # read in printability triplets and put them in a list
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

def get_printable_colors(file):
    colors = []
    with open(file) as f:
        for line in f:
            # print(line.split(","), type(line.split(",")))
            r, g, b = line.split(",")
            b = b.strip() # Remove the newline character
            colors.append(np.array([r, g, b], dtype=np.float32))

    colors = torch.Tensor(colors)
    return colors

def get_nps_score(patch : torch.Tensor, colors):
    colors = colors.to(patch.device)
    # print(colors.device, patch.device)
    
    # Slow Implementation
    loss = 0
    t = time.time()
    for y, x in zip(range(patch.shape[1]), range(patch.shape[2])):
        loss += min([torch.norm(patch[:, y, x] - color) for color in colors])
    print('Time taken for NPS Loss:', time.time() - t)

    return loss

def get_tv_loss(patch: torch.Tensor):
    # print(patch.shape)
    x = (patch[:, :, :-1] - patch[:, :, 1:]).flatten(-2)
    y = (patch[:, :-1, :] - patch[:, 1:, :]).flatten(-2)
    TV = torch.sum(torch.sqrt(x ** 2 + y ** 2))
    return TV

def get_disp_loss(disp, est_disp, mask, target_disp):
    print(disp.shape, mask.shape, target_disp)
    mask = mask.unsqueeze(0)
    fake_disp = torch.mul((1 - mask), disp) + torch.mul(mask, target_disp)
    loss = torch.nn.functional.l1_loss(torch.mul(mask, fake_disp), torch.mul(mask, est_disp)).mean()
    return loss
    

# def get_nps_score(patch, file):

def visualize_disparity(img, disp):
    img = img.detach().cpu()#.numpy()
    disp = disp.detach()#.squeeze().cpu().numpy()
    # disp = disp.resize(disp.shape[1], disp.shape[0]).cpu().numpy()
    
    print(img.shape, disp.shape)
    img = np.array(T.to_pil_image(T.resize(img[0], (img.shape[-1], img.shape[-2])))) 
    disp = T.resize(disp, (disp.shape[-1], disp.shape[-2])).squeeze().cpu().numpy()

    # disparity_np = np.reshape(disparity_np, (disparity_np.shape[1], disparity_np.shape[0]))
    # Normalize the disparity map to 0-255  
    disp_normalized = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply a color map for better visualization
    disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_PLASMA)

    cv2.imshow('Disparity', disp_color)
    cv2.imshow('Image', img)
    cv2.waitKey(1)


    # plt.imshow(disp, cmap='jet')
    # plt.colorbar()  
    # plt.show()