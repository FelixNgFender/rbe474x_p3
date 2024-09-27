import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
import argparse
import cv2
import numpy as np
import time
import os
from models.adversarial_models import AdversarialModels
from utils.dataloader import LoadFromImageFile
from utils.utils import makedirs, to_cuda_vars, format_time
from patch_augment import *
from utils.utils import *

from torch import nn
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Generating Adversarial Patches')
parser.add_argument('--data_root', type=str, help='path to dataset', default='Src/input_img')
parser.add_argument('--train_list', type=str, default='Src/list/test_list.txt')
parser.add_argument('--print_file', type=str, default='Src/list/printable30values.txt')
parser.add_argument('--distill_ckpt', type=str, default="models/guo/distill_model.ckpt")
parser.add_argument('--height', type=int, help='input image height', default=256)
parser.add_argument('--width', type=int, help='input image width', default=512)
parser.add_argument('-b', '--batch_size', type=int, help='mini-batch size', default=2)
parser.add_argument('-j', '--num_threads', type=int, help='data loading workers', default=0)
parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-3)
parser.add_argument('--num_epochs', type=int, help='number of total epochs', default=40)
parser.add_argument('--seed', type=int, help='seed for random functions, and network initialization', default=0)
parser.add_argument('--patch_size', type=int, help='Resolution of patch', default=256)
parser.add_argument('--patch_shape', type=str, help='circle or square', default='circle')
parser.add_argument('--patch_path', type=str, help='Initialize patch from file')
parser.add_argument('--mask_path', type=str, help='Initialize mask from file')
parser.add_argument('--target_disp', type=int, default=120)
parser.add_argument('--model', nargs='*', type=str, default='distill', choices=['distill'], help='Model architecture')
parser.add_argument('--name', type=str, help='output directory', default="result")
args = parser.parse_args()


def main():
    save_path = 'Dst/checkpoints/' + args.name
    print('===============================')
    print('=> Everything will be saved to \"{}\"'.format(save_path))
    makedirs(save_path)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # setup your torchvision/anyother transforms here. This is for adding noise/perspective transforms and other changes to the patch
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((args.width, args.height)),
    ])
    
    train_transform = T.Lambda(lambda x: torch.stack([transform(image) for image in x], dim=0).to(x.device))
    
    train_set = LoadFromImageFile(
        args.data_root,
        args.train_list,
        seed=args.seed,
        train=True,
        monocular=True,
        transform=train_transform,
        extension=".png"
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,  
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_threads,
        pin_memory=True,
        drop_last=True
    )

    print('===============================')
    # Attacked Models
    models = AdversarialModels(args)
    models.load_ckpt()

    # Patch and Mask
    # Initialize a random patch image
    patch_size = (3, 256, 256)
    padding = (patch_size[1] // 2, patch_size[2] // 2)
    patch_cpu = torch.rand(patch_size, requires_grad=True)
    
    # Initialize a mask
    mask_cpu = create_alpha_mask(patch_cpu)
    
    # Optimizer
    # pass the patch to the optimizer
    optimizer = torch.optim.Adam([patch_cpu], lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

    # Train
    print('===============================')
    print("Start training ...")
    start_time = time.time()
    for epoch in range(args.num_epochs):
        ep_nps_loss, ep_tv_loss, ep_loss, ep_disp_loss = 0, 0, 0, 0
        ep_time = time.time()

        for i_batch, sample in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=len(train_loader), leave=False):
            with torch.autograd.detect_anomaly():
                sample = to_cuda_vars(sample)  # send item to gpu
                # print('left', sample['left'].shape)
                sample.update(models.get_original_disp(sample))  # get non-attacked disparity

                img, original_disp = sample['left'], sample['original_distill_disp']
                print('thing', img.shape, original_disp.shape)
                visualize_disparity(img[0], original_disp[0])
                patch, mask = patch_cpu.cuda(), mask_cpu.cuda()

                # transform patch and maybe the mask corresponding to the transformed patch(binary iamge)
                patch_t, mask_t = patch, mask

                # apply transformed patch to clean image

                # Loss
                # calculate the loss function here
                nps_calc = NPSCalculator(args.print_file, args.height)

                nps_loss = nps_calc(patch_t)
                tv_loss = torch.tensor(0.0, requires_grad=True)
                disp_loss = torch.tensor(0.0, requires_grad=True)
                loss = torch.tensor(0.0, requires_grad=True)

                ep_nps_loss += nps_loss.detach().cpu().numpy()
                ep_tv_loss += tv_loss.detach().cpu().numpy()
                ep_disp_loss += disp_loss.detach().cpu().numpy()
                ep_loss += loss.detach().cpu().numpy()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                models.distill.zero_grad()

                patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                del patch_t, loss, nps_loss, tv_loss, disp_loss
                torch.cuda.empty_cache()

        ep_disp_loss = ep_disp_loss/len(train_loader)
        ep_nps_loss = ep_nps_loss/len(train_loader)
        ep_tv_loss = ep_tv_loss/len(train_loader)
        ep_loss = ep_loss/len(train_loader)
        scheduler.step(ep_loss)

        ep_time = time.time() - ep_time
        total_time = time.time() - start_time
        print('===============================')
        print(' FIN EPOCH: ', epoch)
        print('TOTAL TIME: ', format_time(int(total_time)))
        print('EPOCH TIME: ', format_time(int(ep_time)))
        print('EPOCH LOSS: ', ep_loss)
        print(' DISP LOSS: ', ep_disp_loss)
        print('  NPS LOSS: ', ep_nps_loss)
        print('   TV LOSS: ', ep_tv_loss)
        np.save(save_path + '/epoch_{}_patch.npy'.format(str(epoch)), patch_cpu.data.numpy())
        np.save(save_path + '/epoch_{}_mask.npy'.format(str(epoch)), mask_cpu.data.numpy())

# def get_printable_colors(file):
#     colors = []
#     with open(file) as f:
#         for line in f:
#             # print(line.split(","), type(line.split(",")))
#             r, g, b = line.split(",")
#             b = b.strip() # Remove the newline character
#             colors.append(np.array([r, g, b], dtype=np.float32))

#     colors = torch.Tensor(colors)
#     return colors

# def get_nps_score(patch : torch.Tensor, colors):
#     colors = colors.to(patch.device)
#     # print(colors.device, patch.device)
    
#     # Slow Implementation
#     loss = 0
#     t = time.time()
#     for y, x in zip(range(patch.shape[1]), range(patch.shape[2])):
#         loss += min([torch.norm(patch[:, y, x] - color) for color in colors])
#     print('Time taken for NPS Loss:', time.time() - t)

#     return loss

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

if __name__ == '__main__':
    main()
