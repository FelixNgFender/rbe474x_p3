import torch
import argparse
import cv2
import numpy as np
import os
from patch_augment import apply_patch_to_images
from models.adversarial_models import AdversarialModels
from utils.utils import makedirs, to_cuda_vars
from utils.dataloader import SingleImageLoader

import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import torchvision.transforms as T
import torchvision.transforms.functional as F

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Generating Adversarial Patches")
parser.add_argument("--img_path", type=str, default="Src/input_img/test_image.png")
parser.add_argument("--distill_ckpt", type=str, default="models/guo/distill_model.ckpt")
parser.add_argument("--height", type=int, help="input image height", default=256)
parser.add_argument("--width", type=int, help="input image width", default=512)
parser.add_argument(
    "--seed",
    type=int,
    help="seed for random functions, and network initialization",
    default=0,
)
parser.add_argument("--patch_size", type=int, help="Resolution of patch", default=256)
parser.add_argument(
    "--patch_path",
    type=str,
    default="Dst/checkpoints/result/epoch_970_patch.npy",
)
parser.add_argument(
    "--model",
    nargs="*",
    type=str,
    default="distill",
    choices=["distill"],
    help="Model architecture",
)
parser.add_argument("--name", type=str, help="output directory", default="")
args = parser.parse_args()


def main():
    if args.name:
        save_path = os.path.join("Dst/test_result", args.name)
    else:
        name = (os.path.splitext(args.patch_path)[0]).replace("/", "_")
        save_path = os.path.join("Dst/test_result", name)
    print("===============================")
    print('=> Everything will be saved to "{}"'.format(save_path))
    makedirs(save_path)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # DataLoader
    test_transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((args.height, args.width)),
        ]
    )
    mask_transform = T.Compose(
        [
            T.RandomPerspective(distortion_scale=0.1, p=0.8),
            T.RandomAffine(degrees=20, scale=(0.35, 0.45), translate=(0.2, 0.05)),
        ]
    )

    patch_jitter = T.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.1, hue=0.1)

    test_set = SingleImageLoader(
        img_path=args.img_path, seed=args.seed, transform=test_transform
    )
    # test_set = LoadFromImageFile(
    #     args.img_path,
    #     args.train_list,
    #     seed=args.seed,
    #     train=False,
    #     monocular=True,
    #     transform=test_transform,
    #     extension=".png",
    # )

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1)

    print("===============================")
    # Attacked Models
    models = AdversarialModels(args)
    models.load_ckpt()

    patch_size = (3, args.patch_size, args.patch_size)
    patch_cpu_as_array = np.load(args.patch_path)
    patch_cpu = torch.from_numpy(patch_cpu_as_array)
    print(patch_cpu.shape)
    mask_cpu = torch.ones_like(patch_cpu)

    print("===============================")
    for i_batch, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        sample = to_cuda_vars(sample)  # send item to gpu
        sample.update(
            models.get_original_disp(to_cuda_vars(sample))
        )  # get non-attacked disparity

        img, original_disp = sample["left"], sample["original_distill_disp"]
        patch, mask = patch_cpu.cuda(), mask_cpu.cuda()

        # pad patch to be the same size as the image
        padding = (
            (args.width - patch_size[1]) // 2,
            (args.height - patch_size[2]) // 2,
        )
        padded_patch = F.pad(patch, padding, fill=0)
        padded_mask = F.pad(mask, padding, fill=0)

        pad_and_mask = torch.cat(
            (padded_patch.unsqueeze(0), padded_mask.unsqueeze(0)), dim=0
        )  # trick to apply same transformation to both patch and mask
        # transform patch and maybe the mask corresponding to the transformed patch(binary iamge)
        patch_t, mask_t = mask_transform(pad_and_mask)

        # turn this off for now
        patch_t = patch_jitter(patch_t)

        # apply transformed patch to clean image
        print(img.shape, patch_t.shape, mask_t.shape)
        patched_img = apply_patch_to_images(img, patch_t, mask_t)[0]
        est_disp = models.distill(patched_img)
        patched_img = np.transpose(patched_img[0].data.cpu().numpy(), (1, 2, 0))
        original_img = np.transpose(img[0].data.cpu().numpy(), (1, 2, 0))
        attacked_disp = est_disp[0, 0].data.cpu().numpy()
        original_disp = original_disp[0, 0].data.cpu().numpy()

        cv2.imwrite(save_path + "/img_attacked.png", patched_img * 255)
        plt.imsave(
            save_path + "/disp_attacked.png",
            attacked_disp,
            cmap="plasma",
            vmin=0,
            vmax=40,
        )
        cv2.imwrite(save_path + "/img_original.png", original_img * 255)
        plt.imsave(
            save_path + "/disp_original.png",
            original_disp,
            cmap="plasma",
            vmin=0,
            vmax=40,
        )
        plt.imsave(
            save_path + "/difference.png",
            abs(attacked_disp - original_disp),
            cmap="bwr",
        )

    print("Finish test !")


if __name__ == "__main__":
    main()
