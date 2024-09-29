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

import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T

import wandb

from tqdm import tqdm
import warnings

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Generating Adversarial Patches")
parser.add_argument(
    "--data_root", type=str, help="path to dataset", default="Src/input_img"
)
parser.add_argument("--train_list", type=str, default="Src/list/test_list.txt")
parser.add_argument("--print_file", type=str, default="Src/list/printable30values.txt")
parser.add_argument("--distill_ckpt", type=str, default="models/guo/distill_model.ckpt")
parser.add_argument("--height", type=int, help="input image height", default=256)
parser.add_argument("--width", type=int, help="input image width", default=512)
parser.add_argument("-b", "--batch_size", type=int, help="mini-batch size", default=2)
parser.add_argument(
    "-j", "--num_threads", type=int, help="data loading workers", default=0
)
parser.add_argument("--lr", type=float, help="initial learning rate", default=1e-3)
parser.add_argument("--num_epochs", type=int, help="number of total epochs", default=40)
parser.add_argument(
    "--seed",
    type=int,
    help="seed for random functions, and network initialization",
    default=0,
)
parser.add_argument("--patch_size", type=int, help="Resolution of patch", default=256)
parser.add_argument(
    "--patch_shape", type=str, help="circle or square", default="circle"
)
parser.add_argument("--patch_path", type=str, help="Initialize patch from file")
parser.add_argument("--mask_path", type=str, help="Initialize mask from file")
parser.add_argument("--target_disp", type=int, default=120)
parser.add_argument(
    "--model",
    nargs="*",
    type=str,
    default="distill",
    choices=["distill"],
    help="Model architecture",
)
parser.add_argument("--name", type=str, help="output directory", default="result")
args = parser.parse_args()


def main():
    save_path = "Dst/checkpoints/" + args.name
    print("===============================")
    print('=> Everything will be saved to "{}"'.format(save_path))
    makedirs(save_path)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # setup your torchvision/anyother transforms here. This is for adding noise/perspective transforms and other changes to the patch
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((args.width, args.height)),
        ]
    )
    mask_transform = T.Compose(
        [
            T.RandomPerspective(distortion_scale=0.1, p=0.8),
            T.RandomAffine(degrees=20, scale=(0.35, 0.45), translate=(0.2, 0.05)),
        ]
    )
    patch_jitter = T.ColorJitter(brightness=.05, contrast=.1, saturation=0.1, hue=0.1)
        

    train_transform = T.Lambda(
        lambda x: torch.stack([transform(image) for image in x], dim=0)
    )

    train_set = LoadFromImageFile(
        args.data_root,
        args.train_list,
        seed=args.seed,
        train=True,
        monocular=True,
        transform=train_transform,
        extension=".png",
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_threads,
        pin_memory=True,
        drop_last=True,
    )

    print("===============================")
    # Attacked Models
    models = AdversarialModels(args)
    models.load_ckpt()

    # Patch and Mask
    # Initialize a random patch image
    patch_size = (3, args.patch_size, args.patch_size)
    patch_cpu = torch.rand(patch_size, requires_grad=True)
    mask_cpu = torch.ones_like(patch_cpu)

    colors = get_printable_colors(args.print_file)

    # Optimizer
    # pass the patch to the optimizer
    optimizer = torch.optim.Adam([patch_cpu], lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=50
    )
    
    wandb.init(
        project="RBE474x-pr3",
        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
        },
    )
    
    # Train
    print("===============================")
    print("Start training ...")
    start_time = time.time()

    plt.ion()  # Interactive mode on
    fig, ax = plt.subplots()

    # save loss values for plotting
    epoch_loss = []
    epoch_disp_loss = []
    epoch_tv_loss = []

    for epoch in range(args.num_epochs):
        ep_nps_loss, ep_tv_loss, ep_loss, ep_disp_loss = 0, 0, 0, 0
        ep_time = time.time()

        
        losses = []
        tv_losses = []
        disp_losses = []

        for i_batch, sample in tqdm(
            enumerate(train_loader),
            desc=f"Running epoch {epoch}",
            total=len(train_loader),
            leave=False,
        ):
            with torch.autograd.detect_anomaly():
                sample = to_cuda_vars(sample)  # send item to gpu
                # print('left', sample['left'].shape)
                sample.update(
                    models.get_original_disp(sample)
                )  # get non-attacked disparity

                img, original_disp = sample["left"], sample["original_distill_disp"]
                # print('thing', img.shape, original_disp.shape)
                # visualize_disparity(img, original_disp[0])
                patch = patch_cpu.cuda()
                mask = mask_cpu.cuda()

                # how to run inference/forward?
                # est_disp = models.distill(img)

                # set of printable colors used in the original paper are given below,
                # Src/list/printable30values.txt

                # pad patch to be the same size as the image
                padding = (
                    (args.height - patch_size[2]) // 2,
                    (args.width - patch_size[1]) // 2,
                )
                padded_patch = F.pad(patch, padding, fill=0)
                padded_mask = F.pad(mask, padding, fill=0)

                pad_and_mask = torch.cat((padded_patch.unsqueeze(0), padded_mask.unsqueeze(0)), dim=0) # trick to apply same transformation to both patch and mask
                # transform patch and maybe the mask corresponding to the transformed patch(binary iamge)
                patch_t, mask_t = mask_transform(pad_and_mask)
                
                #turn this off for now
                patch_t = patch_jitter(patch_t)

                # apply transformed patch to clean image
                patched_img = apply_patch_to_images(img, patch_t, mask_t)[0]
                # print(patched_img.shape)
                est_disp = models.distill(patched_img)
                fake_disp = torch.mul((1 - mask_t[0]), original_disp) + torch.mul(mask_t[0], args.target_disp)
                #print(mask_t.shape, original_disp.shape, est_disp.shape, fake_disp.shape, torch.max(est_disp), torch.max(fake_disp))
                #est_disp = fake_disp
                # if epoch % 20 == 0 and i_batch == 0:
                #     # visualize_disparity(img, original_disp[0])
                #     # time.sleep(10)
                #     # stacked_mask = torch.stack([mask_t] * patched_img.shape[0], dim=0)
                #     # visualize_disparity(patched_img, est_disp[0])

                    # visualize_disparity_2(img, original_disp[0], patched_img, est_disp[0])
                    


                # Loss

                # calculate the loss function here
                # nps_calc = NPSCalculator(args.print_file, 512)

                # nps_loss = nps_calc(patch_t)
                nps_loss = get_nps_score(patch_t, colors)
                tv_loss = get_tv_loss(patch)
                disp_loss = get_disp_loss(
                    original_disp, est_disp, mask_t[0], args.target_disp
                )
                # constants taken from paper
                loss = (
                    disp_loss
                    + nps_loss * 0.01
                    + torch.max(tv_loss, torch.tensor(0.1).cuda()) * 2.5
                )

                # ep_nps_loss += nps_loss.detach().cpu().numpy()
                ep_tv_loss += torch.max(tv_loss, torch.tensor(0.1).cuda()).detach().cpu().numpy()
                ep_disp_loss += disp_loss.detach().cpu().numpy()
                ep_loss += loss.detach().cpu().numpy()
                
                if i_batch == 0:
                    wandb_patch = wandb.Image(np.concatenate([
                        T.functional.to_pil_image(patch_cpu),
                        T.functional.to_pil_image(patch_t.cpu()),
                    ], axis=0))
                    
                    wandb_images = []
                    # wandb_patches = []
                    wandb_test = []
                    for batch in range(min(args.batch_size, 8)):
                        img1 = visualize_disparity_2(img[batch], original_disp[batch], display=False)
                        img2 = visualize_disparity_2(patched_img[batch], est_disp[batch], display=False)
                        
                        img3 = np.concatenate([
                                    disp_viz(torch.mul(mask_t[0], fake_disp[batch])), 
                                     disp_viz(torch.mul(mask_t[0], est_disp[batch]))
                            ], axis=1)
                        
                        wandb_images.append(
                            wandb.Image(
                                    np.concatenate([img1, img2], axis=0)
                                )
                            )
                        wandb_test.append(wandb.Image(img3))
                        
                    wandb.log(
                        {
                            "images/train": wandb_images,
                            "images/patch": wandb_patch,
                            "images/test": wandb_test,
                        }
                    )

                wandb.log(
                    {
                        "batchloss": loss.item(),
                        "batchtv": tv_loss.item(),
                        "batch_disp": disp_loss.item(),
                    }
                )
                
                losses.append(loss.item())
                tv_losses.append(tv_loss.item())
                disp_losses.append(disp_loss.item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                models.distill.zero_grad()

                patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                del patch_t, loss, nps_loss, tv_loss, disp_loss
                torch.cuda.empty_cache()

        ep_disp_loss = ep_disp_loss / len(train_loader)
        ep_nps_loss = ep_nps_loss / len(train_loader)
        ep_tv_loss = ep_tv_loss / len(train_loader)
        ep_loss = ep_loss / len(train_loader)
        scheduler.step(ep_loss)

        ep_time = time.time() - ep_time
        total_time = time.time() - start_time
        print("===============================")
        print(" FIN EPOCH: ", epoch)
        print("TOTAL TIME: ", format_time(int(total_time)))
        print("EPOCH TIME: ", format_time(int(ep_time)))
        print("EPOCH LOSS: ", ep_loss)
        print(" DISP LOSS: ", ep_disp_loss)
        print("  NPS LOSS: ", ep_nps_loss)
        print("   TV LOSS: ", ep_tv_loss)


        epoch_loss.append(ep_loss)
        epoch_disp_loss.append(ep_disp_loss)
        epoch_tv_loss.append(ep_tv_loss)
        
        wandb.log(
            {
                "epochloss": ep_loss,
                "epochtv": ep_tv_loss,
                "epoch_disp": ep_disp_loss
            }
        )
        
        # Update plot
        # ax.clear()
        # ax.plot(losses, label='Loss')
        # ax.plot(tv_losses, label='TV Loss')
        # ax.plot(disp_losses, label='Disp Loss')
        # ax.set_xlabel('Batch')
        # ax.set_ylabel('Loss')
        # ax.set_title('Training Loss Over Time')
        # ax.legend()
        # plt.pause(0.1)  # Pause to allow the plot to update

        #if epoch == 0 or epoch == args.num_epochs - 1:
        np.save(
            save_path + "/epoch_{}_patch.npy".format(str(epoch)), patch_cpu.data.numpy()
        )
        np.save(
            save_path + "/epoch_{}_mask.npy".format(str(epoch)), mask_cpu.data.numpy()
        )

    print("Training finished")
    # Plot loss
    # plt.plot(epoch_loss, label="Total Loss")
    # plt.plot(epoch_disp_loss, label="Disp Loss")
    plt.plot(epoch_tv_loss, label="TV Loss")
    plt.legend()
    plt.savefig(save_path + "/loss.png")


if __name__ == "__main__":
    main()
