"""Perform inference on a video file using a trained model."""

from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from models.adversarial_models import AdversarialModels
import argparse

from utils.utils import visualize_disparity

# from utils.utils import visualize_disparity

parser = argparse.ArgumentParser(description="Inferencing Monocular Depth Estimation")
parser.add_argument("--distill_ckpt", type=str, default="models/guo/distill_model.ckpt")
parser.add_argument(
    "--model",
    nargs="*",
    type=str,
    default="distill",
    choices=["distill"],
    help="Model architecture",
)
parser.add_argument(
    "--input",
    type=str,
    default="fake.mp4",
    help="Path to the input video file",
)
parser.add_argument(
    "--output",
    type=str,
    default="output.mp4",
    help="Path to the output video file",
)
parser.add_argument(
    "--fps",
    type=float,
    default=30,
    help="Frames per second for the output video",
)
args = parser.parse_args()


def video_inference():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = AdversarialModels(args)
    models.load_ckpt()

    cap = cv2.VideoCapture(args.input)
    res = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (512, 256))

    # i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        # if i == 0:
        #     print("")
        #     i += 1
        # else:
        #     return
        if not ret:
            print("finished. saving...")
            break
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # im_pil = Image.fromarray(rgb_frame)
        im_pil = Image.fromarray(frame)
        rgb = transforms.ToTensor()(im_pil).to(device)
        rgb = F.resize(
            rgb,
            [512, 256],
        )
        disp = models.distill(rgb)
        # remove the batch dimension
        # disp = disp.squeeze(0)
        # rgb = rgb.squeeze(0)
        # print("disp", disp.shape)
        # print("rgb", rgb.shape)
        # visualize_disparity(rgb, disp)

        disp = disp.squeeze().detach().cpu().numpy()
        disp_normalized = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )

        disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_PLASMA)
        # cv2.imshow("Disparity", disp_color)

        print("here", disp_color.shape)  # (256, 512, 3)
        disp_color = cv2.resize(disp_color, (512, 256))
        res.write(disp_color)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    res.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_inference()
