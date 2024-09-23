import random
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image


def apply_patch_to_image(
    image: torch.Tensor, patch: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Applies the patch to a small, random part of the image based on the mask.

    Args:
    - image: the original image tensor (C, H, W).
    - patch: the patch tensor (C, h, w) (smaller than the image).
    - mask: the mask tensor (same shape as the patch, binary values 0 or 1).

    Returns:
    - The image with the patch applied at the given location.
    """
    img_c, img_h, img_w = image.shape
    patch_c, patch_h, patch_w = patch.shape

    if patch_c != img_c:
        raise ValueError(
            f"Patch and image must have the same number of channels. Image has {img_c}, patch has {patch_c}"
        )
    if patch_h > img_h or patch_w > img_w:
        raise ValueError("Patch size must be smaller than or equal to the image size.")

    x = random.randint(0, img_w - patch_w)
    y = random.randint(0, img_h - patch_h)

    patched_image = image.clone()
    patched_image[:, y : y + patch_h, x : x + patch_w] = (1 - mask) * image[
        :, y : y + patch_h, x : x + patch_w
    ] + mask * patch

    return patched_image


def create_alpha_mask(patch: torch.Tensor) -> torch.Tensor:
    alpha_mask = torch.ones_like(patch)
    alpha_mask[patch == 0] = 0
    return alpha_mask


def pad_patch(patch: torch.Tensor, padding: list[int]) -> torch.Tensor:
    return F.pad(patch, padding, fill=0)


def crop_patch(patch: torch.Tensor, original_size: list[int]) -> torch.Tensor:
    _, h, w = original_size
    _, ph, pw = patch.shape
    start_y = (ph - h) // 2
    start_x = (pw - w) // 2
    return patch[:, start_y : start_y + h, start_x : start_x + w]


def save_image(tensor: torch.Tensor, file_path: str):
    image_pil = F.to_pil_image(tensor)
    image_pil.save(file_path)
    print(f"Image saved to {file_path}")


image = Image.open("Src/input_img/test_image.jpg")
image = F.to_tensor(image)

patch_size = (3, 256, 256)
padding = (patch_size[1] // 2, patch_size[2] // 2)

patch = torch.rand(patch_size, requires_grad=True)
original_patch = patch.clone()
save_image(patch, "random_patch.png")
mask = torch.ones_like(patch)
patched_image = apply_patch_to_image(image, patch, mask)
save_image(patched_image, "patched_image.png")

perspective_transform = T.RandomPerspective(distortion_scale=0.5, p=1.0)
padded_patch = pad_patch(original_patch.clone(), padding)
patch = perspective_transform(padded_patch)
cropped_patch = crop_patch(patch, patch_size)
# mask = torch.ones_like(patch)
alpha_mask = create_alpha_mask(patch)  # to avoid black pixels
patched_image = apply_patch_to_image(image, patch, alpha_mask)
save_image(patched_image, "patched_image_perspective.png")

affine_transform = T.RandomAffine(degrees=45, scale=(0.8, 1.2))
padded_patch = pad_patch(original_patch.clone(), padding)
patch = affine_transform(padded_patch)
cropped_patch = crop_patch(patch, patch_size)
# mask = torch.ones_like(patch)
alpha_mask = create_alpha_mask(patch)  # to avoid black pixels
patched_image = apply_patch_to_image(image, patch, alpha_mask)
save_image(patched_image, "patched_image_affine.png")

color_jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
patch = color_jitter(original_patch)
mask = torch.ones_like(patch)
patched_image = apply_patch_to_image(image, patch, mask)
save_image(patched_image, "patched_image_color_jitter.png")
