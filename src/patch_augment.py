from functools import wraps
import time
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from PIL import Image


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result, total_time

    return timeit_wrapper


@timeit
def apply_patch_to_images(
    images: torch.Tensor, padded_patch: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Applies the patch to a batch of images.

    Args:
    - images: the original batch of images tensor (B, C, H, W).
    - padded_patch: the padded patch tensor (C, H, W) (same size as the images).
    - mask: the mask tensor (C, H, W) (same size as the images).

    Returns:
    - The batch of images with the patch applied.
    """
    # if images.shape[2:] != padded_patch.shape[1:]:
    #     raise ValueError("Image and patch must have the same shape.")

    # patched_image = images.clone()
    # mask = torch.ones_like(padded_patch)
    # mask[padded_patch == 0] = 0
    # patched_image = torch.mul((1 - mask), images) + torch.mul(mask, padded_patch)

    # return patched_image
    B, C, H, W = images.shape
    patch_c, patch_h, patch_w = padded_patch.shape

    if (patch_c, patch_h, patch_w) != (C, H, W):
        raise ValueError(
            f"Patch must have the same shape as the images (excluding batch size). "
            f"Expected {(C, H, W)}, but got {(patch_c, patch_h, patch_w)}."
        )

    stacked_patches = torch.stack([padded_patch] * B, dim=0)
    stacked_masks = torch.stack([mask] * B, dim=0)

    patched_images = (1 - stacked_masks) * images + stacked_masks * stacked_patches

    return patched_images


def pad_patch(patch: torch.Tensor, padding: list[int]) -> torch.Tensor:
    return F.pad(patch, padding, fill=0)


def crop_patch(patch: torch.Tensor, original_size: list[int]) -> torch.Tensor:
    _, h, w = original_size
    _, ph, pw = patch.shape
    start_y = (ph - h) // 2
    start_x = (pw - w) // 2
    return patch[:, start_y : start_y + h, start_x : start_x + w]


def save_image(tensor: torch.Tensor, file_path: str):
    pass
    # image_pil = F.to_pil_image(tensor)
    # image_pil.save(file_path)
    # print(f"Image saved to {file_path}")


# if __name__ == "__main__":
#     image = Image.open("Src/input_img/test_image.jpg")
#     image = F.to_tensor(image)

#     patch_size = (3, 256, 256)
#     padding = (patch_size[1] // 2, patch_size[2] // 2)

#     vanilla_results = []
#     perspective_results = []
#     affine_results = []
#     color_jitter_results = []

#     for i in range(100):
#         print(f"Run {i + 1}/100")

#         patch = torch.rand(patch_size, requires_grad=True)
#         original_patch = patch.clone()
#         save_image(patch, "random_patch.png")
#         patched_image, time_taken = apply_patch_to_image(image, patch)
#         save_image(patched_image, "patched_image.png")
#         vanilla_results.append(time_taken)

#         perspective_transform = T.RandomPerspective(distortion_scale=0.5, p=1.0)
#         padded_patch = pad_patch(original_patch.clone(), padding)
#         patch = perspective_transform(padded_patch)
#         cropped_patch = crop_patch(patch, patch_size)
#         # mask = torch.ones_like(patch)
#         alpha_mask = create_alpha_mask(patch)  # to avoid black pixels
#         patched_image, time_taken = apply_patch_to_image(image, patch)
#         save_image(patched_image, "patched_image_perspective.png")
#         perspective_results.append(time_taken)

#         affine_transform = T.RandomAffine(degrees=45, scale=(0.8, 1.2))
#         padded_patch = pad_patch(original_patch.clone(), padding)
#         patch = affine_transform(padded_patch)
#         cropped_patch = crop_patch(patch, patch_size)
#         # mask = torch.ones_like(patch)
#         alpha_mask = create_alpha_mask(patch)  # to avoid black pixels
#         patched_image, time_taken = apply_patch_to_image(image, patch)
#         save_image(patched_image, "patched_image_affine.png")
#         affine_results.append(time_taken)

#         color_jitter = T.ColorJitter(
#             brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
#         )
#         patch = color_jitter(original_patch)
#         mask = torch.ones_like(patch)
#         patched_image, time_taken = apply_patch_to_image(image, patch)
#         save_image(patched_image, "patched_image_color_jitter.png")
#         color_jitter_results.append(time_taken)

#     average_time = (
#         sum(vanilla_results)
#         + sum(perspective_results)
#         + sum(affine_results)
#         + sum(color_jitter_results)
#     ) / (
#         len(vanilla_results)
#         + len(perspective_results)
#         + len(affine_results)
#         + len(color_jitter_results)
#     )
#     print(f"Average time taken: {average_time:.4f} seconds")

#     vanilla_time = sum(vanilla_results) / len(vanilla_results)
#     perspective_time = sum(perspective_results) / len(perspective_results)
#     affine_time = sum(affine_results) / len(affine_results)
#     color_jitter_time = sum(color_jitter_results) / len(color_jitter_results)

#     print(f"Vanilla time: {vanilla_time:.4f} seconds")
#     print(f"Perspective time: {perspective_time:.4f} seconds")
#     print(f"Affine time: {affine_time:.4f} seconds")
#     print(f"Color jitter time: {color_jitter_time:.4f} seconds")
