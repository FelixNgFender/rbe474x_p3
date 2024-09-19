import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# A random patch tensor with gradients enabled
patch_size = (100, 100)  # Smaller patch size for visibility
patch = torch.rand((3, *patch_size), requires_grad=True)  # Include channel dimension

# Augmentation techniques
transform = T.Compose(
    [
        T.RandomPerspective(distortion_scale=0.5, p=1.0),
        T.RandomRotation(degrees=45),
        T.RandomResizedCrop(size=patch_size, scale=(0.8, 1.0)),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
    ]
)

# Apply augmentations to the patch
augmented_patch = transform(F.to_pil_image(patch.detach().cpu()))

# Convert augmented patch back to tensor
augmented_patch = F.to_tensor(augmented_patch)

# Load an example image
image = Image.open("Src/input_img/test_image.jpg")
image = F.to_tensor(image)

# Ensure the patch is applied at a noticeable location
# For simplicity, let's place it at the center of the image
image_height, image_width = image.shape[1], image.shape[2]
center_y, center_x = image_height // 2, image_width // 2
start_y, start_x = center_y - patch_size[0] // 2, center_x - patch_size[1] // 2

# Create a mask (containing 0's and 1's)
mask = torch.zeros_like(image)
mask[:, start_y : start_y + patch_size[0], start_x : start_x + patch_size[1]] = 1

# Resize augmented patch to match the mask dimensions
augmented_patch_resized = F.resize(augmented_patch, (patch_size[0], patch_size[1]))

# Apply the patch using the given equation
patched_image = image.clone()  # Clone the original image to avoid modifying it
patched_image[
    :, start_y : start_y + patch_size[0], start_x : start_x + patch_size[1]
] = augmented_patch_resized

# Convert patched image to PIL for visualization
patched_image_pil = F.to_pil_image(patched_image)

# Display the original and patched images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(F.to_pil_image(image))
ax[0].set_title("Original Image")
ax[1].imshow(patched_image_pil)
ax[1].set_title("Patched Image")

# Save the plot to a file
plt.savefig("patched_image_comparison.png")
