"""
Grad-CAM for ResNet on CIFAR-10. Call run_gradcam(model) after training in the same session.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from models.gradcam import GradCAM

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def run_gradcam(model, num_images=6):
    """Run Grad-CAM on a few test images. Saves gradcam_resnet.png to current dir."""
    device = next(model.parameters()).device
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    images, labels = next(iter(torch.utils.data.DataLoader(testset, batch_size=num_images, shuffle=True)))
    images = images.to(device)
    images.requires_grad_(True)

    gradcam = GradCAM(model, model.layer3)
    names = testset.classes

    fig, axes = plt.subplots(2, num_images, figsize=(2 * num_images, 4))
    if num_images == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_images):
        x = images[i : i + 1]
        _, cam = gradcam(x)

        img = images[i].detach().cpu().permute(1, 2, 0).numpy()
        img = np.clip(img * np.array(STD) + np.array(MEAN), 0, 1)

        pred = model(x).argmax(dim=1).item()
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{names[labels[i]]} → {names[pred]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(img)
        axes[1, i].imshow(cam, cmap="jet", alpha=0.5)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_resnet.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: gradcam_resnet.png")
