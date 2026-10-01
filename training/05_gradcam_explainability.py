from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

CHECKPOINT = Path("outputs/efficientnet_b0/efficientnet_b0_best.pth")
IMAGE_PATH = Path("datasets/example_leaf.jpg")
OUTPUT_PATH = Path("outputs/gradcam_example.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    if not CHECKPOINT.exists() or not IMAGE_PATH.exists():
        print("Required model checkpoint or test image missing for Grad-CAM generation.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 27)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    target_layer = model.features[-1]
    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(IMAGE_PATH).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    model.zero_grad(set_to_none=True)
    logits = model(input_tensor)
    class_id = int(logits.argmax(dim=1).item())
    logits[0, class_id].backward()

    feature_maps = activations[0]
    gradient_maps = gradients[0]
    weights = gradient_maps.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feature_maps).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(
        cam, size=(224, 224), mode="bilinear", align_corners=False
    )[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    display_image = image.resize((224, 224))
    plt.figure(figsize=(7, 7))
    plt.imshow(display_image)
    plt.imshow(cam, cmap="jet", alpha=0.4)
    plt.title(f"Grad-CAM — predicted class index {class_id}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved Grad-CAM visualization to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
