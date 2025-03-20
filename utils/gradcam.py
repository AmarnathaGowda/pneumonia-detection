import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam(model, image, target_class, device):
    # Prepare the image
    image = image.to(device)
    image = image.unsqueeze(0)  # Add batch dimension

    # Forward pass
    output = model(image)
    output = output.squeeze()

    # Backward pass for the target class
    model.zero_grad()
    class_loss = output[target_class]
    class_loss.backward()

    # Get gradients and activations
    gradients = model.gradients
    activations = model.activations

    # Compute Grad-CAM
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
    heatmap /= np.max(heatmap)

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[3]))

    # Convert image to numpy for visualization
    image = image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]

    # Overlay heatmap on image
    cam_image = show_cam_on_image(image, heatmap, use_rgb=True)

    return cam_image