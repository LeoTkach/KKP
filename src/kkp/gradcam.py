"""Grad-CAM saliency maps for CNN interpretability."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

if TYPE_CHECKING:
    from torchvision.transforms import Compose


def get_gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "efficientnet_b0":
        return model.features[-1]
    msg = f"Grad-CAM target layer not defined for {model_name}"
    raise ValueError(msg)


def compute_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    *,
    target_layer: nn.Module,
    target_class: int,
) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        activations.append(output)

    def backward_hook(_module: nn.Module, _grad_input: tuple, grad_output: tuple) -> None:
        gradients.append(grad_output[0])

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    logits = model(input_tensor)
    score = logits[0, target_class]
    score.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    grads = gradients[0][0]
    acts = activations[0][0]
    weights = grads.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=0))
    cam = cam.detach().cpu().numpy()
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    return cam.astype(np.float32)


def overlay_gradcam(
    image: Image.Image,
    cam: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Return RGB uint8 array: original image with jet colormap overlay."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    cam_resized = (
        F.interpolate(
            cam_tensor,
            size=(rgb.shape[0], rgb.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )
    heatmap = _jet_rgb(cam_resized)
    blended = np.clip((1 - alpha) * rgb + alpha * heatmap, 0.0, 1.0)
    return (blended * 255).astype(np.uint8)


def _jet_rgb(values: np.ndarray) -> np.ndarray:
    """Map float array [0, 1] to jet-like RGB without matplotlib."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def explain_image(
    model: nn.Module,
    model_name: str,
    image: Image.Image,
    transform: Compose,
    device: torch.device,
    *,
    target_class: int,
    image_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (original RGB uint8 at eval resolution, Grad-CAM overlay RGB uint8)."""
    eval_image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    target_layer = get_gradcam_target_layer(model, model_name)
    model.eval()
    cam = compute_gradcam(
        model,
        tensor,
        target_layer=target_layer,
        target_class=target_class,
    )
    original = np.asarray(eval_image, dtype=np.uint8)
    overlay = overlay_gradcam(eval_image, cam)
    return original, overlay
