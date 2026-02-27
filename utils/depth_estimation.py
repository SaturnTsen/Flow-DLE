import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from PIL.Image import Image

def estimate_depth_from_image(
    img: Image,
    device,
    output_uint8: bool = True,
    depth_model: torch.nn.Module = None,
):
    # Load model
    if depth_model is None:
        print("Depth Model not provided, loading a model from scratch...")
        depth_model = DepthAnything.from_pretrained(f"LiheYoung/depth_anything_vitl14").to(device).eval()

    # Preprocess
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        PrepareForNet(),
    ])

    if not isinstance(img, np.ndarray):
        img = np.array(img.convert("RGB"))

    image = img.astype(np.float32) / 255.0
    h, w = image.shape[:2]

    image = transform({"image": image})["image"]
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        depth = depth_model(image)
        depth = F.interpolate(
            depth[None],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    depth = depth.cpu().numpy()

    if output_uint8:
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth = (depth * 255.0).astype(np.uint8)

    return depth