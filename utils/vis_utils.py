# *************************************************************************
# Copyright (2026) Yiming Chen, Linh Vu Tu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *************************************************************************

from typing import List
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def overlay_points_and_mask(image, points, mask):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = image.copy()
    # Overlay mask (red)
    red_mask = np.zeros_like(image)
    red_mask[..., 0] = mask
    image = np.where(red_mask > 0, red_mask, image)

    # Overlay points
    for idx, point in enumerate(points):
        color = (255, 0, 0) if idx % 2 == 0 else (
            0, 0, 255)  # Red for handle, Blue for target
        cv2.circle(image, (point[0], point[1]),
                   radius=10, color=color, thickness=-1)

    return image


def show_images_in_row(
    images: List[Image.Image],
    titles: List[str],
    padding: int = 10,
    title_height: int = 30,
    bg_color=(255, 255, 255),
):
    assert len(images) == len(
        titles), "images and titles must have the same length"

    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths) + padding * (len(images) - 1)
    max_height = max(heights) + title_height

    canvas = Image.new("RGB", (total_width, max_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    font = ImageFont.load_default()

    x_offset = 0
    for img, title in zip(images, titles):
        canvas.paste(img, (x_offset, 0))

        if title and font is not None:
            # Pillow >= 10: 使用 textbbox
            bbox = draw.textbbox((0, 0), title, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            text_x = x_offset + (img.width - text_w) // 2
            text_y = img.height + (title_height - text_h) // 2
            draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)

        x_offset += img.width + padding

    return canvas