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

import ipywidgets as widgets
from ipycanvas import Canvas
from IPython.display import display
import numpy as np
import cv2
from PIL import Image
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed+np.random.randint(0, 1000000))

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
        color = (255, 0, 0) if idx % 2 == 0 else (0, 0, 255)  # Red for handle, Blue for target
        cv2.circle(image, (point[0], point[1]), radius=10, color=color, thickness=-1)

    return image

class MaskPainter:
    def __init__(self, img_np, brush_radius=40, fill_style="rgba(255, 255, 255, 0.4)"):
        """
        img_np: H x W x 3 uint8
        """
        if isinstance(img_np, np.ndarray):
            img_np = img_np
        elif isinstance(img_np, Image.Image):
            img_np = np.array(img_np)
        else:
            raise ValueError("img_np should be either a numpy array or a PIL Image.")
        self.img_np = img_np
        self.H, self.W, _ = img_np.shape
        self.brush_radius = brush_radius
        self.fill_style = fill_style
        self.mask = np.zeros((self.H, self.W), dtype=np.uint8)
        self.drawing = False

        self.canvas = Canvas(width=self.W, height=self.H)
        self.canvas.put_image_data(self.img_np, 0, 0)

        self._bind_events()
        self._build_ui()

    def _on_mouse_down(self, x, y):
        self.drawing = True

    def _on_mouse_up(self, x, y):
        self.drawing = False

    def _on_mouse_move(self, x, y):
        if not self.drawing:
            return

        self.canvas.fill_style = self.fill_style
        self.canvas.fill_circle(x, y, self.brush_radius)

        yy, xx = np.ogrid[:self.H, :self.W]
        circle = (xx - x) ** 2 + (yy - y) ** 2 <= self.brush_radius ** 2
        self.mask[circle] = 1

    def _bind_events(self):
        self.canvas.on_mouse_down(self._on_mouse_down)
        self.canvas.on_mouse_up(self._on_mouse_up)
        self.canvas.on_mouse_move(self._on_mouse_move)

    def _clear_mask(self, b=None):
        self.mask[:] = 0
        self.canvas.clear()
        self.canvas.put_image_data(self.img_np, 0, 0)

    def _build_ui(self):
        self.clear_btn = widgets.Button(description="Clear Mask")
        self.clear_btn.on_click(self._clear_mask)
        self.hint = widgets.Label("draw the mask above")
        self.ui = widgets.HBox([self.clear_btn, self.hint])

    def show(self):
        display(self.canvas, self.ui)

    def get_mask(self):
        return self.mask.copy()


class PointArrowEditor:
    def __init__(self, img_np, radius=6, arrow_width=2):
        if isinstance(img_np, np.ndarray):
            img_np = img_np
        elif isinstance(img_np, Image.Image):
            img_np = np.array(img_np)
        else:
            raise ValueError("img_np should be either a numpy array or a PIL Image.")
        self.img_np = img_np
        self.H, self.W, _ = img_np.shape

        self.radius = radius
        self.arrow_width = arrow_width

        self.points = []      # [(x, y), ...]
        self._pending_src = None

        self.canvas = Canvas(width=self.W, height=self.H)
        self.canvas.put_image_data(self.img_np, 0, 0)

        self._bind_events()
        self._build_ui()

    # -------- mouse callbacks --------
    def _on_mouse_down(self, x, y):
        if self._pending_src is None:
            # source point
            self._pending_src = (x, y)
            self._draw_source(x, y)
        else:
            # target point
            src = self._pending_src
            tgt = (x, y)
            self._pending_src = None

            self.points.extend([src, tgt])
            self._draw_target(x, y)
            self._draw_arrow(src, tgt)

    # -------- drawing helpers --------
    def _draw_source(self, x, y):
        self.canvas.fill_style = "red"
        self.canvas.fill_circle(x, y, self.radius)

    def _draw_target(self, x, y):
        self.canvas.stroke_style = "blue"
        self.canvas.line_width = 2
        self.canvas.stroke_circle(x, y, self.radius)

    def _draw_arrow(self, src, tgt):
        self.canvas.stroke_style = "yellow"
        self.canvas.line_width = self.arrow_width

        x0, y0 = src
        x1, y1 = tgt

        self.canvas.begin_path()
        self.canvas.move_to(x0, y0)
        self.canvas.line_to(x1, y1)
        self.canvas.stroke()

        # arrow head
        dx, dy = x1 - x0, y1 - y0
        L = np.hypot(dx, dy) + 1e-6
        ux, uy = dx / L, dy / L

        head_len = 10
        left = (x1 - head_len * (ux - uy),
                y1 - head_len * (uy + ux))
        right = (x1 - head_len * (ux + uy),
                 y1 - head_len * (uy - ux))

        self.canvas.begin_path()
        self.canvas.move_to(x1, y1)
        self.canvas.line_to(*left)
        self.canvas.move_to(x1, y1)
        self.canvas.line_to(*right)
        self.canvas.stroke()

    # -------- helpers --------
    def _bind_events(self):
        self.canvas.on_mouse_down(self._on_mouse_down)

    def _clear(self, b=None):
        self.points.clear()
        self._pending_src = None
        self.canvas.clear()
        self.canvas.put_image_data(self.img_np, 0, 0)

    def _build_ui(self):
        self.clear_btn = widgets.Button(description="Clear Points")
        self.clear_btn.on_click(self._clear)
        self.hint = widgets.Label("Click to add source/target points")
        self.ui = widgets.HBox([self.clear_btn, self.hint])

    # -------- public API --------
    def show(self):
        display(self.canvas, self.ui)

    def get_points(self):
        return list(self.points)
    
