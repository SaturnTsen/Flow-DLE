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
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List
import math


def to_numpy_image(img):
    """
    Convert PIL Image or numpy array to uint8 numpy array [H, W, 3]
    """
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return img
    elif isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    else:
        raise ValueError("Image must be PIL.Image or numpy.ndarray")

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

def show_images_in_row(
    images: List[Image.Image],
    titles: List[str],
    padding: int = 10,
    title_height: int = 30,
    bg_color=(255, 255, 255),
):
    assert len(images) == len(titles), "images and titles must have the same length"


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


def euler_to_R(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to rotation matrix.
    
    Args:
        roll  : rotation around X axis (radians)
        pitch : rotation around Y axis (radians)
        yaw   : rotation around Z axis (radians)

    Returns:
        R : 3x3 rotation matrix
    """

    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)

    Rx = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ], dtype=np.float32)

    Ry = np.array([
        [ cp, 0, sp],
        [  0, 1,  0],
        [-sp, 0, cp]
    ], dtype=np.float32)

    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ], dtype=np.float32)

    # Z * Y * X
    return Rz @ Ry @ Rx

class MaskPainter:
    def __init__(self, img_np, brush_radius=40, fill_style="rgba(255, 255, 255, 0.4)"):
        """
        img_np: H x W x 3 uint8
        """
        img_np = to_numpy_image(img_np)
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
        img_np = to_numpy_image(img_np)
        self.img_np = img_np
        self.H, self.W, _ = img_np.shape

        self.radius = radius
        self.arrow_width = arrow_width

        self.points = []
        self._pending_src = None

        self.canvas = Canvas(width=self.W, height=self.H)
        self.canvas.put_image_data(self.img_np, 0, 0)

        self._bind_events()
        self._build_ui()

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

    def show(self):
        display(self.canvas, self.ui)

    def get_points(self):
        return list(self.points)


# ---------- main editor ----------
DEG2RAD = math.pi / 180.0

def colored_label(text, color):
    return f"<span style='color:{color}; font-weight:600'>{text}</span>"

def project_points(P, K):
    P = P / (P[:, 2:3] + 1e-8)
    uv = (K @ P.T).T
    return uv[:, :2]

def sliders_3_per_row(sliders):
    return widgets.VBox([
        widgets.HBox(sliders[:3]),
        widgets.HBox(sliders[3:6]),
    ])


class DepthPoseEditor:
    """
    True 3D pose editor with:
    - Left: camera view (RGB + projected 3D axes)
    - Right: top-down view (XZ plane)
    """

    def __init__(self, rgb, depth, max_display_width=640):
        self.rgb = to_numpy_image(rgb)
        self.depth = depth.astype(np.float32)
        self.H, self.W, _ = self.rgb.shape

        f = max(self.H, self.W)
        self.K = np.array([[f, 0, self.W / 2],
                           [0, f, self.H / 2],
                           [0, 0, 1]])

        self.src = self._make_pose_sliders(
            "Source",
            axis_colors={"x": "red", "y": "green", "z": "blue"}
        )
        self.tgt = self._make_pose_sliders(
            "Target",
            axis_colors={"x": "goldenrod", "y": "cyan", "z": "magenta"}
        )

        # main camera canvas
        display_width = min(self.W, max_display_width)
        display_height = int(self.H * display_width / self.W)
        self.canvas = Canvas(
            width=self.W,
            height=self.H,
            layout=widgets.Layout(
                width=f"{display_width}px",
                height=f"{display_height}px",
                border="1px solid #ccc"
            )
        )

        # top-down canvas
        self.top_canvas = Canvas(
            width=300,
            height=300,
            layout=widgets.Layout(
                width="300px",
                height="300px",
                border="1px solid #ccc",
                margin="0 0 0 10px"
            )
        )

        for s in self.src["all"] + self.tgt["all"]:
            s.observe(self._redraw, names="value")

        self._redraw()

    # ------------------------------------------------------------------

    def _make_pose_sliders(self, name, axis_colors):
        def slider(**kw):
            s = widgets.FloatSlider(**kw)
            s.description_allow_html = True
            return s

        sliders = {
            "x": slider(value=0.0, min=-1.0, max=1.0, step=0.01,
                        description=colored_label(f"{name} x", axis_colors["x"])),
            "y": slider(value=0.0, min=-1.0, max=1.0, step=0.01,
                        description=colored_label(f"{name} y", axis_colors["y"])),
            "z": slider(value=1.0, min=0.1, max=5.0, step=0.01,
                        description=colored_label(f"{name} z", axis_colors["z"])),
            "roll": slider(value=0.0, min=-180, max=180, step=1,
                           description=f"{name} roll"),
            "pitch": slider(value=-20.0, min=-180, max=180, step=1,
                            description=f"{name} pitch"),
            "yaw": slider(value=30.0, min=-180, max=180, step=1,
                          description=f"{name} yaw"),
        }
        sliders["all"] = [sliders[k] for k in
                          ("x", "y", "z", "roll", "pitch", "yaw")]
        return sliders

    # ------------------------------------------------------------------

    def _draw_axis(self, pose, colors):
        x, y, z = pose["x"].value, pose["y"].value, pose["z"].value
        roll = pose["roll"].value * DEG2RAD
        pitch = pose["pitch"].value * DEG2RAD
        yaw = pose["yaw"].value * DEG2RAD

        R = euler_to_R(roll, pitch, yaw)
        origin = np.array([[x, y, z]])

        axes = np.eye(3) * 0.2
        P = np.vstack([origin, origin + (R @ axes.T).T])
        uv = project_points(P, self.K)

        self.canvas.line_width = 4
        self.canvas.stroke_style = colors[0]
        self.canvas.stroke_line(*uv[0], *uv[1])
        self.canvas.stroke_style = colors[1]
        self.canvas.stroke_line(*uv[0], *uv[2])
        self.canvas.stroke_style = colors[2]
        self.canvas.stroke_line(*uv[0], *uv[3])

    # ------------------------------------------------------------------
    def _draw_topdown_axis(self, canvas, pose, colors, scale=80):
        """
        Top-down view (camera at bottom, looking upward)
        X -> right
        Z -> up (forward, away from camera)
        yaw = 0 points upward
        """
        x = pose["x"].value
        z = pose["z"].value
        yaw = pose["yaw"].value * DEG2RAD   

        cx = canvas.width // 2
        cy = canvas.height // 2 

        # world -> canvas
        ox = cx + x * scale
        oy = cy - z * scale   # +Z goes upward  

        # forward direction (camera looks toward +Z)
        dx = np.sin(yaw)
        dz = np.cos(yaw)    

        L = 0.25 * scale    

        canvas.line_width = 3   

        # forward (Z axis, facing direction)
        canvas.stroke_style = colors[2]
        canvas.stroke_line(
            ox, oy,
            ox + dx * L,
            oy - dz * L
        )   

        # right (X axis)
        canvas.stroke_style = colors[0]
        canvas.stroke_line(
            ox, oy,
            ox + dz * L,
            oy + dx * L
        )   

        # origin
        canvas.fill_style = colors[1]
        canvas.fill_circle(ox, oy, 4)

    def _redraw_topdown(self):
        c = self.top_canvas
        c.clear()

        c.stroke_style = "#eee"
        for i in range(0, c.width, 40):
            c.stroke_line(i, 0, i, c.height)
            c.stroke_line(0, i, c.width, i)

        self._draw_topdown_axis(c, self.src,
                                ("red", "black", "blue"))
        self._draw_topdown_axis(c, self.tgt,
                                ("orange", "black", "magenta"))

    # ------------------------------------------------------------------

    def _redraw(self, *_):
        self.canvas.clear()
        self.canvas.put_image_data(self.rgb, 0, 0)

        self._draw_axis(self.src, ("red", "green", "blue"))
        self._draw_axis(self.tgt, ("yellow", "cyan", "magenta"))

        self._redraw_topdown()

    # ------------------------------------------------------------------

    def show(self):
        views = widgets.HBox([
            self.canvas,
            self.top_canvas
        ])

        ui = widgets.VBox([
            views,
            widgets.HTML("<b>Source Frame</b>"),
            sliders_3_per_row(self.src["all"]),
            widgets.HTML("<b>Target Frame</b>"),
            sliders_3_per_row(self.tgt["all"]),
        ])
        display(ui)

    # ------------------------------------------------------------------

    def get_output(self):
        def read(p):
            return {
                "x": p["x"].value,
                "y": p["y"].value,
                "z": p["z"].value,
                "roll": p["roll"].value * DEG2RAD,
                "pitch": p["pitch"].value * DEG2RAD,
                "yaw": p["yaw"].value * DEG2RAD,
            }

        src = read(self.src)
        tgt = read(self.tgt)

        return {
            "source": src,
            "target": tgt,
            "relative": {
                "translation": {
                    "x": tgt["x"] - src["x"],
                    "y": tgt["y"] - src["y"],
                    "z": tgt["z"] - src["z"],
                },
                "rotation": {
                    "type": "euler",
                    "roll": tgt["roll"] - src["roll"],
                    "pitch": tgt["pitch"] - src["pitch"],
                    "yaw": tgt["yaw"] - src["yaw"],
                },
            }
        }