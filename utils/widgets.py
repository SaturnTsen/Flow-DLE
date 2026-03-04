import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ipywidgets as widgets
from ipycanvas import Canvas
from IPython.display import display
import matplotlib.pyplot as plt
from typing import Dict, Optional
from .calc_utils import (
    to_numpy_image,
    pred_unidepth,
    transform_mesh,
    unidepth_to_trimesh,
    euler_xyz_to_matrix,
    transform_depth_map,
    transform_rgb_map,
    clean_mask_by_depth
)

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

# Helps to define source and target point pairs (for dragging)
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

class MaskedMeshTransformWidget:
    def __init__(
        self,
        img: Image.Image,
        mask: np.ndarray,
        depth_model,
        resolution_level: int = 9,
    ):
        """
        img: PIL.Image (RGB)
        mask: (H,W) bool / {0,1}
        depth_model: UniDepth model
        """

        self.img = img
        self.mask = mask
        self.depth_model = depth_model
        self.resolution_level = resolution_level

        self.base_mesh = None
        self.current_mesh = None

        self.transform = {
            "rx": 0.0, "ry": 0.0, "rz": 0.0,
            "tx": 0.0, "ty": 0.0, "tz": 0.0,
        }

        self._build_widgets()
        self._bind_callbacks()

        self._init_mesh()

    def _init_mesh(self):
        preds = pred_unidepth(self.img, self.depth_model, self.resolution_level)
        self.base_mesh = unidepth_to_trimesh(self.img, preds, self.mask, flip_y=True, flip_z=True)

    def _build_widgets(self):
        self.rx = widgets.FloatSlider(
            value=0.0, min=-180.0, max=180.0,
            step=1.0, description="Rot X",
            continuous_update=False,
        )
        self.ry = widgets.FloatSlider(
            value=0.0, min=-180.0, max=180.0,
            step=1.0, description="Rot Y",
            continuous_update=False,
        )
        self.rz = widgets.FloatSlider(
            value=0.0, min=-180.0, max=180.0, step=1.0,
            description="Rot Z",
            continuous_update=False,
        )

        self.tx = widgets.FloatSlider(
            value=0.0, min=-1.0, max=1.0, step=0.01,
            description="Trans X",
            continuous_update=False,
        )
        self.ty = widgets.FloatSlider(
            value=0.0, min=-1.0, max=1.0, step=0.01,
            description="Trans Y",
            continuous_update=False,
        )
        self.tz = widgets.FloatSlider(
            value=0.0, min=-1.0, max=1.0, step=0.01,
            description="Trans Z",
            continuous_update=False,
        )

        self.render_btn = widgets.Button(
            description="Go",
            button_style="success",
        )

        self.status = widgets.Label(value="👈 Click the button to proceed")

        self.bottom = widgets.HBox([
            self.render_btn,
            self.status,
        ])

        self.controls = widgets.VBox([
            self.rx, self.ry, self.rz,
            self.tx, self.ty, self.tz,
            self.bottom,
        ])

        self.ui = self.controls

    def _bind_callbacks(self):
        for s in [self.rx, self.ry, self.rz, self.tx, self.ty, self.tz]:
            s.observe(self._on_slider_change, names="value")

        self.render_btn.on_click(self._on_btn_clicked)

    def _on_slider_change(self, change):
        self.transform.update({
            "rx": self.rx.value,
            "ry": self.ry.value,
            "rz": self.rz.value,
            "tx": self.tx.value,
            "ty": self.ty.value,
            "tz": self.tz.value,
        })
        self.status.value = "👈 Click the button to proceed"

        if self.base_mesh is not None:
            self.current_mesh = transform_mesh(
                self.base_mesh,
                self.transform["rx"], self.transform["ry"], self.transform["rz"],
                self.transform["tx"], self.transform["ty"], self.transform["tz"],
            )

    def _on_btn_clicked(self, b):
        t = self.transform
        self.current_mesh = transform_mesh(
            self.base_mesh,
            t["rx"], t["ry"], t["rz"],
            t["tx"], t["ty"], t["tz"],
        )
        self.status.value = "Done!"

    def get_mesh(self):
        return self.current_mesh

    def get_transformation(self):
        t = self.transform
        return euler_xyz_to_matrix(
            t["rx"], t["ry"], t["rz"],
            t["tx"], t["ty"], t["tz"],
        )

    def show(self):
        display(self.ui)


class MaskedDepthTransformWidget:
    def __init__(
        self,
        img: Image.Image,
        mask: np.ndarray,
        depth_preds: Dict[str, np.ndarray],
        resolution_level: int = 9,
        init_transform: Optional[Dict[str, float]] = None,
    ):
        """
        img: PIL.Image (RGB)
        mask: (H, W) {0,1} uint8 / bool
        depth_model: UniDepth model
        """
        self.img = img
        self.mask_raw = mask.astype(bool)
        self.mask = self.mask_raw.copy()
        self.resolution_level = resolution_level

        # predict depth once
        self.depth = depth_preds["depth"]
        self.intrinsics = depth_preds["intrinsics"]

        default_transform = {
            "rx": 0.0, "ry": 0.0, "rz": 0.0,
            "tx": 0.0, "ty": 0.0, "tz": 0.0,
        }

        if init_transform is None:
            self.transform = default_transform.copy()
        else:
            self.transform = {**default_transform, **init_transform}

        self._build_widgets()
        self._bind_callbacks()
        self._render()

    def _build_widgets(self):
        t = self.transform
        self.out = widgets.Output()
        self.rx = widgets.FloatSlider(
            value=t["rx"], min=-180.0, max=180.0, step=1.0,
            description="Rot X", continuous_update=False
        )
        self.ry = widgets.FloatSlider(
            value=t["ry"], min=-180.0, max=180.0, step=1.0,
            description="Rot Y", continuous_update=False
        )
        self.rz = widgets.FloatSlider(
            value=t["rz"], min=-180.0, max=180.0, step=1.0,
            description="Rot Z", continuous_update=False
        )
        self.tx = widgets.FloatSlider(
            value=t["tx"], min=-10.0, max=10.0, step=0.01,
            description="Trans X", continuous_update=False
        )
        self.ty = widgets.FloatSlider(
            value=t["ty"], min=-10.0, max=10.0, step=0.01,
            description="Trans Y", continuous_update=False
        )
        self.tz = widgets.FloatSlider(
            value=t["tz"], min=-10.0, max=10.0, step=0.01,
            description="Trans Z", continuous_update=False
        )
        self.feather_slider = widgets.IntSlider(
            value=5, min=0, max=30, step=1,
            description="Feather px", continuous_update=False
        )

        self.feather_btn = widgets.Button(
            description="Apply Feather",
            button_style="info",
        )
        self.preview_btn = widgets.Button(
            description="Preview",
            button_style="success",
        )
        self.reset_btn = widgets.Button(
            description="Reset",
            button_style="warning",
        )
        self.controls = widgets.VBox([
            widgets.HBox([self.feather_slider, self.feather_btn]),
            self.rx, self.ry, self.rz,
            self.tx, self.ty, self.tz,
            widgets.HBox([self.preview_btn, self.reset_btn]),
        ])

    def _bind_callbacks(self):
        for s in [self.rx, self.ry, self.rz, self.tx, self.ty, self.tz]:
            s.observe(self._on_change, names="value")
        self.preview_btn.on_click(lambda _: self._render())
        self.reset_btn.on_click(lambda _: self._reset())
        self.feather_btn.on_click(lambda _: self._apply_feather())
    
    def _apply_feather(self):
        fp = int(self.feather_slider.value)
        self.mask = clean_mask_by_depth(
            self.mask_raw,
            self.depth,
            feather_pixels=fp,
        )
        self._render()

    def _on_change(self, change):
        self.transform.update({
            "rx": self.rx.value,
            "ry": self.ry.value,
            "rz": self.rz.value,
            "tx": self.tx.value,
            "ty": self.ty.value,
            "tz": self.tz.value,
        })
    
    def _reset(self):
        self.rx.value = 0.0
        self.ry.value = 0.0
        self.rz.value = 0.0
        self.tx.value = 0.0
        self.ty.value = 0.0
        self.tz.value = 0.0
        self.transform = {
            "rx": 0.0, "ry": 0.0, "rz": 0.0,
            "tx": 0.0, "ty": 0.0, "tz": 0.0,
        }
        self._render()

    def _get_T(self):
        t = self.transform
        return euler_xyz_to_matrix(
            t["rx"], t["ry"], t["rz"],
            t["tx"], t["ty"], t["tz"],
        )
    def _render(self):
        transformed_maps = self.get_transformed_maps()
        
        vmin = np.nanmin(self.depth)
        vmax = np.nanmax(self.depth)
        
        with self.out:
            self.out.clear_output(wait=True)

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # ----- Row 1: Depth -----
            axes[0,0].imshow(self.depth, cmap="inferno", vmin=vmin, vmax=vmax)
            axes[0,0].set_title("Original Depth")
            axes[0,0].axis("off")

            axes[0,1].imshow(transformed_maps["masked_depth"], cmap="inferno", vmin=vmin, vmax=vmax)
            axes[0,1].set_title("Masked Depth")
            axes[0,1].axis("off")

            axes[0,2].imshow(transformed_maps["proj_depth"], cmap="inferno", vmin=vmin, vmax=vmax)
            axes[0,2].set_title("Transformed Masked Depth")
            axes[0,2].axis("off")

            # ----- Row 2: RGB -----
            axes[1,0].imshow(transformed_maps["img"])
            axes[1,0].set_title("Original Image")
            axes[1,0].axis("off")

            axes[1,1].imshow(transformed_maps["img_masked"])
            axes[1,1].set_title("Masked Image")
            axes[1,1].axis("off")

            axes[1,2].imshow(transformed_maps["img_proj"])
            axes[1,2].set_title("Transformed Masked Image")
            axes[1,2].axis("off")

            plt.tight_layout()
            plt.show()
    
    def get_transformation(self):
        return self._get_T()
    
    def get_transformation_params(self):
        return self.transform.copy()
    
    def get_mask(self):
        return self.mask.copy()

    def get_transformed_maps(self):
        T = self._get_T()
        rgb = to_numpy_image(self.img)

        depth_masked = np.zeros_like(self.depth)
        depth_masked[self.mask] = self.depth[self.mask]
        depth_proj = transform_depth_map(self.depth, self.intrinsics, T, self.mask)

        rgb_masked = rgb.copy()
        rgb_masked[~self.mask] = 0
        rgb_proj = transform_rgb_map(rgb, self.depth, self.intrinsics, T, self.mask)

        return {
            "orig_depth": self.depth,
            "masked_depth": depth_masked,
            "proj_depth": depth_proj,
            "img": rgb,
            "img_masked": rgb_masked,
            "img_proj": rgb_proj
        }
        

    def show(self):
        ui = widgets.VBox([
            self.controls,
            self.out,
        ])
        display(ui)