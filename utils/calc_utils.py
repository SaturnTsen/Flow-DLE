# *************************************************************************
# Copyright (2026) Yiming Chen
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



from IPython.display import display, clear_output
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import torch
from PIL import Image
from typing import List, Dict
from numpy.typing import NDArray
import math
from collections import deque
import trimesh
import open3d as o3d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed+np.random.randint(0, 1000000))

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

# For 3D aware transformation
def pred_unidepth(img: Image.Image, depth_model, resolutoin_level=9) -> Dict[str, NDArray]:
    depth_model.resolution_level = resolutoin_level
    rgb = torch.from_numpy(to_numpy_image(img)).permute(2, 0, 1)
    predictions = depth_model.infer(rgb)
    preds = {
        "depth": predictions["depth"].cpu().numpy()[0, 0, :, :],
        "points": predictions["points"][0].permute(1, 2, 0).cpu().numpy(),
        "intrinsics": predictions["intrinsics"].cpu().numpy()[0]
    }
    return preds

def decimate_trimesh(mesh, ratio=0.05):
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces),
    )
    target = int(len(mesh.faces) * ratio)
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(target)

    return trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False,
    )

def clean_mask_by_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    *,
    rel_thresh: float = 0.05,
    min_area_ratio: float = 0.02,
    connectivity: int = 4,
    feather_pixels: int = 2,
):
    """
    1) depth-aware connected components
    2) keep large components
    3) inward feather (shrink) mask by N pixels
    """
    mask = mask.astype(bool)
    H, W = mask.shape

    labels = np.zeros((H, W), dtype=np.int32)
    areas = {}
    label = 0

    if connectivity == 8:
        neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    else:
        neigh = [(-1,0),(1,0),(0,-1),(0,1)]

    # --- 1. depth-aware CC ---
    for i in range(H):
        for j in range(W):
            if not mask[i, j] or labels[i, j] != 0:
                continue

            label += 1
            q = deque([(i, j)])
            labels[i, j] = label
            area = 0

            while q:
                x, y = q.popleft()
                area += 1
                d0 = depth[x, y]

                for dx, dy in neigh:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W:
                        if not mask[nx, ny] or labels[nx, ny] != 0:
                            continue
                        d1 = depth[nx, ny]
                        if not np.isfinite(d0) or not np.isfinite(d1):
                            continue
                        if abs(d1 - d0) < rel_thresh * max(d0, 1e-6):
                            labels[nx, ny] = label
                            q.append((nx, ny))

            areas[label] = area

    if not areas:
        return np.zeros_like(mask, dtype=bool)

    max_area = max(areas.values())
    keep = {
        k for k, v in areas.items()
        if v >= min_area_ratio * max_area
    }

    mask_cc = np.isin(labels, list(keep)).astype(np.uint8)

    if feather_pixels > 0:
        dist = cv2.distanceTransform(mask_cc, cv2.DIST_L2, 5)
        mask_cc = (dist >= feather_pixels)

    return mask_cc

def unidepth_to_trimesh(
    img,
    predictions,
    mask=None,
    flip_y=True,
    flip_z=True,
):
    """
    img: (H,W,3)
    mask: (H,W) bool / {0,1}
    """

    xyz = predictions["points"]      # (H,W,3)
    H, W, _ = xyz.shape

    if mask is not None:
        mask = mask.astype(bool)
    vidx = -np.ones((H, W), dtype=np.int64)

    vertices = []
    colors = []

    rgb = np.asarray(img)

    # 1. collect vertices
    idx = 0
    for i in range(H):
        for j in range(W):
            if mask is not None and not mask[i, j]:
                continue
            p = xyz[i, j]
            if not np.isfinite(p).all():
                continue

            v = p.copy()
            if flip_y:
                v[1] *= -1
            if flip_z:
                v[2] *= -1

            vidx[i, j] = idx
            vertices.append(v)
            colors.append(rgb[i, j])
            idx += 1

    vertices = np.asarray(vertices, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)

    # 2. faces
    faces = []
    for i in range(H - 1):
        for j in range(W - 1):
            ids = [
                vidx[i, j],
                vidx[i + 1, j],
                vidx[i, j + 1],
                vidx[i + 1, j + 1],
            ]
            if any(x < 0 for x in ids):
                continue
            faces.append([ids[0], ids[1], ids[2]])
            faces.append([ids[2], ids[1], ids[3]])

    faces = np.asarray(faces, dtype=np.int64)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=colors,
        process=False,
    )

    return mesh


def transform_mesh(mesh, rx, ry, rz, tx, ty, tz):
    mesh_t = mesh.copy()

    rot = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()

    verts = mesh_t.vertices
    center = verts.mean(axis=0, keepdims=True)

    verts = verts - center
    verts = verts @ rot.T
    verts = verts + center
    verts = verts + np.array([tx, ty, tz], dtype=np.float32)

    mesh_t.vertices = verts
    return mesh_t


def compare_scene(
    mesh_a,
    mesh_b,
    *,
    ratio=0.05,
    offset_scale=1e-3,
    colors=((255, 0, 0, 255), (0, 0, 255, 255)),
    show=True,
):
    """
    Compare two meshes in a single trimesh.Scene for preview.

    Args:
        mesh_a: trimesh.Trimesh (original)
        mesh_b: trimesh.Trimesh (transformed)
        ratio: face ratio kept for decimation
        offset_scale: z-offset factor to avoid z-fighting
        colors: RGBA colors for (mesh_a, mesh_b)
        show: if True, call scene.show()

    Returns:
        scene: trimesh.Scene
    """
    scene = trimesh.Scene()

    # ---- mesh A ----
    a = mesh_a.copy()
    a = decimate_trimesh(a, ratio=ratio)
    a.remove_unreferenced_vertices()
    a.visual.face_colors = colors[0] # type: ignore
    scene.add_geometry(a, node_name="mesh_a")

    # ---- mesh B ----
    b = mesh_b.copy()
    b = decimate_trimesh(b, ratio=ratio)
    b.remove_unreferenced_vertices()
    b.visual.face_colors = colors[1] # type: ignore

    # avoid z-fighting
    eps = offset_scale * a.scale
    b.apply_translation([0, 0, eps])

    scene.add_geometry(b, node_name="mesh_b")

    if show:
        scene.show(
            viewer="jupyter",
            flags={
                "lighting": False,
                "smooth": False,
            }
        )

    return scene

def transform_depth_map(
    depth: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    mask: np.ndarray,
):
    """
    Apply SE(3) transform T to a depth map, only for masked pixels.

    Args:
        depth: (H, W) depth map
        K: (3, 3) camera intrinsics
        T: (4, 4) homogeneous transform
        mask: (H, W) bool / {0,1}, transform only where mask==True

    Returns:
        depth_new: (H, W) transformed depth map (zeros elsewhere)
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth

    # 👇 only masked + valid depth
    valid = (z > 0) & mask.astype(bool)

    u = u[valid]
    v = v[valid]
    z = z[valid]

    # back-project to camera frame
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts = np.stack([x, y, z, np.ones_like(z)], axis=0)  # (4, N)

    # apply transform
    pts_t = T @ pts
    x_t, y_t, z_t = pts_t[:3]

    # project back
    u_t = (fx * x_t / z_t + cx).astype(np.int32)
    v_t = (fy * y_t / z_t + cy).astype(np.int32)

    # initialize new depth
    depth_new = np.zeros_like(depth)

    # z-buffer
    mask_proj = (
        (u_t >= 0) & (u_t < W) &
        (v_t >= 0) & (v_t < H) &
        (z_t > 0)
    )

    u_t = u_t[mask_proj]
    v_t = v_t[mask_proj]
    z_t = z_t[mask_proj]

    for uu, vv, zz in zip(u_t, v_t, z_t):
        if depth_new[vv, uu] == 0 or zz < depth_new[vv, uu]:
            depth_new[vv, uu] = zz

    return depth_new

def transform_rgb_map(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    mask: np.ndarray,
):
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth

    valid = (z > 0) & mask

    u = u[valid]
    v = v[valid]
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts = np.stack([x, y, z, np.ones_like(z)], axis=0)

    pts_t = T @ pts
    x_t, y_t, z_t = pts_t[:3]

    u_t = (fx * x_t / z_t + cx).astype(np.int32)
    v_t = (fy * y_t / z_t + cy).astype(np.int32)

    rgb_new = np.zeros_like(rgb)
    depth_buffer = np.full((H, W), np.inf)

    mask_proj = (
        (u_t >= 0) & (u_t < W) &
        (v_t >= 0) & (v_t < H) &
        (z_t > 0)
    )

    u_t = u_t[mask_proj]
    v_t = v_t[mask_proj]
    z_t = z_t[mask_proj]

    rgb_vals = rgb[valid][mask_proj]

    for uu, vv, zz, color in zip(u_t, v_t, z_t, rgb_vals):
        if zz < depth_buffer[vv, uu]:
            depth_buffer[vv, uu] = zz
            rgb_new[vv, uu] = color

    return rgb_new
    
def euler_xyz_to_matrix(
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
) -> np.ndarray:
    """
    Build 4x4 homogeneous transformation matrix.
    
    Rotation order: X -> Y -> Z (degrees)
    Right-handed coordinate system.

    Returns:
        T: (4, 4) float32
    """
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)],
    ])

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0,           1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1],
    ])

    R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]

    return T