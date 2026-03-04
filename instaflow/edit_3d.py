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
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Callable
from .pipeline_edit import InferenceState
from torch import Tensor
import numpy as np
LayerFeature = Tuple[str, Tensor]



@dataclass
class Edit3DConfig:
    start_step: int = 20
    end_step: int = 25
    optim_steps_per_inference: int = 10
    lr: float = 0.1
    lamfg: float = 0.01
    lambg: float = 0.001

def _prepare_3d_inputs(
    *,
    depth_np: np.ndarray,          # [Himg, Wimg]
    mask_np: np.ndarray,           # [Himg, Wimg]
    K_np: np.ndarray,              # [3, 3]
    T_np: np.ndarray,              # [4, 4]
    feature_hw: Tuple[int, int],   # (Hf, Wf)
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Returns:
        depth: [1, 1, Hf, Wf] float32
        mask : [1, 1, Hf, Wf] float32 in {0,1}
        K    : [3, 3] float32, scaled to (Hf, Wf)
        T    : [4, 4] float32
    """
    if depth_np.ndim != 2:
        raise ValueError(f"depth_np must be HxW, got {depth_np.shape}")
    if mask_np.ndim != 2:
        raise ValueError(f"mask_np must be HxW, got {mask_np.shape}")
    if K_np.shape != (3, 3):
        raise ValueError(f"K_np must be 3x3, got {K_np.shape}")
    if T_np.shape != (4, 4):
        raise ValueError(f"T_np must be 4x4, got {T_np.shape}")

    Himg, Wimg = depth_np.shape
    Hf, Wf = feature_hw

    # depth -> [1,1,Hf,Wf]
    depth = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)
    depth = depth[None, None]
    depth = F.interpolate(depth, size=(Hf, Wf), mode="bilinear", align_corners=False)

    # mask -> [1,1,Hf,Wf] {0,1}
    mask = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
    mask = mask[None, None]
    mask = F.interpolate(mask, size=(Hf, Wf), mode="bilinear", align_corners=False)
    mask = (mask > 0.5).to(torch.float32)

    # K scaled to feature resolution
    K = torch.from_numpy(K_np).to(device=device, dtype=torch.float32).clone()
    sx = float(Wf) / float(Wimg)
    sy = float(Hf) / float(Himg)
    K[0, 0] *= sx  # fx
    K[1, 1] *= sy  # fy
    K[0, 2] *= sx  # cx
    K[1, 2] *= sy  # cy

    # T
    T = torch.from_numpy(T_np).to(device=device, dtype=torch.float32)

    return depth, mask, K, T



@torch.no_grad()
def _forward_splat_feature_mask_and_corr(
    *,
    feature: Tensor,   # [1, C, H, W]  reference feature
    depth: Tensor,     # [1, 1, H, W]
    mask: Tensor,      # [1, 1, H, W]  source FG mask
    K: Tensor,         # [3, 3]
    T: Tensor,         # [4, 4]
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """
    Forward splat reference feature using depth + object transform.
    Also returns sparse pixel correspondences (src -> tgt) after z-buffer visibility.

    Returns:
        feat_warp: [1, C, H, W]
        mask_warp: [1, 1, H, W]  (1 where a visible FG point lands)
        corr: dict of int64 tensors (all shape [N]):
            original_x, original_y, transformed_x, transformed_y
    """
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError(f"feature must be [1,C,H,W], got {tuple(feature.shape)}")
    if depth.shape[0] != 1 or depth.shape[1] != 1:
        raise ValueError(f"depth must be [1,1,H,W], got {tuple(depth.shape)}")
    if mask.shape[0] != 1 or mask.shape[1] != 1:
        raise ValueError(f"mask must be [1,1,H,W], got {tuple(mask.shape)}")

    _, C, H, W = feature.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    v, u = torch.meshgrid(
        torch.arange(H, device=feature.device),
        torch.arange(W, device=feature.device),
        indexing="ij",
    )
    u = u.to(torch.float32)
    v = v.to(torch.float32)

    z = depth[0, 0]  # [H,W]
    m = mask[0, 0]   # [H,W]
    valid = (z > 0) & (m > 0.5)

    empty_corr = {
        "original_x": torch.empty((0,), device=feature.device, dtype=torch.int64),
        "original_y": torch.empty((0,), device=feature.device, dtype=torch.int64),
        "transformed_x": torch.empty((0,), device=feature.device, dtype=torch.int64),
        "transformed_y": torch.empty((0,), device=feature.device, dtype=torch.int64),
    }

    if valid.sum().item() == 0:
        feat_warp = torch.zeros_like(feature)
        mask_warp = torch.zeros((1, 1, H, W), device=feature.device, dtype=torch.float32)
        return feat_warp, mask_warp, empty_corr

    # source linear indices for valid pixels
    src_lin_all = torch.nonzero(valid.flatten(), as_tuple=False).squeeze(1)  # [Nvalid]

    u0 = u[valid]
    v0 = v[valid]
    z0 = z[valid]

    x0 = (u0 - cx) * z0 / fx
    y0 = (v0 - cy) * z0 / fy

    ones = torch.ones_like(z0)
    pts = torch.stack([x0, y0, z0, ones], dim=0)  # [4, Nvalid]

    pts_t = (T @ pts)
    x1, y1, z1 = pts_t[0], pts_t[1], pts_t[2]

    u1 = fx * x1 / z1 + cx
    v1 = fy * y1 / z1 + cy

    ui = torch.round(u1).to(torch.int64)
    vi = torch.round(v1).to(torch.int64)

    inb = (z1 > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if inb.sum().item() == 0:
        feat_warp = torch.zeros_like(feature)
        mask_warp = torch.zeros((1, 1, H, W), device=feature.device, dtype=torch.float32)
        return feat_warp, mask_warp, empty_corr

    ui_inb = ui[inb]
    vi_inb = vi[inb]
    z1_inb = z1[inb].to(torch.float32)
    src_lin_inb = src_lin_all[inb]  # [Ninb]

    # source feature vectors at valid pixels
    feat_src = feature[0, :, valid]  # [C, Nvalid]
    feat_src = feat_src[:, inb]      # [C, Ninb]

    # z-buffer per target pixel
    idx = vi_inb * W + ui_inb  # [Ninb]
    min_z = torch.full((H * W,), float("inf"), device=feature.device, dtype=torch.float32)
    min_z.scatter_reduce_(0, idx, z1_inb, reduce="amin", include_self=True)

    keep = (z1_inb <= (min_z[idx] + eps))
    if keep.sum().item() == 0:
        feat_warp = torch.zeros_like(feature)
        mask_warp = torch.zeros((1, 1, H, W), device=feature.device, dtype=torch.float32)
        return feat_warp, mask_warp, empty_corr

    idx_k = idx[keep]          # [K]
    feat_k = feat_src[:, keep] # [C, K]

    feat_warp = torch.zeros((1, C, H * W), device=feature.device, dtype=feature.dtype)
    feat_warp[0].scatter_(1, idx_k[None, :].expand(C, -1), feat_k)
    feat_warp = feat_warp.view(1, C, H, W)

    mask_warp = torch.zeros((1, 1, H * W), device=feature.device, dtype=torch.float32)
    mask_warp[0, 0, idx_k] = 1.0
    mask_warp = mask_warp.view(1, 1, H, W)

    # correspondences (src pixel -> tgt pixel)
    src_lin_k = src_lin_inb[keep]  # [K]
    src_y = (src_lin_k // W).to(torch.int64)
    src_x = (src_lin_k % W).to(torch.int64)
    tgt_y = (idx_k // W).to(torch.int64)
    tgt_x = (idx_k % W).to(torch.int64)

    corr = {
        "original_x": src_x,
        "original_y": src_y,
        "transformed_x": tgt_x,
        "transformed_y": tgt_y,
    }
    return feat_warp, mask_warp, corr


def local_average_feat_l1_loss_torch(
    feat_map_1: Tensor,  # [C,H,W]
    feat_map_2: Tensor,  # [C,H,W]
    x1: Tensor, y1: Tensor,
    x2: Tensor, y2: Tensor,
    patch_size: int = 1,
) -> Tensor:
    """
    Reference-like sparse correspondence loss using local weighted averages.

    Equivalent in spirit to the provided reference:
      weights -> AvgPool -> normalize -> compare at correspondence points.
    """
    if x1.numel() == 0:
        return torch.zeros((), device=feat_map_1.device, dtype=torch.float32)

    assert feat_map_1.ndim == 3 and feat_map_2.ndim == 3
    C, H, W = feat_map_1.shape
    device = feat_map_1.device
    dtype = feat_map_1.dtype

    # weights: [H,W]
    w1 = torch.zeros((H, W), device=device, dtype=dtype)
    w2 = torch.zeros((H, W), device=device, dtype=dtype)

    # allow duplicates (scatter_add is fine)
    lin1 = (y1.to(torch.int64) * W + x1.to(torch.int64))
    lin2 = (y2.to(torch.int64) * W + x2.to(torch.int64))
    ones = torch.ones_like(lin1, dtype=dtype, device=device)
    w1.view(-1).scatter_add_(0, lin1, ones)
    w2.view(-1).scatter_add_(0, lin2, ones)

    pooling = torch.nn.AvgPool2d(patch_size, stride=1, padding=patch_size // 2)

    f1 = pooling((w1[None, None] * feat_map_1[None]))  # [1,C,H,W]
    f2 = pooling((w2[None, None] * feat_map_2[None]))

    ww1 = pooling(w1[None, None])  # [1,1,H,W]
    ww2 = pooling(w2[None, None])

    EPS = 1e-10
    f1 = f1 / (ww1 + EPS)
    f2 = f2 / (ww2 + EPS)

    diff = (f1[0, :, y1, x1] - f2[0, :, y2, x2]).abs()  # [C,N]
    return diff.mean(dim=0).mean()  # mean over C then over N


def _morph_close_open(mask01: Tensor, k_close: int = 3, k_open: int = 3) -> Tensor:
    """
    Torch approximation of cv2 morphology close+open for binary masks.

    mask01: [1,1,H,W] float in {0,1}
    """
    def dilate(x: Tensor, k: int) -> Tensor:
        return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    def erode(x: Tensor, k: int) -> Tensor:
        return 1.0 - F.max_pool2d(1.0 - x, kernel_size=k, stride=1, padding=k // 2)

    x = mask01
    x = erode(dilate(x, k_close), k_close)  # close
    x = dilate(erode(x, k_open), k_open)    # open
    return (x > 0.5).to(mask01.dtype)


def edit_3d(
    pipe,
    state_fg_ref,  # InferenceState
    *,
    depth_np: np.ndarray,   # [Himg, Wimg]
    K_np: np.ndarray,       # [3,3]
    T_np: np.ndarray,       # [4,4]
    mask_np: np.ndarray,    # [Himg, Wimg]
    config,                 # Edit3DConfig
) -> Callable[[int, Tensor, float, Tensor, List[Tuple[str, Tensor]], Tensor], Tensor]:
    """
    Reference-like 3D edit:
      - Build sparse correspondences from forward splat + z-buffer visibility
      - Foreground loss: correspondence-driven local-average L1 (patch-based)
      - Background loss: keep features close to reference outside union masks
    """

    loss_history: Dict[int, List[float]] = {}

    def _get_ref_features_for_step(step_i: int) -> Optional[List[Tuple[str, Tensor]]]:
        if state_fg_ref.intermediate_latents is None:
            raise AssertionError("Intermediate latents must not be None for reference latent")
        for item in state_fg_ref.intermediate_latents:
            if item.get("step", None) == step_i and "features" in item:
                return item["features"]
        return None

    def feature_loss_callback(
        i: int,
        t: Tensor,
        dt: float,
        latent: Tensor,                  # [B, Cl, Hl, Wl]
        features: List[Tuple[str, Tensor]],  # not used; we re-hook after updates
        v_pred: Tensor,                  # [B, Cl, Hl, Wl]
    ) -> Tensor:
        if i not in loss_history:
            loss_history[i] = []

        # outside window: pure ODE step
        if not (config.start_step <= i < config.end_step):
            return latent + dt * v_pred

        ref_features = _get_ref_features_for_step(i)
        if ref_features is None:
            raise AssertionError("Missing reference features for this step")

        # normal ODE step first
        latent_next = latent + dt * v_pred

        torch.set_grad_enabled(True)
        latent_next = latent_next.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([latent_next], lr=config.lr)

        # cache geometry inputs at some base feature resolution
        prepared: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None

        for _ in range(config.optim_steps_per_inference):
            optimizer.zero_grad()

            # ---- re-run UNet to get current features ----
            current_features: List[Tuple[str, Tensor]] = []
            name_map = {m: n for n, m in pipe.unet.named_modules()}

            def hook(mod, inp, out):
                current_features.append((name_map.get(mod, ""), out))

            handles = [pipe.unet.mid_block.register_forward_hook(hook)]
            handles += [b.register_forward_hook(hook) for b in pipe.unet.up_blocks]

            do_cfg = state_fg_ref.do_cfg
            latent_input = torch.cat([latent_next] * 2, dim=0) if do_cfg else latent_next
            vec_t = torch.ones((latent_input.shape[0],), device=latent_next.device, dtype=latent_next.dtype) * t

            _ = pipe.unet(latent_input, vec_t, encoder_hidden_states=state_fg_ref.prompt_embeds)

            for h in handles:
                h.remove()

            # ---- prepare geometry inputs once, using a representative feature resolution ----
            if prepared is None:
                if len(current_features) == 0:
                    raise RuntimeError("No features captured by hooks. Check hook points.")
                _, feat0 = current_features[0]
                feat0_pos = feat0.chunk(2)[1] if do_cfg and feat0.shape[0] == 2 * latent_next.shape[0] else feat0
                Hf, Wf = int(feat0_pos.shape[-2]), int(feat0_pos.shape[-1])

                prepared = _prepare_3d_inputs(
                    depth_np=depth_np,
                    mask_np=mask_np,
                    K_np=K_np,
                    T_np=T_np,
                    feature_hw=(Hf, Wf),
                    device=latent_next.device,
                )

            depth_f, mask_f, K_f, T_f = prepared  # depth/mask/K/T at base feature res

            # ---- loss over matched layers ----
            loss = torch.zeros((), device=latent_next.device, dtype=torch.float32)

            ref_dict: Dict[str, Tensor] = {n: f for (n, f) in ref_features}

            for (name_cur, feat_cur_full) in current_features:
                if name_cur not in ref_dict:
                    continue

                feat_ref_full = ref_dict[name_cur]

                # CFG: positive branch only
                feat_cur = feat_cur_full
                if do_cfg and feat_cur_full.shape[0] == 2 * latent_next.shape[0]:
                    feat_cur = feat_cur_full.chunk(2)[1]

                feat_ref = feat_ref_full
                if do_cfg and feat_ref_full.shape[0] == 2 * latent_next.shape[0]:
                    feat_ref = feat_ref_full.chunk(2)[1]

                # match geometry inputs to this layer resolution
                Hc, Wc = int(feat_cur.shape[-2]), int(feat_cur.shape[-1])
                if (Hc, Wc) != (int(depth_f.shape[-2]), int(depth_f.shape[-1])):
                    depth_l = F.interpolate(depth_f, size=(Hc, Wc), mode="bilinear", align_corners=False)
                    mask_l = F.interpolate(mask_f, size=(Hc, Wc), mode="bilinear", align_corners=False)
                    mask_l = (mask_l > 0.5).to(torch.float32)

                    # scale K for this layer resolution
                    Himg, Wimg = depth_np.shape
                    sx = float(Wc) / float(Wimg)
                    sy = float(Hc) / float(Himg)
                    K_l = torch.from_numpy(K_np).to(device=latent_next.device, dtype=torch.float32).clone()
                    K_l[0, 0] *= sx
                    K_l[1, 1] *= sy
                    K_l[0, 2] *= sx
                    K_l[1, 2] *= sy
                    T_l = T_f
                else:
                    depth_l, mask_l, K_l, T_l = depth_f, mask_f, K_f, T_f

                # ---- forward splat + correspondences (ref-only) ----
                feat_warp, mask_warp, corr = _forward_splat_feature_mask_and_corr(
                    feature=feat_ref.detach(),
                    depth=depth_l.detach(),
                    mask=mask_l.detach(),
                    K=K_l.detach(),
                    T=T_l.detach(),
                )

                # optional mask cleaning (closer to reference's morphology)
                clean_mask = getattr(config, "clean_mask", True)
                if clean_mask:
                    k_close = int(getattr(config, "mask_close_k", 3))
                    k_open = int(getattr(config, "mask_open_k", 3))
                    mask_warp_use = _morph_close_open(mask_warp, k_close=k_close, k_open=k_open)
                else:
                    mask_warp_use = (mask_warp > 0.5).to(mask_warp.dtype)

                # filter correspondences by cleaned target mask (reference does this)
                if corr["original_x"].numel() > 0:
                    tx = corr["transformed_x"]
                    ty = corr["transformed_y"]
                    keep_corr = (mask_warp_use[0, 0, ty, tx] > 0.5)
                    if keep_corr.numel() == 0:
                        # nothing left
                        corr = {
                            "original_x": tx[:0],
                            "original_y": ty[:0],
                            "transformed_x": tx[:0],
                            "transformed_y": ty[:0],
                        }
                    else:
                        for k in list(corr.keys()):
                            corr[k] = corr[k][keep_corr]

                # background region: outside union of old mask and warped mask
                mask_union = torch.clamp(mask_l + mask_warp_use, 0.0, 1.0)
                mask_bg = 1.0 - mask_union

                # ---- foreground loss: correspondence local-average (reference-like) ----
                patch_size = int(getattr(config, "patch_size", 1))
                loss_fg = local_average_feat_l1_loss_torch(
                    feat_map_1=feat_ref.detach()[0],   # [C,H,W]
                    feat_map_2=feat_cur[0],            # [C,H,W]
                    x1=corr["original_x"],
                    y1=corr["original_y"],
                    x2=corr["transformed_x"],
                    y2=corr["transformed_y"],
                    patch_size=patch_size,
                )

                # ---- background loss: keep current feature close to reference in BG ----
                diff_bg = torch.abs(feat_cur - feat_ref.detach())
                loss_bg = (diff_bg * mask_bg).mean()

                loss = loss + (config.lamfg * loss_fg + config.lambg * loss_bg)

            loss_history[i].append(float(loss.detach().cpu()))
            loss.backward()
            optimizer.step()

        torch.set_grad_enabled(False)
        return latent_next.detach()

    return feature_loss_callback