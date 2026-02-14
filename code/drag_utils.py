# Drag utilities for InstaFlow/RectifiedFlow
# Adapted from DragDiffusion

import copy
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Union, Tuple


def point_tracking(F0, F1, handle_points, handle_points_init, args):
    """
    Track handle points by finding the closest matching feature in F1.
    """
    with torch.no_grad():
        _, _, max_r, max_c = F0.shape
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

            r1, r2 = max(0, int(pi[0]) - args.r_p), min(max_r, int(pi[0]) + args.r_p + 1)
            c1, c2 = max(0, int(pi[1]) - args.r_p), min(max_c, int(pi[1]) + args.r_p + 1)
            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) - F1_neighbor).abs().sum(dim=1)
            all_dist = all_dist.squeeze(dim=0)
            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            handle_points[i][0] = r1 + row
            handle_points[i][1] = c1 + col
        return handle_points


def check_handle_reach_target(handle_points, target_points):
    """Check if all handle points have reached their targets."""
    all_dist = list(map(lambda p, q: (p - q).norm(), handle_points, target_points))
    return (torch.tensor(all_dist) < 2.0).all()


def interpolate_feature_patch(feat, y1, y2, x1, x2):
    """
    Obtain the bilinear interpolated feature patch.
    """
    x1_floor = torch.floor(x1).long()
    x1_cell = x1_floor + 1
    dx = torch.floor(x2).long() - torch.floor(x1).long()

    y1_floor = torch.floor(y1).long()
    y1_cell = y1_floor + 1
    dy = torch.floor(y2).long() - torch.floor(y1).long()

    wa = (x1_cell.float() - x1) * (y1_cell.float() - y1)
    wb = (x1_cell.float() - x1) * (y1 - y1_floor.float())
    wc = (x1 - x1_floor.float()) * (y1_cell.float() - y1)
    wd = (x1 - x1_floor.float()) * (y1 - y1_floor.float())

    Ia = feat[:, :, y1_floor: y1_floor + dy, x1_floor: x1_floor + dx]
    Ib = feat[:, :, y1_cell: y1_cell + dy, x1_floor: x1_floor + dx]
    Ic = feat[:, :, y1_floor: y1_floor + dy, x1_cell: x1_cell + dx]
    Id = feat[:, :, y1_cell: y1_cell + dy, x1_cell: x1_cell + dx]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


# =====================================================================
# Override UNet forward to return intermediate features
# =====================================================================

def get_base_unet(unet):
    """Get the base UNet model, handling PEFT wrapper if present."""
    # Check if this is a PEFT model
    if hasattr(unet, 'base_model'):
        # PEFT wraps the model in base_model.model
        return unet.base_model.model
    return unet


def override_unet_forward(unet):
    """
    Override UNet forward to return intermediate features from up blocks.
    This is the same approach as DragDiffusion.
    Handles both regular UNet and PEFT-wrapped UNet.
    """
    # Get the actual UNet for accessing blocks
    base_unet = get_base_unet(unet)

    def forward(
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ):
        default_overall_up_factor = 2 ** base_unet.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if base_unet.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])

        t_emb = base_unet.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = base_unet.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if base_unet.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if base_unet.config.class_embed_type == "timestep":
                class_labels = base_unet.time_proj(class_labels)
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = base_unet.class_embedding(class_labels).to(dtype=sample.dtype)

            if base_unet.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        emb = emb + aug_emb if aug_emb is not None else emb

        if hasattr(base_unet, 'time_embed_act') and base_unet.time_embed_act is not None:
            emb = base_unet.time_embed_act(emb)

        if hasattr(base_unet, 'encoder_hid_proj') and base_unet.encoder_hid_proj is not None:
            if base_unet.config.encoder_hid_dim_type == "text_proj":
                encoder_hidden_states = base_unet.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        sample = base_unet.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in base_unet.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if base_unet.mid_block is not None:
            if hasattr(base_unet.mid_block, "has_cross_attention") and base_unet.mid_block.has_cross_attention:
                sample = base_unet.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = base_unet.mid_block(sample, emb)

        all_intermediate_features = [sample]

        # 5. up
        for i, upsample_block in enumerate(base_unet.up_blocks):
            is_final_block = i == len(base_unet.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
            all_intermediate_features.append(sample)

        # 6. post-process
        if base_unet.conv_norm_out:
            sample = base_unet.conv_norm_out(sample)
            sample = base_unet.conv_act(sample)
        sample = base_unet.conv_out(sample)

        if return_intermediates:
            return sample, all_intermediate_features
        else:
            return sample

    return forward


# =====================================================================
# Drag update function for RectifiedFlow
# =====================================================================

def drag_rf_update(
    pipe,
    latent: torch.Tensor,
    t: float,                          # current timestep (e.g., 800.0)
    prompt_embeds: torch.Tensor,       # text embeddings
    handle_points: List[torch.Tensor],
    target_points: List[torch.Tensor],
    mask: torch.Tensor,
    args,
    dt: float = 0.02,                  # dt for the Euler step (1/n_inference_steps)
):
    """
    Drag update for RectifiedFlow pipeline.
    
    Unlike DragDiffusion which uses DDIM, RectifiedFlow uses Euler ODE solver:
        z_next = z + dt * v_pred
    
    Args:
        pipe: The RectifiedFlow pipeline
        latent: Current latent code [B, C, H, W]
        t: Current timestep value
        prompt_embeds: Text embeddings for the UNet
        handle_points: List of handle point tensors [row, col] in feature resolution
        target_points: List of target point tensors [row, col] in feature resolution
        mask: Binary mask indicating regions to drag [1, 1, H, W]
        args: Namespace containing drag parameters
        dt: Time step size for Euler solver
    
    Returns:
        Updated latent code
    """
    assert len(handle_points) == len(target_points), \
        "number of handle points must equal target points"

    device = latent.device
    dtype = latent.dtype

    # Extract UNet features at initial state (for reference)
    with torch.no_grad():
        vec_t = torch.ones((latent.shape[0],), device=device) * t
        # Get UNet output with intermediate features
        v_pred_init, F0 = forward_unet_features(
            pipe, latent, vec_t, prompt_embeds,
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w
        )
        # Compute x_prev for background regularization
        # For RectifiedFlow: z_next = z + dt * v_pred
        x_prev_0 = latent + dt * v_pred_init

    # Prepare optimizable latent and optimizer
    latent = latent.detach().clone()
    latent.requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=args.lr)

    # Prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (latent.shape[2], latent.shape[3]), mode='nearest')
    using_mask = interp_mask.sum() != 0.0

    # Note: We don't use GradScaler because BFloat16 doesn't need it
    # (BFloat16 has the same exponent range as float32)

    for step_idx in range(args.n_pix_step):
        vec_t = torch.ones((latent.shape[0],), device=device) * t
        v_pred, F1 = forward_unet_features(
            pipe, latent, vec_t, prompt_embeds,
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w
        )
        x_prev_updated = latent + dt * v_pred

        # Point tracking (except for first step)
        if step_idx != 0:
            handle_points = point_tracking(F0, F1, handle_points, handle_points_init, args)

        # Check if all handles reached targets
        if check_handle_reach_target(handle_points, target_points):
            break

        # Compute motion supervision loss
        loss = 0.0
        _, _, max_r, max_c = F0.shape
        for i in range(len(handle_points)):
            pi, ti = handle_points[i], target_points[i]
            # Skip if already close
            if (ti - pi).norm() < 2.0:
                continue

            # Direction vector
            di = (ti - pi) / (ti - pi).norm()

            # Motion supervision with boundary protection
            r1, r2 = max(0, int(pi[0]) - args.r_m), min(max_r, int(pi[0]) + args.r_m + 1)
            c1, c2 = max(0, int(pi[1]) - args.r_m), min(max_c, int(pi[1]) + args.r_m + 1)
            f0_patch = F1[:, :, r1:r2, c1:c2].detach()
            f1_patch = interpolate_feature_patch(F1, r1 + di[0], r2 + di[0], c1 + di[1], c2 + di[1])

            loss += ((2 * args.r_m + 1) ** 2) * F.l1_loss(f0_patch, f1_patch)

        # Background regularization
        if using_mask:
            loss += args.lam * ((x_prev_updated - x_prev_0) * (1.0 - interp_mask)).abs().sum()

        # Standard backward pass (no scaling needed for bfloat16)
        if isinstance(loss, float) and loss == 0.0:
            continue  # Skip if no valid loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return latent.detach()


def forward_unet_features(
    pipe,
    z: torch.Tensor,
    t: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    layer_idx: List[int] = [3],
    interp_res_h: int = 256,
    interp_res_w: int = 256,
):
    """
    Run UNet forward and extract intermediate features.
    
    Returns:
        v_pred: velocity prediction from UNet
        features: concatenated features from specified layers
    """
    # Run UNet with intermediate feature extraction
    v_pred, all_intermediate_features = pipe.unet(
        z,
        t,
        encoder_hidden_states=encoder_hidden_states,
        return_intermediates=True
    )
    
    # Extract and interpolate features from specified layers
    all_return_features = []
    for idx in layer_idx:
        feat = all_intermediate_features[idx]
        feat = F.interpolate(feat, (interp_res_h, interp_res_w), mode='bilinear')
        all_return_features.append(feat)
    
    return_features = torch.cat(all_return_features, dim=1)
    return v_pred, return_features
