"""Core drag editing operations for Flow-DLE."""

import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple
from types import SimpleNamespace

from instaflow.drag_utils import DragOutput, drag_rf_update

from .config import DragConfig
from .pipeline_manager import PipelineManager


def convert_points(
    points: List[Tuple[float, float]],
    full_h: int,
    full_w: int,
    sup_res_h: int,
    sup_res_w: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Convert pixel coordinates to feature map coordinates."""
    handle_points, target_points = [], []
    
    for idx, point in enumerate(points):
        # Convert (x, y) to (row, col) = (y, x) and normalize
        cur_point = torch.tensor([
            point[1] / full_h * sup_res_h,
            point[0] / full_w * sup_res_w
        ])
        cur_point = torch.round(cur_point)
        
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    
    return handle_points, target_points


def run_rf_drag(
    pipe: PipelineManager,
    state,
    source_image: np.ndarray,
    mask: np.ndarray,
    points: List[Tuple[float, float]],
    config: DragConfig
) -> DragOutput:
    """
    Run drag editing on a RectifiedFlow generation.
    
    Args:
        pipe: Initialized PipelineManager
        state: Current inference state
        source_image: Original image
        mask: Binary mask (H x W), values in [0, 255]
        points: List of (x, y) points alternating handle/target
        config: Drag configuration
    
    Returns:
        DragOutput containing results
    """
    # Create args namespace for drag_rf_update
    args = SimpleNamespace(
        n_pix_step=config.n_pix_step,
        lr=config.lr,
        lam=config.lam,
        unet_feature_idx=config.unet_feature_idx,
        sup_res_h=config.sup_res_h,
        sup_res_w=config.sup_res_w,
        r_m=config.r_m,
        r_p=config.r_p,
    )
    
    # Reset state and run until drag_step
    state = pipe.reset_state(state)
    # FIXME: this starts a brand new generation with the prompt in the metadata.
    # Instead we want to configure the pipe latents with the current image.
    state = pipe.infer_until(state, config.drag_step)
    
    # Decode for visualization
    drag_code_vis = pipe.decode_latents(
        state.latent, 
        disable_safety_checker=True
    )[0][0]
    
    # Prepare mask
    mask_tensor = torch.from_numpy(mask).float() / 255.0
    mask_tensor[mask_tensor > 0.0] = 1.0
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(state.device)
    mask_tensor = F.interpolate(
        mask_tensor, 
        (config.sup_res_h, config.sup_res_w), 
        mode='nearest'
    )
    
    # Get current image dimensions
    full_h = source_image.shape[0] #state.latent.shape[2] * 8
    full_w = source_image.shape[1] #state.latent.shape[3] * 8
    
    # Convert points to feature coordinates
    handle_points, target_points = convert_points(
        points, full_h, full_w, 
        config.sup_res_h, config.sup_res_w
    )
    
    # Get current timestep
    t = state.timesteps[state.i - 1] if state.i > 0 else state.timesteps[0]
    
    # Get text embeddings
    if state.do_cfg:
        text_embeds = state.prompt_embeds.chunk(2)[1]
    else:
        text_embeds = state.prompt_embeds
    
    # Apply drag update
    print(f"Applying drag at step {state.i} (t={t:.1f})")
    drag_output: DragOutput = drag_rf_update(
        pipe=pipe.pipe,  # Access underlying pipe
        latent=state.latent,
        t=t,
        prompt_embeds=text_embeds,
        handle_points=handle_points,
        target_points=target_points,
        mask=mask_tensor,
        args=args,
        dt=state.dt,
        show_optim_process=config.show_optim_process,
        vis_interval=config.vis_interval,
    )
    
    print(f"Drag completed: {drag_output.total_steps} steps, converged={drag_output.converged}")
    
    # Update state with dragged latent
    state.latent = drag_output.latent
    
    # Continue inference to end
    state = pipe.infer_until(state, config.drag_step)
    drag_code_vis_after = pipe.decode_latents(
        state.latent, 
        disable_safety_checker=True
    )[0][0]
    
    state = pipe.infer_until(state, config.end_step)
    
    # Decode final image
    final_image = pipe.decode_latents(
        state.latent, 
        disable_safety_checker=True
    )[0][0]
    
    # Add extra image info to output
    drag_output.drag_code_vis = drag_code_vis
    drag_output.drag_code_vis_after = drag_code_vis_after
    drag_output.final_image = final_image
    drag_output.state = state
    drag_output.drag_step = config.drag_step
    drag_output.end_step = config.end_step
    
    return drag_output