"""Utility functions for Flow-DLE."""

import os
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image

from .config import DragConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flow_dle.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 0xdeadbeef):
    """Set random seeds for reproducibility."""
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dragbench_sample(
    sample_path: str
) -> Tuple[np.ndarray, str, np.ndarray, list]:
    """
    Load a DragBench sample.
    
    Args:
        sample_path: Path to sample directory
    
    Returns:
        Tuple of (image, prompt, mask, points)
    """
    try:
        # Load image
        image_path = os.path.join(sample_path, 'original_image.png')
        source_image = Image.open(image_path)
        source_image = np.array(source_image)
        
        # Load metadata
        meta_path = os.path.join(sample_path, 'meta_data.pkl')
        with open(meta_path, 'rb') as f:
            meta_data = pickle.load(f)
        
        prompt = meta_data['prompt']
        mask = meta_data['mask']
        points = meta_data['points']
        
        logger.debug(f"Loaded sample: {source_image.shape}, {len(points)} points")
        logger.debug(f"Mask sum: {mask.sum()}")
        logger.debug(f"Prompt: {prompt}")

        return source_image, prompt, mask, points
        
    except Exception as e:
        logger.error(f"Failed to load sample {sample_path}: {e}")
        raise


def save_results(
    output_dir: str,
    category: str,
    sample_name: str,
    drag_output,
    save_intermediates: bool = False
):
    """Save drag editing results."""
    save_dir = Path(output_dir) / category / sample_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final image
    final_img_path = save_dir / 'dragged_image.png'
    drag_output.final_image.save(final_img_path)
    logger.info(f"Saved final image: {final_img_path}")
    
    # Save intermediate visualizations if requested
    if save_intermediates and hasattr(drag_output, 'optim_steps'):
        for i, step_data in enumerate(drag_output.optim_steps):
            if hasattr(step_data, 'latent'):
                img = drag_output.pipe.decode_latents(
                    step_data.latent,
                    disable_safety_checker=True
                )[0][0]
                img.save(save_dir / f'intermediate_{i:03d}.png')
    
    # Save metadata
    metadata = {
        'drag_step': drag_output.drag_step,
        'end_step': drag_output.end_step,
        'converged': drag_output.converged,
        'total_steps': drag_output.total_steps,
    }
    with open(save_dir / 'metadata.json', 'w') as f:
        import json
        json.dump(metadata, f, indent=2)


def setup_result_directory(
    result_dir: Optional[str],
    config: DragConfig
) -> Path:
    """Create result directory with descriptive name from DragConfig.
    
    Args:
        result_dir: Optional custom directory path
        config: DragConfig instance with all hyperparameters
    
    Returns:
        Path object for the result directory
    """
    if result_dir:
        result_path = Path(result_dir)
    else:
        # Format unet_feature_idx as hyphen-separated values (e.g., "3" or "3-5-7")
        unet_str = "-".join(str(idx) for idx in config.unet_feature_idx)
        
        # Format floats with consistent precision, avoiding dots in filenames
        lr_str = f"{config.lr:.2f}".replace(".", "p")  # e.g., "0p01"
        lam_str = f"{config.lam:.2f}".replace(".", "p")  # e.g., "0p50"
        
        result_name = (
            f"flow_dle_res_"
            f"drag{config.drag_step}_"
            f"npix{config.n_pix_step}_"
            f"lr{lr_str}_"
            f"lam{lam_str}_"
            f"unet{unet_str}"
        )
        result_path = Path(result_name)
    
    result_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Result directory: {result_path.absolute()}")
    return result_path