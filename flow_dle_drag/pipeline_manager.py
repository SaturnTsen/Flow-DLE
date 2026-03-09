"""Pipeline management for Flow-DLE."""

import torch
from typing import Optional, Tuple

from instaflow.pipeline_edit import RectifiedFlowStateMachine

from .config import PipelineConfig


class PipelineManager:
    """Manages Flow-DLE pipeline lifecycle."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipe = None
        self._initialized = False
    
    def initialize(self) -> "PipelineManager":
        """Initialize the pipeline."""
        if self._initialized:
            return self
        
        print(f"Loading pipeline from {self.config.model_path}...")
        
        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)
        
        # Load pipeline
        self.pipe = RectifiedFlowStateMachine.from_pretrained(
            self.config.model_path,
            torch_dtype=torch_dtype,
            device_map=self.config.device
        )
        
        # Disable safety checker if requested
        if not self.config.safety_checker_enabled:
            self.pipe.safety_checker = None
        
        # Modify UNet for feature extraction
        self.pipe.modify_unet_forward()
        
        self._initialized = True
        print("Pipeline initialized successfully")
        return self
    
    def prepare_state(
        self,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None
    ):
        """Prepare inference state for generation."""
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        return self.pipe.prepare_state(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps or self.config.num_inference_steps,
            guidance_scale=guidance_scale or self.config.guidance_scale,
            capture_all_intermediate_features=False
        )
    
    def infer_until(
        self,
        state,
        until_step: int
    ):
        """Run inference until a specific step."""
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized")
        
        return self.pipe.infer_from_state(state, until=until_step)
    
    def decode_latents(
        self,
        latent,
        disable_safety_checker: bool = True
    ) -> Tuple:
        """Decode latents to images."""
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized")
        
        return self.pipe.decode_latents(
            latent,
            disable_safety_checker=disable_safety_checker
        )
    
    def reset_state(self, state):
        """Reset inference state."""
        state.reset()
        return state
    
    def close(self):
        """Clean up resources."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self._initialized = False


def create_pipeline(config: PipelineConfig) -> PipelineManager:
    """Factory function to create and initialize a pipeline."""
    manager = PipelineManager(config)
    return manager.initialize()