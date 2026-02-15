# LoRA training utilities for InstaFlow/RectifiedFlow using PEFT
# Modern approach using the peft package instead of manual LoRA injection

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from torchvision import transforms
from accelerate import Accelerator
from accelerate.utils import set_seed
from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer, PretrainedConfig
from diffusers import AutoencoderKL
from peft import LoraConfig, get_peft_model


@dataclass
class RFLoraConfig:
    """Configuration for LoRA training on RectifiedFlow."""
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple = ("to_q", "to_k", "to_v", "to_out.0")
    lora_steps: int = 100
    lora_lr: float = 1e-4
    lora_batch_size: int = 4
    lora_save_dir: str = "lora_tmp/"
    seed: int = 42


def train_rf_lora(
    pipe,
    image: Image.Image,
    prompt: str,
    config: RFLoraConfig,
    progress=None,
):
    """
    Train LoRA on a single image for RectifiedFlow.
    
    Args:
        pipe: RectifiedFlowStateMachine pipeline
        image: PIL Image to train on
        prompt: Text prompt describing the image
        config: RFLoraConfig with training parameters
        progress: Optional tqdm-like progress tracker
    
    Returns:
        The pipeline with LoRA weights loaded
    """
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='bf16'  # Use bf16 for RectifiedFlow
    )
    set_seed(config.seed)
    
    device = pipe._execution_device
    dtype = pipe.unet.dtype
    
    # Unload any existing LoRA before training a new one
    if hasattr(pipe.unet, 'peft_config') or hasattr(pipe.unet, 'base_model'):
        print("Unloading existing LoRA weights...")
        # Get the base model without LoRA
        if hasattr(pipe.unet, 'merge_and_unload'):
            # This merges LoRA weights and returns base model
            pipe.unet = pipe.unet.merge_and_unload()
        elif hasattr(pipe.unet, 'base_model'):
            # Access the underlying model
            pipe.unet = pipe.unet.base_model.model
        print("Existing LoRA unloaded.")
    
    # Freeze VAE and text encoder
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    
    # Configure LoRA using PEFT
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(pipe.unet, lora_config)
    unet.print_trainable_parameters()
    
    # Optimizer
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lora_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    # Prepare with accelerator
    unet, optimizer = accelerator.prepare(unet, optimizer)
    
    # Encode text prompt
    with torch.no_grad():
        text_input = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embedding = pipe.text_encoder(text_input.input_ids.to(device))[0]
        text_embedding = text_embedding.repeat(config.lora_batch_size, 1, 1)
    
    # Image transforms
    image_transforms_pil = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
    ])
    image_transforms_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Prepare image
    if isinstance(image, Image.Image):
        train_image = image_transforms_pil(image)
    else:
        # numpy array
        train_image = image_transforms_pil(Image.fromarray(image))
    
    # Training loop
    iterator = range(config.lora_steps)
    if progress is not None:
        iterator = progress.tqdm(iterator, desc="Training LoRA")
    else:
        iterator = tqdm(iterator, desc="Training LoRA")
    
    for step in iterator:
        unet.train()
        
        # Random augmentation
        image_batch = []
        for _ in range(config.lora_batch_size):
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                img = transforms.functional.hflip(train_image)
            else:
                img = train_image
            image_batch.append(image_transforms_tensor(img))
        
        pixel_values = torch.stack(image_batch).to(device, dtype=dtype)
        
        # Encode to latent
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
        
        # Sample random timesteps for RectifiedFlow
        # In RF, we use continuous time in [0, 1], but the model expects [0, 1000]
        t = torch.rand(config.lora_batch_size, device=device)
        timesteps = t * 1000.0
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # For RectifiedFlow: x_t = (1 - t) * x_0 + t * noise
        # which means: x_t = x_0 + t * (noise - x_0)
        t_expanded = t.view(-1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        
        # Predict velocity (target is: noise - x_0)
        target = noise - latents
        
        # Forward pass
        v_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embedding,
        )
        if hasattr(v_pred, 'sample'):
            v_pred = v_pred.sample
        
        # Loss
        loss = F.mse_loss(v_pred.float(), target.float(), reduction="mean")
        
        # Backward
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        if step % 20 == 0:
            print(f"Step {step}/{config.lora_steps}, Loss: {loss.item():.6f}")
    
    # Save LoRA weights
    unet = accelerator.unwrap_model(unet)
    unet.save_pretrained(config.lora_save_dir)
    print(f"LoRA weights saved to {config.lora_save_dir}")
    
    # Update pipeline's unet
    pipe.unet = unet
    
    # Re-apply the forward override for feature extraction (needed for drag)
    if hasattr(pipe, 'modify_unet_forward'):
        pipe.modify_unet_forward()
    
    return pipe


def load_rf_lora(pipe, lora_path: str):
    """
    Load LoRA weights into the pipeline.
    
    Args:
        pipe: RectifiedFlowStateMachine pipeline
        lora_path: Path to the saved LoRA weights
    
    Returns:
        Pipeline with LoRA weights loaded
    """
    from peft import PeftModel
    
    # Check if already a PEFT model
    if hasattr(pipe.unet, 'peft_config'):
        # Already a PEFT model, just load the adapter
        pipe.unet.load_adapter(lora_path, adapter_name="default")
    else:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    
    # Re-apply the forward override for feature extraction
    # The base model is accessible via pipe.unet.base_model.model for PEFT models
    if hasattr(pipe, 'modify_unet_forward'):
        pipe.modify_unet_forward()
    
    print(f"LoRA weights loaded from {lora_path}")
    
    return pipe


def disable_rf_lora(pipe):
    """Disable LoRA layers (use base model weights only)."""
    if hasattr(pipe.unet, 'disable_adapter_layers'):
        pipe.unet.disable_adapter_layers()
    return pipe


def enable_rf_lora(pipe):
    """Enable LoRA layers."""
    if hasattr(pipe.unet, 'enable_adapter_layers'):
        pipe.unet.enable_adapter_layers()
    return pipe


def unload_rf_lora(pipe):
    """
    Completely unload LoRA and return to the base model.
    Use this before training a new LoRA or to free memory.
    """
    if hasattr(pipe.unet, 'peft_config') or hasattr(pipe.unet, 'base_model'):
        print("Unloading LoRA weights...")
        if hasattr(pipe.unet, 'merge_and_unload'):
            # Merge LoRA weights into base model and unload
            pipe.unet = pipe.unet.merge_and_unload()
        elif hasattr(pipe.unet, 'base_model'):
            # Just get the base model without merging
            pipe.unet = pipe.unet.base_model.model
        
        # Re-apply the forward override for feature extraction
        if hasattr(pipe, 'modify_unet_forward'):
            pipe.modify_unet_forward()
        
        print("LoRA unloaded, using base model.")
    else:
        print("No LoRA to unload.")
    
    return pipe