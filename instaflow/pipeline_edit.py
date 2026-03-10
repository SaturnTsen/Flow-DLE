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


from dataclasses import dataclass, replace
from typing import List, Optional, Callable
import torch
from .pipeline_rf import RectifiedFlowPipeline
from .drag_utils import override_unet_forward
import copy

@dataclass
class InferenceState:
    initial_latent: torch.FloatTensor
    latent: torch.FloatTensor | torch.Tensor
    timesteps: list
    i: int
    dt: float
    prompt_embeds: torch.FloatTensor
    guidance_scale: float
    do_cfg: bool
    device: torch.device
    intermediate_latents: Optional[List[torch.FloatTensor]]
    
    @torch.no_grad()
    def reset(self):
        self.latent = self.initial_latent.clone()
        self.intermediate_latents = []
        self.i = 0
    
    @torch.no_grad()
    def clone(self):
        return copy.deepcopy(self)
    
    def replace_latent(self, new_latent: torch.Tensor, step: Optional[int] = None):
        """
        Replace current latent with a new one, optionally setting the step.
        
        Args:
            new_latent: The new latent tensor to use
            step: If provided, also set the inference step to this value
        
        Returns:
            self for chaining
        """
        self.latent = new_latent.detach().clone()
        if step is not None:
            self.i = step
        return self


class RectifiedFlowStateMachine(RectifiedFlowPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker=False):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker
        )
    
    def modify_unet_forward(self):
        """Override UNet forward to return intermediate features for drag."""
        self.unet.forward = override_unet_forward(self.unet)
    
    def get_text_embeddings(self, prompt):
        """Get text embeddings for a given prompt."""
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self._execution_device))[0]
        return text_embeddings
    
    @torch.no_grad()
    def prepare_state(
        self,
        prompt=None,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=2.5,
        negative_prompt=None,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        cross_attention_kwargs=None,
        capture_all_intermediate_features=False,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device
        do_cfg = guidance_scale > 1.0

        # batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]

        # encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_cfg,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=(
                cross_attention_kwargs.get("scale", None)
                if cross_attention_kwargs else None
            ),
        )

        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # timesteps & dt
        timesteps = [(1. - i / num_inference_steps) * 1000. for i in range(num_inference_steps)]
        dt = 1.0 / num_inference_steps

        # latents
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        return InferenceState(
            initial_latent=latents.detach().clone(),
            latent=latents,
            timesteps=timesteps,
            i=0,
            dt=dt,
            prompt_embeds=prompt_embeds,
            guidance_scale=guidance_scale,
            do_cfg=do_cfg,
            device=device,
            intermediate_latents=[] if capture_all_intermediate_features else None
        )
    
    @torch.no_grad()
    def infer_from_state(
        self,
        state: InferenceState,
        until: Optional[int] = None,
        do_not_store_features = False,
        callback: Optional[Callable[[int, float, float, torch.FloatTensor, list, torch.FloatTensor], torch.FloatTensor]] = None,
    ):
        """Run inference from current state until specified step."""
        for p in self.unet.parameters():
            p.requires_grad_(False)
        
        # Hook 
        while state.i < (len(state.timesteps) if until is None else until):
            features = []
            name_map = {m: n for n, m in self.unet.named_modules()}
            
            def hook(mod, inp, out):
                if not do_not_store_features:
                    features.append((name_map.get(mod, "<unnamed>"), out))
            
            if not do_not_store_features:
                handles =  [self.unet.mid_block.register_forward_hook(hook)]
                handles += [b.register_forward_hook(hook) for b in self.unet.up_blocks]

            features.clear()

            state.latent = state.latent.detach().requires_grad_(True)

            if state.intermediate_latents is not None:
                state.intermediate_latents.append({
                    "step": state.i,
                    "timestep": state.timesteps[state.i],
                    "latent": state.latent.detach().clone(),
                })

            latent_model_input = (torch.cat([state.latent] * 2) if state.do_cfg else state.latent)
            t = state.timesteps[state.i]

            # Reset all gradients to ensure only the current step's computations are tracked

            vec_t = torch.ones((latent_model_input.shape[0],), device=state.device) * t
            v_pred = self.unet(
                latent_model_input,
                vec_t,
                encoder_hidden_states=state.prompt_embeds
            )
            # Handle both tuple and tensor returns
            if isinstance(v_pred, tuple):
                v_pred = v_pred[0]
            elif hasattr(v_pred, 'sample'):
                v_pred = v_pred.sample

            if state.do_cfg:
                v_pred_neg, v_pred_text = v_pred.chunk(2)
                v_pred = v_pred_neg + state.guidance_scale * (v_pred_text - v_pred_neg)

            if callback is None:
                next_latent = state.latent + state.dt * v_pred
            else:
                next_latent = callback(state.i, t, state.dt, state.latent, features, v_pred)

            state.latent = next_latent.detach()

            if not do_not_store_features and state.intermediate_latents is not None:
                state.intermediate_latents[-1]["features"] = [(name, feat.detach().clone()) for name, feat in features]
                state.intermediate_latents[-1]["v_pred"] = v_pred.detach().clone()

            # Remove hooks
            if not do_not_store_features:
                for h in handles:
                    h.remove()

            state.i += 1
        
        return state

    @torch.no_grad()
    def invert_from_state(
        self,
        state: InferenceState,
        until: int = 0,
        do_not_store_features: bool = False,
        callback: Optional[Callable[[int, float, float, torch.FloatTensor, list, torch.FloatTensor], torch.FloatTensor]] = None,
    ):
        """Run RF inversion from current state index down to `until` (inclusive-exclusive).

        This integrates the same ODE backward with Euler steps:
            x <- x - dt * v(x, t)

        Notes:
        - `state.i` is treated as the next forward step index.
        - If `state.i == len(state.timesteps)`, latent is treated as terminal sample.
        - Inversion iterates step indices `state.i-1, ..., until`.
        """
        if until < 0 or until > len(state.timesteps):
            raise ValueError(f"`until` must be in [0, {len(state.timesteps)}], got {until}.")

        for p in self.unet.parameters():
            p.requires_grad_(False)

        while state.i > until:
            features = []
            name_map = {m: n for n, m in self.unet.named_modules()}

            def hook(mod, inp, out):
                if not do_not_store_features:
                    features.append((name_map.get(mod, "<unnamed>"), out))

            if not do_not_store_features:
                handles = [self.unet.mid_block.register_forward_hook(hook)]
                handles += [b.register_forward_hook(hook) for b in self.unet.up_blocks]

            features.clear()

            state.latent = state.latent.detach().requires_grad_(True)

            step_idx = state.i - 1
            t = state.timesteps[step_idx]

            if state.intermediate_latents is not None:
                state.intermediate_latents.append({
                    "step": step_idx,
                    "timestep": t,
                    "latent": state.latent.detach().clone(),
                })

            latent_model_input = torch.cat([state.latent] * 2) if state.do_cfg else state.latent

            vec_t = torch.ones((latent_model_input.shape[0],), device=state.device) * t
            v_pred = self.unet(
                latent_model_input,
                vec_t,
                encoder_hidden_states=state.prompt_embeds,
            )

            if isinstance(v_pred, tuple):
                v_pred = v_pred[0]
            elif hasattr(v_pred, "sample"):
                v_pred = v_pred.sample

            if state.do_cfg:
                v_pred_neg, v_pred_text = v_pred.chunk(2)
                v_pred = v_pred_neg + state.guidance_scale * (v_pred_text - v_pred_neg)

            if callback is None:
                prev_latent = state.latent - state.dt * v_pred
            else:
                prev_latent = callback(step_idx, t, state.dt, state.latent, features, v_pred)

            state.latent = prev_latent.detach()

            if not do_not_store_features and state.intermediate_latents is not None:
                state.intermediate_latents[-1]["features"] = [(name, feat.detach().clone()) for name, feat in features]
                state.intermediate_latents[-1]["v_pred"] = v_pred.detach().clone()

            if not do_not_store_features:
                for h in handles:
                    h.remove()

            state.i -= 1

        return state