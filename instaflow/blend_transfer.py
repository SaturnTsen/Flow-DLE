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

# Method 1: Optimize latent to reduce feature loss (using provided features_bg)
import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class BlendConfig:
    blend_start_step: int = 5
    blend_end_step: int = 30
    optim_steps_per_inference: int = 2
    lr: float = 0.02
    lamfg: float = 0.001
    lambg: float = 0.002

def get_callback_with_fg_features(
    pipe,
    state_fg,
    state_bg_ref,  # Reference bg to preserve background features
    mask_tensor,
    config: BlendConfig
):
    loss_history = {}
    
    def feature_loss_callback(i, t, dt, latent_bg, features_bg, v_pred_bg):
        if i not in loss_history:
            loss_history[i] = []
    
        """Optimize latent to reduce feature loss with fg."""
    
        # Check if we have fg features for this step
        fg_item = None
        bg_ref_item = None
        if i >= config.blend_start_step and i < config.blend_end_step:
            for item in state_fg.intermediate_latents:
                if item['step'] == i and 'features' in item:
                    fg_item = item
                    break
            for item in state_bg_ref.intermediate_latents:
                if item['step'] == i and 'features' in item:
                    bg_ref_item = item
                    break
    
        if fg_item is None or bg_ref_item is None:
            # Outside blend window or no features, just do normal update
            next_latent_bg = latent_bg + dt * v_pred_bg
            return next_latent_bg
    
        # Normal update first
        next_latent_bg = latent_bg + dt * v_pred_bg
    
        # Get fg and bg_ref features at this step
        fg_features = fg_item['features']
        bg_ref_features = bg_ref_item['features']
    
        # Enable gradients and optimize
        torch.set_grad_enabled(True)
        next_latent_bg = next_latent_bg.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([next_latent_bg], lr=config.lr)
    
        for opt_step in range(config.optim_steps_per_inference):
            optimizer.zero_grad()
        
            # Re-compute features for the optimized latent
            current_features = []
            name_map = {m: n for n, m in pipe.unet.named_modules()}
            def hook(mod, inp, out): current_features.append((name_map.get(mod, "<unnamed>"), out))
            handles = [pipe.unet.mid_block.register_forward_hook(hook)]
            handles += [b.register_forward_hook(hook) for b in pipe.unet.up_blocks]

            latent_input = torch.cat([next_latent_bg] * 2) if state_bg_ref.do_cfg else next_latent_bg
            vec_t = torch.ones((latent_input.shape[0],), device=state_bg_ref.device) * t
        
            _ = pipe.unet(latent_input, vec_t, encoder_hidden_states=state_bg_ref.prompt_embeds)
            for h in handles: h.remove()
        
            # Compute feature loss
            loss = 0.0
            for (name_cur, feat_cur), (name_fg, feat_fg), (name_bg_ref, feat_bg_ref) in zip(current_features, fg_features, bg_ref_features):
                if name_cur != name_fg or name_cur != name_bg_ref:
                    continue
            
                # Get positive features if using CFG
                if state_bg_ref.do_cfg and feat_cur.shape[0] == 2 * next_latent_bg.shape[0]:
                    feat_cur_pos = feat_cur.chunk(2)[1]
                else:
                    feat_cur_pos = feat_cur
            
                # Resize mask to feature resolution
                feat_h, feat_w = feat_cur_pos.shape[2], feat_cur_pos.shape[3]
                mask_feat = F.interpolate(mask_tensor, size=(feat_h, feat_w), mode='bilinear', align_corners=False)
            
                # Foreground loss: masked region should match fg
                feat_diff_fg = torch.abs(feat_cur_pos - feat_fg.detach())
                loss_fg = (feat_diff_fg * mask_feat).mean()
                
                # Background loss: unmasked region should match bg_ref
                feat_diff_bg = torch.abs(feat_cur_pos - feat_bg_ref.detach())
                loss_bg = (feat_diff_bg * (1 - mask_feat)).mean()
                
                loss += config.lamfg * loss_fg + config.lambg * loss_bg
        
            if loss > 0:
                loss_history[i].append(loss.item())
                loss.backward()
                optimizer.step()
        
        print(f"timestep {i}...", end="")
        if i == len(state_bg_ref.timesteps) -1:
            print(loss_history)
            
        
        torch.set_grad_enabled(False)
        return next_latent_bg.detach()
    return feature_loss_callback
