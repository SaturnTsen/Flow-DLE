# FlowDLE: Differentiable Latent Editing on Flow Models

This repository explores differentiable latent editing on top of InstaFlow's Rectified Flow pipeline. The core idea is to guide background generation using foreground features while preserving background features outside the mask.

## Highlights
- Feature-guided latent optimization at selected timesteps.
- Dual loss: foreground feature alignment + background feature preservation.
- Supports hard blending and gradient-guided variants for comparison.

## Environment
This project uses the InstaFlow codebase with a Stable Diffusion 1.5 rectified flow checkpoint.

Recommended Python: 3.11 with CUDA enabled PyTorch.

## Quick Start
Open the notebook and run cells in order:

1. Generate foreground with feature capture enabled.
2. Generate background.
3. Paint a mask for the target region.
4. Run Method 1 (feature-guided optimization).
5. Compare results across methods.

Notebook:
- transfer.ipynb

## Drag (Notebook)
This repo also includes a drag-style editing workflow in drag.ipynb. It demonstrates interactive point-based control for editing trajectories during inference.

Typical flow:
1. Run the notebook setup and load the model.
2. Provide source/target point pairs (or use the UI widget if available).
3. Run the drag editing loop to update the latent trajectory.

Notebook:
- drag.ipynb

## Blend (Notebook)

### Method 1: Feature-Guided Optimization
The main method optimizes the latent at each denoising step within a window:

- Foreground loss: match masked features to foreground features.
- Background loss: preserve unmasked features to a reference background.

Reference background is obtained by cloning the background inference state and re-running with feature capture enabled.

### Notes
- The mask is a 0-1 float map.
- Feature losses are L1 over intermediate UNet features.
- CFG batches are handled by selecting the positive branch.

### Results
Use the comparison cell in the notebook to visualize:

- Image-level blending
- Method 1: Optimize (Adam)
- Method 2: Direct Correction
- Method 3: Hard Blend
- Method 4: Gradient Guidance

## Acknowledgements
Built on InstaFlow and diffusers. Thanks to the original authors of their excellent work.