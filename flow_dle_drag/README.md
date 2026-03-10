# Flow-DLE Drag

A Python module for Flow-DLE drag-based image editing, supporting both programmatic usage and DragBench benchmarking.

## Quick start

### Benchmark Mode

```bash
python -m flow_dle_drag.download_dragbench --output-dir drag_bench_data

# Run full benchmark
python -m flow_dle_drag benchmark --root_dir drag_bench_data --result_dir drag_bench_results

# Custom configuration
python -m flow_dle_drag benchmark \
    --device cuda \
    --root_dir drag_bench_data \
    --result_dir drag_bench_results \
    --drag_step 5 \
    --n_pix_step 90 \
    --lr 0.01 \
    --lam 0.5
```

### As Library

```python
from flow_dle_drag import (
    create_pipeline, 
    run_rf_drag, 
    DragConfig, 
    PipelineConfig,
    load_dragbench_sample
)

# Initialize
config = PipelineConfig(model_path="XCLiu/2_rectified_flow_from_sd_1_5")
pipe = create_pipeline(config)

# Load sample
image, prompt, mask, points = load_dragbench_sample("path/to/sample")

# Prepare state
state = pipe.prepare_state(prompt=prompt)
state = pipe.infer_until(state, 50)

# Run drag
drag_config = DragConfig(drag_step=3, n_pix_step=90, lr=0.01)
output = run_rf_drag(pipe, state, mask, points, drag_config)

# Save result
output.final_image.save("result.png")

# Cleanup
pipe.close()
```

## Command-Line Options

### Pipeline Configuration

|Argument|Default|Description|
|---|---|---|
|`--model_path`|`XCLiu/2_rectified_flow_from_sd_1_5`|Path to rectified flow model|
|`--dtype`|`bfloat16`|Torch dtype (float16/bfloat16/float32)|
|`--device`|`cuda`|Device to run on (cuda/cpu)|
|`--height`|`512`|Image height|
|`--width`|`512`|Image width|
|`--num_steps`|`50`|Number of inference steps|
|`--guidance_scale`|`1.5`|CFG guidance scale|

### Drag Editing Parameters

|Argument|Default|Description|
|---|---|---|
|`--drag_step`|`3`|Step to apply drag (earlier = more effect)|
|`--n_pix_step`|`90`|Number of optimization iterations|
|`--lr`|`0.01`|Learning rate for latent optimization|
|`--lam`|`0.5`|Background regularization weight|
|`--unet_feature_idx`|`[3]`|UNet layer indices for features|
|`--sup_res_h`|`256`|Feature supervision resolution height|
|`--sup_res_w`|`256`|Feature supervision resolution width|
|`--r_m`|`1`|Motion supervision radius|
|`--r_p`|`3`|Point tracking search radius|
|`--end_step`|`50`|Final inference step|
|`--show_optim`|`False`|Show optimization process|
|`--vis_interval`|`5`|Visualization interval|

### Benchmark-Specific Options

|Argument|Default|Description|
|---|---|---|
|`--root_dir`|`drag_bench_data`|Root directory for DragBench dataset|
|`--result_dir`|`Auto-generated`|Output directory path|
|`--skip_failed`|`True`|Skip failed samples instead of aborting|
|`--save_intermediates`|`False`|Save intermediate optimization steps|


### Output Structure

```
results/
├── category_name/
│   └── sample_name/
│       ├── dragged_image.png
│       ├── metadata.json          # Contains IF, MD metrics
│       └── intermediate_XXX.png   # Optional
└── benchmark_metrics.json         # Aggregated averages
```

## Evaluation Metrics

The benchmark mode computes two metrics per sample:

|Metric|Description|Lower is Better|
|---|---|---|
|**Image Fidelity (IF)**|LPIPS-based similarity between original and edited image|✓|
|**Mean Distance (MD)**|DIFT-based point tracking accuracy|✓|

Metrics are saved in `benchmark_metrics.json` with per-category averages.


### Example Results

```json
{
  "individual_metrics": [
    {
      "category": "animals",
      "sample_name": "bear_001",
      "image_fidelity": 0.123,
      "mean_distance": 0.456
    }
  ],
  "category_averages": {
    "animals": {
      "image_fidelity": 0.145,
      "mean_distance": 0.512
    }
  },
  "total_samples": 150
}
```

