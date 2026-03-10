"""Configuration and argument parsing for Flow-DLE editor."""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DragConfig:
    """Configuration for drag editing operations."""
    drag_step: int = 20
    n_pix_step: int = 80
    lr: float = 0.01
    lam: float = 0.1
    unet_feature_idx: List[int] = field(default_factory=lambda: [3])
    sup_res_h: int = 256
    sup_res_w: int = 256
    r_m: int = 1
    r_p: int = 3
    end_step: int = 50
    show_optim_process: bool = False
    vis_interval: int = 10


@dataclass
class PipelineConfig:
    """Configuration for the flow model pipeline."""
    model_path: str = "XCLiu/2_rectified_flow_from_sd_1_5"
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    safety_checker_enabled: bool = False
    num_inference_steps: int = 50
    guidance_scale: float = 1.5


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking on DragBench."""
    root_dir: str = "drag_bench_data"
    result_dir: Optional[str] = None
    categories: List[str] = field(default_factory=lambda: [
        'art_work', 'land_scape', 'building_city_view',
        'building_countryside_view', 'animals', 'human_head',
        'human_upper_body', 'human_full_body', 'interior_design',
        'other_objects'
    ])
    skip_failed: bool = True
    save_intermediates: bool = False


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Flow-DLE: Depth-Aware Differentiable Latent Editing on Flow Models"
    )
    
    # Subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Benchmark mode
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmark on DragBench dataset"
    )
    _add_pipeline_args(benchmark_parser)
    _add_drag_args(benchmark_parser)
    _add_benchmark_args(benchmark_parser)
    
    return parser


def _add_pipeline_args(parser: argparse.ArgumentParser):
    """Add pipeline-related arguments."""
    parser.add_argument(
        "--model_path",
        type=str,
        default="XCLiu/2_rectified_flow_from_sd_1_5",
        help="Path to rectified flow model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.5,
        help="CFG guidance scale"
    )


def _add_drag_args(parser: argparse.ArgumentParser):
    """Add drag editing arguments."""
    parser.add_argument(
        "--drag_step",
        type=int,
        default=3,
        help="Step to apply drag (earlier = more effect)"
    )
    parser.add_argument(
        "--n_pix_step",
        type=int,
        default=90,
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for latent optimization"
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.5,
        help="Background regularization weight"
    )
    parser.add_argument(
        "--unet_feature_idx",
        type=int,
        nargs="+",
        default=[3],
        help="UNet layer indices for features"
    )
    parser.add_argument(
        "--sup_res_h",
        type=int,
        default=256,
        help="Feature supervision resolution height"
    )
    parser.add_argument(
        "--sup_res_w",
        type=int,
        default=256,
        help="Feature supervision resolution width"
    )
    parser.add_argument(
        "--r_m",
        type=int,
        default=1,
        help="Motion supervision radius"
    )
    parser.add_argument(
        "--r_p",
        type=int,
        default=3,
        help="Point tracking search radius"
    )
    parser.add_argument(
        "--end_step",
        type=int,
        default=50,
        help="Final inference step"
    )
    parser.add_argument(
        "--show_optim",
        action="store_true",
        help="Show optimization process"
    )
    parser.add_argument(
        "--vis_interval",
        type=int,
        default=5,
        help="Visualization interval"
    )


def _add_benchmark_args(parser: argparse.ArgumentParser):
    """Add benchmark-specific arguments."""
    parser.add_argument(
        "--root_dir",
        type=str,
        default="drag_bench_data",
        help="Root directory for DragBench dataset"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if None)"
    )
    parser.add_argument(
        "--skip_failed",
        action="store_true",
        default=True,
        help="Skip failed samples instead of aborting"
    )
    parser.add_argument(
        "--save_intermediates",
        action="store_true",
        help="Save intermediate optimization steps"
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = create_parser()
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> tuple:
    """Convert parsed arguments to configuration objects."""
    drag_config = DragConfig(
        drag_step=args.drag_step,
        n_pix_step=args.n_pix_step,
        lr=args.lr,
        lam=args.lam,
        unet_feature_idx=args.unet_feature_idx,
        sup_res_h=args.sup_res_h,
        sup_res_w=args.sup_res_w,
        r_m=args.r_m,
        r_p=args.r_p,
        end_step=args.end_step,
        show_optim_process=args.show_optim,
        vis_interval=args.vis_interval
    )
    
    pipeline_config = PipelineConfig(
        model_path=args.model_path,
        torch_dtype=args.dtype,
        device=args.device,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale
    )
    
    benchmark_config = None
    if hasattr(args, 'root_dir'):
        benchmark_config = BenchmarkConfig(
            root_dir=args.root_dir,
            result_dir=args.result_dir,
            skip_failed=args.skip_failed,
            save_intermediates=args.save_intermediates
        )
    
    return drag_config, pipeline_config, benchmark_config