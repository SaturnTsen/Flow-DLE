"""Benchmarking on DragBench dataset with evaluation metrics."""

import argparse
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

from .config import parse_args, config_from_args
from .pipeline_manager import create_pipeline
from .drag_operations import run_rf_drag
from .utils import (
    set_seed, 
    load_dragbench_sample, 
    setup_result_directory,
    logger
)

from benchmarks import LPIPSImageFidelity, DIFTMeanDistance

import torchvision.transforms as T


class MetricsCollector:
    """Collects and aggregates evaluation metrics across samples."""
    
    def __init__(self):
        self.metrics_by_category: Dict[str, List[Dict]] = {}
        self.all_metrics: List[Dict] = []
    
    def add_sample(self, category: str, sample_name: str, metrics: Dict):
        """Add metrics for a single sample."""
        record = {
            'category': category,
            'sample_name': sample_name,
            **metrics
        }
        self.all_metrics.append(record)
        
        if category not in self.metrics_by_category:
            self.metrics_by_category[category] = []
        self.metrics_by_category[category].append(metrics)
    
    def compute_averages(self) -> Dict[str, Dict[str, float]]:
        """Compute average metrics per category."""
        averages = {}
        
        for category, samples in self.metrics_by_category.items():
            if not samples:
                continue
            
            avg_metrics = {}
            for key in samples[0].keys():
                if isinstance(samples[0][key], (int, float)):
                    values = [s[key] for s in samples if key in s]
                    if values:
                        avg_metrics[key] = sum(values) / len(values)
            
            averages[category] = avg_metrics
        
        return averages
    
    def save_metrics(self, output_path: Path):
        """Save all metrics to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'individual_metrics': self.all_metrics,
                'category_averages': self.compute_averages(),
                'total_samples': len(self.all_metrics)
            }, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")


class MetricsEvaluator:
    """Wrapper for evaluation metrics with lazy initialization."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.if_metric = None
        self.md_metric = None
        self.transform = create_tensor_transform()
    
    def initialize(self):
        """Initialize metrics once (call before processing loop)."""
        logger.info("Initializing evaluation metrics...")
        
        self.if_metric = LPIPSImageFidelity(net="alex", device=self.device)
        self.md_metric = DIFTMeanDistance(
            model_name="sd2-community/stable-diffusion-2-1",
            device=self.device,
        )
        
        logger.info("Evaluation metrics initialized successfully")
    
    def compute(self, base_image: np.ndarray, dragged_image, points: list, prompt: str) -> Dict[str, float]:
        """
        Compute Image Fidelity (IF) and Mean Distance (MD) metrics.
        
        Args:
            base_image: Original image as numpy array (H, W, C)
            dragged_image: Edited image (PIL Image from DragOutput)
            points: List of (x, y) points alternating handle/target
            prompt: Text prompt used for generation
        
        Returns:
            Dictionary with 'image_fidelity' and 'mean_distance' keys
        """
        if self.if_metric is None or self.md_metric is None:
            raise RuntimeError("Metrics not initialized. Call initialize() first.")
        
        # Convert base image (numpy) to tensor
        base_tensor = self.transform(base_image).unsqueeze(0).to(self.device)
        
        # Convert dragged image (PIL) to tensor
        dragged_tensor = self.transform(np.array(dragged_image)).unsqueeze(0).to(self.device)
        
        # Compute Image Fidelity
        image_fidelity = self.if_metric.image_fidelity(base_tensor, dragged_tensor).item()
        
        # Separate handle and target points
        handle_points = [(int(p[0]), int(p[1])) for p in points[::2]]
        target_points = [(int(p[0]), int(p[1])) for p in points[1::2]]
        
        # Compute Mean Distance
        mean_distance = self.md_metric.mean_distance(
            base_tensor,
            dragged_tensor,
            handle_points=handle_points,
            target_points=target_points,
            prompt=prompt,
        )
        
        return {
            'image_fidelity': image_fidelity,
            'mean_distance': mean_distance
        }


def create_tensor_transform() -> T.Compose:
    """Create transformation pipeline for numpy arrays to tensors."""
    return T.Compose([
        T.ToTensor(),
        # Normalize to [-1, 1]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])



def save_sample_results(
    output_dir: str,
    category: str,
    sample_name: str,
    drag_output,
    metrics: Dict[str, float],
    save_intermediates: bool = False
):
    """Save editing results and metrics for a single sample."""
    save_dir = Path(output_dir) / category / sample_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final image
    final_img_path = save_dir / 'dragged_image.png'
    drag_output.final_image.save(final_img_path)
    logger.debug(f"Saved final image: {final_img_path}")
    
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
        'metrics': metrics,
    }
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def benchmark_mode(args: argparse.Namespace):
    """Run benchmark on DragBench dataset with evaluation metrics."""
    set_seed(0xdeadbeef)
    
    # Parse configs
    drag_config, pipeline_config, benchmark_config = config_from_args(args)
    
    if benchmark_config is None:
        raise ValueError("Benchmark config not found. Use 'benchmark' subcommand.")
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipe = create_pipeline(pipeline_config)
    
    # Setup result directory
    result_dir = setup_result_directory(
        benchmark_config.result_dir,
        drag_config
    )
    
    # Initialize metrics collector
    metrics_collector = MetricsCollector()
    
    # Initialize models for metrics evaluation
    device = pipeline_config.device
    logger.info(f"Computing metrics on device: {device}")
    metrics_evaluator = MetricsEvaluator(device=device)
    metrics_evaluator.initialize()
    
    try:
        total_samples = 0
        successful = 0
        failed = 0
        skipped_metrics = 0
        
        # Process each category
        for cat in tqdm(benchmark_config.categories, desc="Categories"):
            file_dir = os.path.join(benchmark_config.root_dir, "DragBench", cat)
            
            if not os.path.isdir(file_dir):
                logger.warning(f"Category directory not found: {file_dir}")
                continue
            
            samples = [s for s in os.listdir(file_dir) if s != '.DS_Store']
            
            for sample_name in tqdm(samples, desc=f"{cat}", leave=False):
                sample_path = os.path.join(file_dir, sample_name)
                total_samples += 1
                
                try:
                    # Load sample
                    source_image, prompt, mask, points = load_dragbench_sample(sample_path)
                    
                    # Prepare state from image
                    state = pipe.prepare_state(
                        prompt=prompt,
                        height=source_image.shape[0],
                        width=source_image.shape[1],
                    )
                    state = pipe.infer_until(state, drag_config.drag_step)
                    
                    # Run drag
                    drag_output = run_rf_drag(
                        pipe,
                        state,
                        source_image,
                        mask * 255,
                        points,
                        drag_config,
                    )
                    
                    # Compute evaluation metrics
                    try:
                        metrics = metrics_evaluator.compute(
                            base_image=source_image,
                            dragged_image=drag_output.final_image,
                            points=points,
                            prompt=prompt,
                        )
                        
                        # Add to collector
                        metrics_collector.add_sample(cat, sample_name, metrics)
                        successful += 1
                        
                    except Exception as metric_error:
                        logger.warning(
                            f"Metric computation failed for {cat}/{sample_name}: {metric_error}"
                        )
                        skipped_metrics += 1
                        # Still save the edited image even if metrics fail
                        metrics = {'error': str(metric_error)}
                        successful += 1
                    
                    # Save results
                    save_sample_results(
                        str(result_dir),
                        cat,
                        sample_name,
                        drag_output,
                        metrics,
                        benchmark_config.save_intermediates
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to process {cat}/{sample_name}: {e}")
                    failed += 1
                    
                    if not benchmark_config.skip_failed:
                        raise
        
        # Compute and display category averages
        averages = metrics_collector.compute_averages()
        
        # Save aggregated metrics
        metrics_path = result_dir / 'benchmark_metrics.json'
        metrics_collector.save_metrics(metrics_path)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Benchmark Summary")
        logger.info("=" * 60)
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped metrics: {skipped_metrics}")
        logger.info(f"Success rate: {successful/total_samples*100:.1f}%")
        logger.info("")
        logger.info("Category Averages:")
        logger.info("-" * 60)
        
        for category, cat_avg in averages.items():
            logger.info(f"\n{category}:")
            for metric_name, value in cat_avg.items():
                logger.info(f"  {metric_name}: {value:.4f}")
        
        # Overall averages
        logger.info("\nOverall Averages:")
        overall_avg = {}
        for cat_metrics in averages.values():
            for metric_name, value in cat_metrics.items():
                if metric_name not in overall_avg:
                    overall_avg[metric_name] = []
                overall_avg[metric_name].append(value)
        
        for metric_name, values in overall_avg.items():
            avg_value = sum(values) / len(values)
            logger.info(f"  {metric_name}: {avg_value:.4f}")
        
        logger.info("-" * 60)
        logger.info(f"Results saved to: {result_dir.absolute()}")
        logger.info(f"Metrics saved to: {metrics_path.absolute()}")
        logger.info("=" * 60)
        
    finally:
        pipe.close()
