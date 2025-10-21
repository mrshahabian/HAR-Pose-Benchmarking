#!/usr/bin/env python3
"""Visualize benchmark results with charts and graphs"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Visualize YOLO benchmark results'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to benchmark results JSON file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results/visualizations',
        help='Output directory for charts'
    )
    
    parser.add_argument(
        '--format', '-f',
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Output format for charts'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output images'
    )
    
    return parser.parse_args()


def load_results(json_path: str) -> pd.DataFrame:
    """Load and flatten benchmark results into DataFrame"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Flatten nested structure
    flattened = []
    for result in results:
        flat = {
            'model': result['model'],
            'backend': result['backend'],
            'resolution': f"{result['resolution'][0]}x{result['resolution'][1]}",
            'source': result['source'],
            'device': result['device_info'].get('machine', 'Unknown'),
            'gpu': result['device_info'].get('gpu', 'None'),
        }
        
        # Add performance metrics
        perf = result.get('performance', {})
        flat['fps'] = perf.get('fps', 0)
        flat['latency_ms'] = perf.get('latency_ms', 0)
        flat['latency_p95_ms'] = perf.get('latency_p95_ms', 0)
        flat['dropped_frames'] = perf.get('dropped_frames', 0)
        flat['drop_rate'] = perf.get('drop_rate', 0)
        
        # Add system metrics
        sys_metrics = result.get('system', {})
        flat['cpu_percent'] = sys_metrics.get('cpu_percent_avg', 0)
        flat['gpu_percent'] = sys_metrics.get('gpu_percent_avg', 0)
        flat['ram_mb'] = sys_metrics.get('ram_mb_avg', 0)
        flat['vram_mb'] = sys_metrics.get('vram_mb_avg', 0)
        flat['power_watts'] = sys_metrics.get('power_watts_avg', 0)
        
        # Add accuracy metrics
        acc = result.get('accuracy', {})
        flat['map_keypoint'] = acc.get('map_keypoint', 0)
        flat['oks'] = acc.get('oks_avg', 0)
        
        flattened.append(flat)
    
    return pd.DataFrame(flattened)


def plot_fps_comparison(df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int):
    """Plot FPS comparison across models and backends"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Group by model and backend
    for idx, resolution in enumerate(df['resolution'].unique()):
        data = df[df['resolution'] == resolution]
        
        ax = axes[idx]
        
        # Create bar plot
        x_labels = []
        fps_values = []
        colors = []
        
        for model in sorted(data['model'].unique()):
            for backend in sorted(data['backend'].unique()):
                subset = data[(data['model'] == model) & (data['backend'] == backend)]
                if not subset.empty:
                    x_labels.append(f"{model}\n{backend}")
                    fps_values.append(subset['fps'].values[0])
                    
                    # Color by backend
                    if backend == 'pytorch':
                        colors.append('#3498db')
                    elif backend == 'tensorrt':
                        colors.append('#e74c3c')
                    elif backend == 'openvino':
                        colors.append('#2ecc71')
                    else:
                        colors.append('#95a5a6')
        
        bars = ax.bar(range(len(x_labels)), fps_values, color=colors)
        ax.set_xlabel('Model / Backend')
        ax.set_ylabel('FPS')
        ax.set_title(f'FPS Comparison ({resolution})')
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / f'fps_comparison.{fmt}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_latency_distribution(df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int):
    """Plot latency distribution"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by model
    models = sorted(df['model'].unique())
    data_by_model = [df[df['model'] == model]['latency_ms'].values for model in models]
    
    bp = ax.boxplot(data_by_model, tick_labels=models, patch_artist=True)
    
    # Color boxes
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('End-to-End Latency Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'latency_distribution.{fmt}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_resource_usage(df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int):
    """Plot resource usage heatmap"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Pivot for heatmaps
    metrics = ['cpu_percent', 'gpu_percent', 'ram_mb', 'vram_mb']
    titles = ['CPU Usage (%)', 'GPU Usage (%)', 'RAM Usage (MB)', 'VRAM Usage (MB)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        # Create pivot table
        pivot = df.pivot_table(
            values=metric,
            index='model',
            columns='backend',
            aggfunc='mean'
        )
        
        if not pivot.empty and pivot.sum().sum() > 0:
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': title})
            ax.set_title(title)
            ax.set_xlabel('Backend')
            ax.set_ylabel('Model')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(title)
    
    plt.tight_layout()
    output_path = output_dir / f'resource_usage.{fmt}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_accuracy_vs_performance(df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int):
    """Plot accuracy vs performance scatter plot"""
    # Only plot if accuracy data is available
    if df['oks'].sum() == 0:
        print("Skipping accuracy vs performance plot (no accuracy data)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    for model in df['model'].unique():
        data = df[df['model'] == model]
        ax.scatter(data['fps'], data['oks'], label=model, s=100, alpha=0.7)
    
    ax.set_xlabel('FPS')
    ax.set_ylabel('OKS (Object Keypoint Similarity)')
    ax.set_title('Accuracy vs Performance Trade-off')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'accuracy_vs_performance.{fmt}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_power_efficiency(df: pd.DataFrame, output_dir: Path, fmt: str, dpi: int):
    """Plot power efficiency (FPS per Watt)"""
    # Only plot if power data is available
    if df['power_watts'].sum() == 0:
        print("Skipping power efficiency plot (no power data)")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate efficiency
    df_power = df[df['power_watts'] > 0].copy()
    df_power['fps_per_watt'] = df_power['fps'] / df_power['power_watts']
    
    # Bar plot
    x_labels = []
    efficiency_values = []
    colors = []
    
    for model in sorted(df_power['model'].unique()):
        for backend in sorted(df_power['backend'].unique()):
            subset = df_power[(df_power['model'] == model) & (df_power['backend'] == backend)]
            if not subset.empty:
                x_labels.append(f"{model}\n{backend}")
                efficiency_values.append(subset['fps_per_watt'].values[0])
                colors.append('#2ecc71' if backend == 'tensorrt' else '#3498db')
    
    bars = ax.bar(range(len(x_labels)), efficiency_values, color=colors)
    ax.set_xlabel('Model / Backend')
    ax.set_ylabel('FPS per Watt')
    ax.set_title('Power Efficiency (Higher is Better)')
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / f'power_efficiency.{fmt}'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def generate_summary_report(df: pd.DataFrame, output_dir: Path):
    """Generate text summary report"""
    report_path = output_dir / 'summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("YOLO POSE ESTIMATION BENCHMARK SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Benchmarks: {len(df)}\n")
        f.write(f"Models: {', '.join(df['model'].unique())}\n")
        f.write(f"Backends: {', '.join(df['backend'].unique())}\n")
        f.write(f"Resolutions: {', '.join(df['resolution'].unique())}\n")
        f.write(f"Device: {df['device'].iloc[0]}\n")
        f.write(f"GPU: {df['gpu'].iloc[0]}\n\n")
        
        f.write("="*70 + "\n")
        f.write("TOP PERFORMERS\n")
        f.write("="*70 + "\n\n")
        
        # Highest FPS
        top_fps = df.nlargest(3, 'fps')
        f.write("Highest FPS:\n")
        for idx, row in top_fps.iterrows():
            f.write(f"  {row['model']} ({row['backend']}, {row['resolution']}): {row['fps']:.2f} FPS\n")
        f.write("\n")
        
        # Lowest Latency
        top_latency = df.nsmallest(3, 'latency_ms')
        f.write("Lowest Latency:\n")
        for idx, row in top_latency.iterrows():
            f.write(f"  {row['model']} ({row['backend']}, {row['resolution']}): {row['latency_ms']:.2f} ms\n")
        f.write("\n")
        
        # Best Power Efficiency (if available)
        if df['power_watts'].sum() > 0:
            df_power = df[df['power_watts'] > 0].copy()
            df_power['fps_per_watt'] = df_power['fps'] / df_power['power_watts']
            top_efficiency = df_power.nlargest(3, 'fps_per_watt')
            f.write("Best Power Efficiency:\n")
            for idx, row in top_efficiency.iterrows():
                f.write(f"  {row['model']} ({row['backend']}, {row['resolution']}): {row['fps_per_watt']:.2f} FPS/W\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
    
    print(f"Saved: {report_path}")


def main():
    """Main entry point"""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("BENCHMARK VISUALIZATION")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print(f"Format: {args.format}")
    print("="*70 + "\n")
    
    # Load results
    print("Loading results...")
    df = load_results(args.input)
    print(f"Loaded {len(df)} benchmark results\n")
    
    # Generate plots
    print("Generating visualizations...")
    
    plot_fps_comparison(df, output_dir, args.format, args.dpi)
    plot_latency_distribution(df, output_dir, args.format, args.dpi)
    plot_resource_usage(df, output_dir, args.format, args.dpi)
    plot_accuracy_vs_performance(df, output_dir, args.format, args.dpi)
    plot_power_efficiency(df, output_dir, args.format, args.dpi)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, output_dir)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

