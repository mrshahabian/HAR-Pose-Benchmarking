#!/usr/bin/env python3
"""Cross-platform benchmark analysis and report asset generation

Usage:
  python scripts/analyze_cross_platform.py \
    --csv results/cross_platform/All-benchmark_results.csv \
    --outdir results/cross_platform/visualizations --format png --dpi 300

Generates figures and CSV summary tables per device/platform for inclusion in LaTeX.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze cross-platform benchmark results')
    parser.add_argument('--csv', required=True, type=str, help='Path to combined CSV results')
    parser.add_argument('--outdir', required=False, type=str, default='results/cross_platform/visualizations', help='Output directory for figures and tables')
    parser.add_argument('--format', '-f', choices=['png', 'pdf', 'svg'], default='png', help='Figure format')
    parser.add_argument('--dpi', type=int, default=300, help='Figure DPI')
    return parser.parse_args()


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize/ensure expected columns exist
    required = [
        'timestamp','device','gpu','model','backend','resolution','source','source_type',
        'perf_fps','perf_fps_std','perf_fps_min','perf_fps_max','perf_latency_ms','perf_latency_std_ms',
        'perf_latency_p50_ms','perf_latency_p95_ms','perf_latency_p99_ms','perf_capture_time_ms',
        'perf_inference_time_ms','perf_end_to_end_latency_ms','perf_dropped_frames','perf_total_frames',
        'perf_processed_frames','perf_drop_rate','sys_cpu_percent_avg','sys_cpu_percent_max',
        'sys_cpu_percent_std','sys_ram_mb_avg','sys_ram_mb_max','sys_gpu_percent_avg','sys_gpu_percent_max',
        'sys_vram_mb_avg','sys_vram_mb_max','sys_power_watts_avg','sys_power_watts_max',
        'sys_temperature_celsius_avg','sys_temperature_celsius_max'
    ]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # Ensure resolution is treated as categorical string
    df['resolution'] = df['resolution'].astype(str)

    # Create a platform column based on 'device' and 'gpu'
    def refine_platform(device: str, gpu: str) -> str:
        d = (device or '').lower()
        g = (gpu or '').lower()
        if 'aarch64' in d or 'orin' in d or 'jetson' in d or 'orin' in g or 'jetson' in g:
            return 'Jetson'
        if 'rtx a500' in g:
            return 'Laptop'
        if 'rtx 4090' in g or 'geforce rtx 4090' in g:
            return 'Desktop'
        # Fallbacks
        if 'x86_64' in d:
            return 'Desktop'
        return 'Unknown'

    df['platform'] = [refine_platform(d, g) for d, g in zip(df.get('device'), df.get('gpu'))]
    return df


def ensure_outdirs(base: Path):
    (base / 'figures').mkdir(parents=True, exist_ok=True)
    (base / 'tables').mkdir(parents=True, exist_ok=True)


def save_fig(path: Path, fmt: str, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path.with_suffix('.' + fmt), dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {path.with_suffix('.' + fmt)}")


def plot_fps_by_model_backend(df: pd.DataFrame, outdir: Path, fmt: str, dpi: int):
    # Facet by platform and resolution
    g = sns.catplot(
        data=df,
        x='model', y='perf_fps', hue='backend',
        col='resolution', row='platform', kind='bar',
        height=3.2, aspect=1.6, legend=True
    )
    g.set_axis_labels("Model", "FPS")
    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    for ax_row in g.axes:
        for ax in ax_row:
            if ax is None:
                continue
            for c in ax.containers:
                ax.bar_label(c, fmt='%.1f', fontsize=7)
            ax.tick_params(axis='x', rotation=45)
    save_fig(outdir / 'figures/fps_by_model_backend', fmt, dpi)


def plot_latency_by_model_backend(df: pd.DataFrame, outdir: Path, fmt: str, dpi: int):
    g = sns.catplot(
        data=df,
        x='model', y='perf_end_to_end_latency_ms', hue='backend',
        col='resolution', row='platform', kind='bar',
        height=3.2, aspect=1.6
    )
    g.set_axis_labels("Model", "Latency (ms)")
    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    for ax_row in g.axes:
        for ax in ax_row:
            if ax is None:
                continue
            for c in ax.containers:
                ax.bar_label(c, fmt='%.1f', fontsize=7)
            ax.tick_params(axis='x', rotation=45)
    save_fig(outdir / 'figures/latency_by_model_backend', fmt, dpi)


def plot_power_efficiency(df: pd.DataFrame, outdir: Path, fmt: str, dpi: int):
    dfp = df.copy()
    dfp = dfp[(dfp['sys_power_watts_avg'].notna()) & (dfp['sys_power_watts_avg'] > 0)]
    if dfp.empty:
        print('No power data to plot.')
        return
    dfp['fps_per_watt'] = dfp['perf_fps'] / dfp['sys_power_watts_avg']
    g = sns.catplot(
        data=dfp,
        x='model', y='fps_per_watt', hue='backend',
        col='resolution', row='platform', kind='bar',
        height=3.2, aspect=1.6
    )
    g.set_axis_labels("Model", "FPS/Watt")
    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    for ax_row in g.axes:
        for ax in ax_row:
            if ax is None:
                continue
            for c in ax.containers:
                ax.bar_label(c, fmt='%.2f', fontsize=7)
            ax.tick_params(axis='x', rotation=45)
    save_fig(outdir / 'figures/power_efficiency', fmt, dpi)


def plot_resource_usage(df: pd.DataFrame, outdir: Path, fmt: str, dpi: int):
    metrics = [
        ('sys_cpu_percent_avg', 'CPU Usage (%)'),
        ('sys_gpu_percent_avg', 'GPU Usage (%)'),
        ('sys_ram_mb_avg', 'RAM (MB)'),
        ('sys_vram_mb_avg', 'VRAM (MB)')
    ]
    for metric, title in metrics:
        pivot = df.pivot_table(values=metric, index=['platform', 'resolution', 'model'], columns='backend', aggfunc='mean')
        if pivot.empty:
            continue
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': title})
        plt.title(title)
        plt.ylabel('Platform / Resolution / Model')
        plt.xlabel('Backend')
        save_fig(outdir / f'figures/resource_{metric}', fmt, dpi)


def make_summary_tables(df: pd.DataFrame, outdir: Path):
    # Top FPS per platform
    top_fps = df.sort_values('perf_fps', ascending=False).groupby('platform', group_keys=False).head(5)
    top_fps.to_csv(outdir / 'tables/top_fps_per_platform.csv', index=False)

    # Best latency per platform
    best_latency = df.sort_values('perf_end_to_end_latency_ms', ascending=True).groupby('platform', group_keys=False).head(5)
    best_latency.to_csv(outdir / 'tables/best_latency_per_platform.csv', index=False)

    # Aggregate summary per platform/backend/model
    summary = df.groupby(['platform', 'resolution', 'model', 'backend']).agg(
        fps_mean=('perf_fps', 'mean'),
        fps_std=('perf_fps', 'std'),
        latency_ms=('perf_end_to_end_latency_ms', 'mean'),
        cpu_pct=('sys_cpu_percent_avg', 'mean'),
        gpu_pct=('sys_gpu_percent_avg', 'mean'),
        ram_mb=('sys_ram_mb_avg', 'mean'),
        vram_mb=('sys_vram_mb_avg', 'mean')
    ).reset_index()
    summary.to_csv(outdir / 'tables/summary_by_platform_model_backend.csv', index=False)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_outdirs(outdir)

    df = load_results(args.csv)

    # Plots
    plot_fps_by_model_backend(df, outdir, args.format, args.dpi)
    plot_latency_by_model_backend(df, outdir, args.format, args.dpi)
    plot_power_efficiency(df, outdir, args.format, args.dpi)
    plot_resource_usage(df, outdir, args.format, args.dpi)

    # Tables
    make_summary_tables(df, outdir)

    print('\nAnalysis complete. Outputs in:', outdir)


if __name__ == '__main__':
    main()



