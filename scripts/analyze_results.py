#!/usr/bin/env python3
"""
Script to read and analyze saved results from the cancer invasion simulation.

This script demonstrates how to load results saved in various formats
without needing the original PyTorch model (.pth file).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_results(results_dir: str = "results"):
    """
    Load saved results from multiple formats.

    Parameters
    ----------
    results_dir : str
        Directory containing saved results

    Returns
    -------
    dict
        Dictionary containing loaded results
    """
    results = {}

    # Load numpy compressed data
    npz_path = os.path.join(results_dir, 'predictions.npz')
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        results['leader_density'] = data['leader_density']
        results['follower_density'] = data['follower_density']
        results['x_coords'] = data['x_coords']
        results['t_coords'] = data['t_coords']
        print(f"Loaded numpy data: {results['leader_density'].shape} shape")
    else:
        print(f"Warning: {npz_path} not found")

    # Load summary statistics
    json_path = os.path.join(results_dir, 'results_summary.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results['summary'] = json.load(f)
        print("Loaded summary statistics")
    else:
        print(f"Warning: {json_path} not found")

    # Try to load CSV data
    csv_path = os.path.join(results_dir, 'predictions.csv')
    if os.path.exists(csv_path):
        try:
            import pandas as pd
            results['dataframe'] = pd.read_csv(csv_path)
            print(f"Loaded CSV data: {len(results['dataframe'])} rows")
        except ImportError:
            print("pandas not available for CSV loading")
    else:
        print(f"Warning: {csv_path} not found")

    return results

def analyze_results(results):
    """Analyze and display results."""
    if 'summary' in results:
        summary = results['summary']
        print("\n=== RESULTS SUMMARY ===")
        print(f"Spatial points: {summary['spatial_points']}")
        print(f"Time points: {summary['time_points']}")

        print("\nLeader Density Statistics:")
        for key, value in summary['leader_density_stats'].items():
            print(f"  {key}: {value:.6f}")

        print("\nFollower Density Statistics:")
        for key, value in summary['follower_density_stats'].items():
            print(f"  {key}: {value:.6f}")

    if 'leader_density' in results and 'follower_density' in results:
        leader = results['leader_density']
        follower = results['follower_density']

        print("\n=== DENSITY ANALYSIS ===")
        print(f"Leader density shape: {leader.shape}")
        print(f"Follower density shape: {follower.shape}")
        print(f"Leader density range: [{leader.min():.6f}, {leader.max():.6f}]")
        print(f"Follower density range: [{follower.min():.6f}, {follower.max():.6f}]")

        # Show final time step
        final_leader = leader[:, -1]
        final_follower = follower[:, -1]
        print("\nFinal time step profiles:")
        print(f"  Leader peak: {final_leader.max():.6f} at x={results['x_coords'][np.argmax(final_leader)]:.3f}")
        print(f"  Follower peak: {final_follower.max():.6f} at x={results['x_coords'][np.argmax(final_follower)]:.3f}")

def plot_from_saved_data(results):
    """Create plots from saved data."""
    if 'leader_density' not in results or 'follower_density' not in results:
        print("Cannot create plots: prediction data not available")
        return

    leader = results['leader_density']
    follower = results['follower_density']
    x_coords = results['x_coords']
    t_coords = results['t_coords']

    # Create snapshot at final time point
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    time_value = t_coords[-1]
    fig.suptitle(f'Cancer Invasion Results: Final State (t = {time_value:.2f})', fontsize=16)

    # Determine y-axis limits
    final_leader = leader[:, -1]
    final_follower = follower[:, -1]
    y_min = min(final_leader.min(), final_follower.min()) * 1.1
    y_max = max(final_leader.max(), final_follower.max()) * 1.1

    # Leader density plot
    axes[0].plot(x_coords, final_leader, lw=2, color='blue')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(y_min, y_max)
    axes[0].set_xlabel('Space (x)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Leader Density $p_l(x,t)$')
    axes[0].grid(True, alpha=0.3)

    # Follower density plot
    axes[1].plot(x_coords, final_follower, lw=2, color='red')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_xlabel('Space (x)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Follower Density $p_f(x,t)$')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save plot
    # Load config to get plot directory
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    output_config = config.get('output', {})
    plot_dir = output_config.get('plot_dir', 'plots')
    
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'results_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved to {plot_path}")
    plt.show()

def main():
    """Main function to demonstrate result loading and analysis."""
    print("Cancer Invasion Results Analysis")
    print("=" * 40)

    results_dir = "results"

    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found.")
        print("Please run the simulation first to generate results.")
        return

    # Load results
    print(f"Loading results from {results_dir}/...")
    results = load_results(results_dir)

    if not results:
        print("No results found to analyze.")
        return

    # Analyze results
    analyze_results(results)

    # Create plots from saved data
    print("\nGenerating plots from saved data...")
    plot_from_saved_data(results)

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()