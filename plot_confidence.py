#!/usr/bin/env python3
"""
Plotting functions for RibbonFold confidence data visualization
Similar to ColabFold plotting functionality
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import os
import glob
from typing import Dict, List, Tuple, Optional


def plot_plddt(plddt_scores: List[float], title: str = "Predicted LDDT", save_path: str = None) -> str:
    """
    Plot pLDDT confidence scores
    
    Args:
        plddt_scores: List of per-residue pLDDT scores
        title: Plot title
        save_path: Path to save the plot (optional)
    
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=(12, 4))
    
    # Create residue indices
    residue_indices = list(range(1, len(plddt_scores) + 1))
    
    # Color mapping for confidence levels
    colors = []
    for score in plddt_scores:
        if score > 90:
            colors.append('#0053D6')  # Very high (blue)
        elif score > 70:
            colors.append('#65CBF3')  # Confident (light blue)  
        elif score > 50:
            colors.append('#FFDB13')  # Low (yellow)
        else:
            colors.append('#FF7D45')  # Very low (orange)
    
    # Create bar plot
    bars = plt.bar(residue_indices, plddt_scores, color=colors, width=1.0, edgecolor='none')
    
    # Customize plot
    plt.xlabel('Residue')
    plt.ylabel('Predicted LDDT')
    plt.title(title)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add confidence level legend
    legend_elements = [
        patches.Patch(color='#0053D6', label='Very high (90-100)'),
        patches.Patch(color='#65CBF3', label='Confident (70-90)'),
        patches.Patch(color='#FFDB13', label='Low (50-70)'),
        patches.Patch(color='#FF7D45', label='Very low (0-50)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Add mean pLDDT text
    mean_plddt = np.mean(plddt_scores)
    plt.text(0.02, 0.98, f'Mean pLDDT: {mean_plddt:.1f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'plddt_plot.png'
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_pae(pae_matrix: List[List[float]], title: str = "Predicted Aligned Error", save_path: str = None) -> str:
    """
    Plot PAE (Predicted Aligned Error) heatmap
    
    Args:
        pae_matrix: 2D list/array of PAE values
        title: Plot title  
        save_path: Path to save the plot (optional)
    
    Returns:
        Path to saved plot
    """
    pae_array = np.array(pae_matrix)
    
    plt.figure(figsize=(8, 8))
    
    # Create heatmap
    im = plt.imshow(pae_array, cmap='viridis_r', vmin=0, vmax=np.max(pae_array))
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Expected position error (Å)', rotation=270, labelpad=20)
    
    # Customize plot
    plt.xlabel('Scored residue')
    plt.ylabel('Aligned residue')
    plt.title(title)
    
    # Add max PAE text
    max_pae = np.max(pae_array)
    plt.text(0.02, 0.98, f'Max PAE: {max_pae:.1f} Å', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'pae_plot.png'
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_coverage_placeholder(num_residues: int, title: str = "MSA Coverage", save_path: str = None) -> str:
    """
    Create a placeholder coverage plot (since we don't have MSA coverage data)
    
    Args:
        num_residues: Number of residues
        title: Plot title
        save_path: Path to save the plot (optional)
    
    Returns:
        Path to saved plot
    """
    plt.figure(figsize=(12, 4))
    
    # Create dummy coverage data (you could replace this with actual MSA coverage if available)
    residue_indices = list(range(1, num_residues + 1))
    # Simulate some coverage pattern
    coverage = np.random.beta(2, 1, num_residues) * 100  # Beta distribution for realistic coverage
    
    plt.bar(residue_indices, coverage, color='#2E8B57', width=1.0, edgecolor='none')
    
    plt.xlabel('Residue')
    plt.ylabel('Coverage (%)')
    plt.title(title)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add mean coverage text
    mean_coverage = np.mean(coverage)
    plt.text(0.02, 0.98, f'Mean coverage: {mean_coverage:.1f}%', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'coverage_plot.png'
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def find_best_model(results_dir: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Find the best model based on highest mean pLDDT from confidence JSON files
    
    Args:
        results_dir: Directory containing confidence JSON files
    
    Returns:
        Tuple of (best_confidence_file_path, best_confidence_data)
    """
    confidence_files = glob.glob(os.path.join(results_dir, "*confidence_*.json"))
    
    if not confidence_files:
        print("No confidence files found!")
        return None, None
    
    best_file = None
    best_data = None
    best_plddt = -1
    
    for conf_file in confidence_files:
        try:
            with open(conf_file, 'r') as f:
                data = json.load(f)
            
            mean_plddt = data.get('mean_plddt', 0)
            if mean_plddt > best_plddt:
                best_plddt = mean_plddt
                best_file = conf_file
                best_data = data
                
        except Exception as e:
            print(f"Error reading {conf_file}: {e}")
            continue
    
    if best_file:
        print(f"Best model found: {os.path.basename(best_file)} (mean pLDDT: {best_plddt:.2f})")
    
    return best_file, best_data


def create_confidence_plots(results_dir: str, output_prefix: str = None) -> Dict[str, str]:
    """
    Create all confidence plots from the best model in results directory
    
    Args:
        results_dir: Directory containing confidence JSON files
        output_prefix: Prefix for output plot files
    
    Returns:
        Dictionary with plot types as keys and file paths as values
    """
    # Find best model
    best_file, best_data = find_best_model(results_dir)
    
    if best_data is None:
        print("No confidence data found!")
        return {}
    
    # Extract base name for output files
    if output_prefix is None:
        base_name = os.path.splitext(os.path.basename(best_file))[0]
        base_name = base_name.replace('_confidence', '')
        output_prefix = os.path.join(results_dir, base_name)
    
    plot_files = {}
    
    # Plot pLDDT
    if 'plddt' in best_data:
        plddt_file = f"{output_prefix}_plddt.png"
        plot_plddt(best_data['plddt'], 
                  title=f"Predicted LDDT (Mean: {best_data.get('mean_plddt', 0):.1f})",
                  save_path=plddt_file)
        plot_files['plddt'] = plddt_file
        print(f"Created pLDDT plot: {plddt_file}")
    
    # Plot PAE if available
    if 'pae' in best_data:
        pae_file = f"{output_prefix}_pae.png"
        plot_pae(best_data['pae'], 
                title=f"Predicted Aligned Error (Max: {best_data.get('max_pae', 0):.1f} Å)",
                save_path=pae_file)
        plot_files['pae'] = pae_file
        print(f"Created PAE plot: {pae_file}")
    
    # Create coverage placeholder
    if 'plddt' in best_data:
        coverage_file = f"{output_prefix}_coverage.png"
        plot_coverage_placeholder(len(best_data['plddt']), save_path=coverage_file)
        plot_files['coverage'] = coverage_file
        print(f"Created coverage plot: {coverage_file}")
    
    return plot_files


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create confidence plots for RibbonFold results")
    parser.add_argument("--results_dir", required=True, help="Directory containing confidence JSON files")
    parser.add_argument("--output_prefix", help="Prefix for output plot files")
    
    args = parser.parse_args()
    
    plots = create_confidence_plots(args.results_dir, args.output_prefix)
    
    print("\nGenerated plots:")
    for plot_type, plot_file in plots.items():
        print(f"  {plot_type}: {plot_file}")


if __name__ == "__main__":
    main()
