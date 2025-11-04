import torch
import matplotlib.pyplot as plt
import logging
import numpy as np
from typing import Optional, Dict, List
import os
import json


class Visualizer:
    """
    Comprehensive visualization class for cancer invasion simulation results.

    Provides methods to create density snapshots, training loss curves, and save
    results in multiple formats for the physics-informed neural network simulation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        grid_x: torch.Tensor,
        grid_t: torch.Tensor,
        loss_history: Optional[Dict[str, List[float]]] = None
    ) -> None:
        """
        Initialize the visualizer.

        Parameters
        ----------
        model : nn.Module
            Trained neural network model
        input_tensor : torch.Tensor
            Input tensor used for model predictions
        grid_x : torch.Tensor
            Spatial grid coordinates
        grid_t : torch.Tensor
            Temporal grid coordinates
        loss_history : Dict[str, List[float]], optional
            Training loss history for loss visualization
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.input_tensor = input_tensor.to(self.device)
        self.loss_history = loss_history

        # Store grid coordinates as numpy arrays for plotting
        self.grid_x = grid_x.cpu().detach().numpy()
        self.grid_t = grid_t.cpu().detach().numpy()

        # Create 1D coordinate vectors for plotting
        n_spatial = grid_x.shape[1]
        n_temporal = grid_x.shape[2]
        self.x_coords = np.linspace(0, 1, n_spatial)
        self.t_coords = np.linspace(0, 1, n_temporal)

        # Generate model predictions
        self.logger = logging.getLogger(__name__)
        self.logger.info("Generating model predictions for visualization")

        self.model.eval()
        with torch.no_grad():
            # Get predictions (use first batch element)
            self.predictions = self.model(self.input_tensor).cpu().detach().numpy()[0]

    def plot_density_snapshot(self, time_index: Optional[int] = None) -> None:
        """
        Create a snapshot of density evolution at a specific time point.

        Parameters
        ----------
        time_index : int, optional
            Index of the time point to plot. If None, uses the final time point.
        """
        if time_index is None:
            time_index = len(self.t_coords) - 1  # Use final time point
        
        p_l_t = self.predictions[0, :, time_index]  # Leader density at time t
        p_f_t = self.predictions[1, :, time_index]  # Follower density at time t
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        time_value = self.t_coords[time_index]
        fig.suptitle(f'Cancer Invasion: Density Snapshot at t = {time_value:.2f}', fontsize=16)

        # Determine y-axis limits
        y_min = min(p_l_t.min(), p_f_t.min()) * 1.1
        y_max = max(p_l_t.max(), p_f_t.max()) * 1.1

        # Leader density plot
        axes[0].plot(self.x_coords, p_l_t, lw=2, color='blue')
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(y_min, y_max)
        axes[0].set_xlabel('Space (x)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Leader Density $p_l(x,t)$')
        axes[0].grid(True, alpha=0.3)

        # Follower density plot
        axes[1].plot(self.x_coords, p_f_t, lw=2, color='red')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(y_min, y_max)
        axes[1].set_xlabel('Space (x)')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Follower Density $p_f(x,t)$')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot to file
        # Load config to get plot directory
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        output_config = config.get('output', {})
        plot_dir = output_config.get('plot_dir', 'plots')
        
        # Make plot_dir absolute relative to config file location
        if not os.path.isabs(plot_dir):
            plot_dir = os.path.join(os.path.dirname(config_path), plot_dir)
        
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, 'density_snapshot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Density snapshot saved to {plot_path}")
        

    def save_results(self) -> None:
        """
        Save model predictions and results in multiple formats.
        
        Loads output directory from config.json.
        """
        # Load config to get output directory
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        output_config = config.get('output', {})
        results_dir = output_config.get('output_dir', 'results')
        
        # Make results_dir absolute relative to config file location
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(os.path.dirname(config_path), results_dir)
        
        os.makedirs(results_dir, exist_ok=True)
        
        # Save predictions as numpy arrays
        np.savez_compressed(
            os.path.join(results_dir, 'predictions.npz'),
            leader_density=self.predictions[0],
            follower_density=self.predictions[1],
            x_coords=self.x_coords,
            t_coords=self.t_coords
        )
        
        # Save predictions as CSV
        try:
            import pandas as pd
            n_spatial, n_temporal = self.predictions[0].shape
            
            # Create DataFrame with multi-index for space-time data
            data = []
            for i in range(n_spatial):
                for j in range(n_temporal):
                    data.append({
                        'x': self.x_coords[i],
                        't': self.t_coords[j],
                        'leader_density': float(self.predictions[0, i, j]),
                        'follower_density': float(self.predictions[1, i, j])
                    })
            
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)
        except ImportError:
            self.logger.warning("pandas not available, skipping CSV export")
        
        # Save summary statistics as JSON
        summary = {
            'spatial_points': len(self.x_coords),
            'time_points': len(self.t_coords),
            'leader_density_stats': {
                'mean': float(self.predictions[0].mean()),
                'std': float(self.predictions[0].std()),
                'min': float(self.predictions[0].min()),
                'max': float(self.predictions[0].max())
            },
            'follower_density_stats': {
                'mean': float(self.predictions[1].mean()),
                'std': float(self.predictions[1].std()),
                'min': float(self.predictions[1].min()),
                'max': float(self.predictions[1].max())
            }
        }
        
        with open(os.path.join(results_dir, 'results_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Results saved to {results_dir}/ in multiple formats")

    def plot_loss_comparison(
        self,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Plot all loss components on the same graph for comparison.

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot, if None, plot is not saved
        show_plot : bool, optional
            Whether to display the plot, default is True
        """
        if self.loss_history is None:
            self.logger.warning("No loss history available for visualization")
            return

        epochs = range(1, len(self.loss_history['total']) + 1)

        plt.figure(figsize=(12, 8))

        # Plot all losses
        plt.plot(epochs, self.loss_history['total'], 'k-', linewidth=3, label='Total Loss', alpha=0.8)
        plt.plot(epochs, self.loss_history['physics'], 'b-', linewidth=2, label='PDE Physics Loss', alpha=0.8)
        plt.plot(epochs, self.loss_history['ic'], 'r-', linewidth=2, label='Initial Condition Loss', alpha=0.8)
        plt.plot(epochs, self.loss_history['bc_left'], 'g-', linewidth=2, label='Left BC Loss', alpha=0.8)
        plt.plot(epochs, self.loss_history['bc_right'], 'm-', linewidth=2, label='Right BC Loss', alpha=0.8)

        plt.title('Training Loss Comparison - PINN Cancer Invasion', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()

        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Loss comparison plot saved to: {save_path}")

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

    def save_loss_data(self, save_path: str) -> None:
        """
        Save loss history data to a CSV file.

        Parameters
        ----------
        save_path : str
            Path to save the CSV file
        """
        if self.loss_history is None:
            self.logger.warning("No loss history available for saving")
            return

        try:
            import pandas as pd
        except ImportError:
            self.logger.warning("pandas not available, skipping CSV export")
            return

        # Create DataFrame from loss history
        max_length = max(len(losses) for losses in self.loss_history.values())
        epochs = range(1, max_length + 1)

        data = {'epoch': epochs}
        for loss_type, losses in self.loss_history.items():
            data[loss_type] = losses + [np.nan] * (max_length - len(losses))  # Pad with NaN if needed

        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        self.logger.info(f"Loss data saved to: {save_path}")

    def visualize_all(self) -> None:
        """
        Generate all visualizations: density snapshot, training losses, and save data.
        """
        # Get output directories from config
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        output_config = config.get('output', {})
        plot_dir = output_config.get('plot_dir', 'plots')
        results_dir = output_config.get('output_dir', 'results')

        # Make directories absolute
        if not os.path.isabs(plot_dir):
            plot_dir = os.path.join(os.path.dirname(config_path), plot_dir)
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(os.path.dirname(config_path), results_dir)

        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Generate all visualizations
        self.plot_density_snapshot()

        if self.loss_history is not None:
            self.plot_loss_comparison(
                save_path=os.path.join(plot_dir, 'training_losses.png'),
                show_plot=False
            )
            self.save_loss_data(os.path.join(results_dir, 'loss_history.csv'))

        self.save_results()
        self.logger.info("All visualizations completed")

