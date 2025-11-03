"""
Controller module for the Cancer Invasion Simulation.

This module implements the SimulationController class using the Singleton pattern
to orchestrate the complete workflow of the physics-informed neural network
simulation for cancer invasion dynamics.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import torch

from scripts.data_loader import fno_data_loader
from scripts.models import CoupledSpeciesTFNO
from scripts.trainer import Trainer_FNO
from scripts.utils import get_compute_device
from scripts.visualization import Visualizer


class SimulationController:
    """
    Singleton controller for managing the cancer invasion simulation workflow.

    This class orchestrates the complete simulation process including data preparation,
    model building, training, and result visualization using a physics-informed
    neural network approach.
    """

    _instance: Optional['SimulationController'] = None

    def __new__(cls, config: Dict[str, Any]) -> 'SimulationController':
        """Implement Singleton pattern to ensure only one controller instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the simulation controller with configuration."""
        if hasattr(self, '_initialized'):
            return  # Already initialized due to Singleton

        self.config = config
        self.device = get_compute_device()
        self.model: Optional[CoupledSpeciesTFNO] = None
        self.trainer: Optional[Trainer_FNO] = None
        self.visualizer: Optional[Visualizer] = None

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        self._initialized = True
        self.logger.info("SimulationController initialized")

    def run(self) -> None:
        """
        Execute the complete simulation workflow.

        This method orchestrates the entire simulation process from data preparation
        through training to result visualization and saving.
        """
        try:
            self.logger.info("Starting cancer invasion simulation")

            # Execute workflow steps
            self._prepare_data()
            self._build_model()
            self._run_training()
            self._save_results()
            self._visualize_results()

            self.logger.info("Simulation completed successfully")

        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            raise

    def _prepare_data(self) -> None:
        """Prepare the training data for the simulation."""
        self.logger.info("Preparing training data")

        try:
            # Load data using configuration parameters
            data_config = self.config.get('data', {})
            batch_size = data_config.get('batch_size', 16)
            n_spatial_points = data_config.get('n_spatial_points', 64)
            n_time_steps = data_config.get('n_time_steps', 64)

            self.grid_x, self.grid_t, self.input_tensor, self.true_initial_conditions = fno_data_loader(
                batch_size=batch_size,
                n_spatial_points=n_spatial_points,
                n_time_steps=n_time_steps,
                device=self.device
            )

            self.logger.info(f"Data prepared: spatial points={n_spatial_points}, time steps={n_time_steps}")

        except Exception as e:
            self.logger.error(f"Failed to prepare data: {e}")
            raise

    def _build_model(self) -> None:
        """Build and initialize the neural network model."""
        self.logger.info("Building neural network model")

        try:
            # Get model configuration
            model_config = self.config.get('model', {})
            fno_config = model_config.get('fno', {})
            modes = fno_config.get('modes', 16)
            width = fno_config.get('width', 64)
            n_layers = fno_config.get('n_layers', 4)

            # Create model
            self.model = CoupledSpeciesTFNO(
                modes=modes,
                width=width,
                n_layers=n_layers
            ).to(self.device)

            self.logger.info(f"Model built: modes={modes}, width={width}, layers={n_layers}")

        except Exception as e:
            self.logger.error(f"Failed to build model: {e}")
            raise

    def _run_training(self) -> None:
        """Execute the training process."""
        self.logger.info("Starting training process")

        try:
            if self.model is None:
                raise ValueError("Model not initialized")

            # Get training configuration
            training_config = self.config.get('training', {})
            learning_rate = training_config.get('learning_rate', 0.0005)
            n_epochs = training_config.get('n_epochs', 2500)
            loss_weights = training_config.get('loss_weights', {
                'physics': 10.0,
                'ic': 7.0,
                'bc_left': 2.0,
                'bc_right': 1.0
            })

            # Get PDE constants
            constants_config = self.config.get('constants', {})
            constants = {
                'D_l': constants_config.get('D_l', 0.1),
                'D_f': constants_config.get('D_f', 0.05),
                'a_lf': constants_config.get('a_lf', 1.0),
                'K_l': constants_config.get('K_l', 1.0),
                'K_f': constants_config.get('K_f', 1.0),
                'X': constants_config.get('X', 1.0)
            }

            # Create optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

            # Create trainer
            self.trainer = Trainer_FNO(
                model=self.model,
                optimizer=optimizer,
                loss_weights=loss_weights,
                device=self.device,
                constants=constants,
                wandb_log=False  # Disable wandb for standalone execution
            )

            # Train the model
            self.trained_model = self.trainer.train(
                input_tensor=self.input_tensor,
                true_initial_conditions=self.true_initial_conditions,
                grid_x=self.grid_x,
                grid_t=self.grid_t,
                n_epochs=n_epochs
            )

            self.logger.info(f"Training completed: {n_epochs} epochs")

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def _save_results(self) -> None:
        """Save the trained model weights."""
        self.logger.info("Saving model weights")

        try:
            output_config = self.config.get('output', {})
            model_path = output_config.get('model_path', 'model_weights.pth')
            output_dir = output_config.get('output_dir', 'results')

            # Make output_dir absolute relative to project root (where config.json is)
            if not os.path.isabs(output_dir):
                config_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(config_dir, output_dir)

            # Ensure the model path includes the output directory if it's just a filename
            if os.path.dirname(model_path) == '':
                model_path = os.path.join(output_dir, model_path)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Save model weights
            torch.save(self.trained_model.state_dict(), model_path)
            self.logger.info(f"Model weights saved to {model_path}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise

    def _visualize_results(self) -> None:
        """Generate and save visualization plots."""
        self.logger.info("Generating visualizations")

        try:
            if self.trained_model is None:
                raise ValueError("Trained model not available for visualization")

            # Create visualizer
            self.visualizer = Visualizer(
                self.trained_model,
                self.input_tensor,
                self.grid_x,
                self.grid_t
            )

            # Generate plots
            self.visualizer.plot_density_snapshot()
            
            # Save results in multiple formats
            self.visualizer.save_results()

            self.logger.info("Visualizations and results saved successfully")

        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            raise