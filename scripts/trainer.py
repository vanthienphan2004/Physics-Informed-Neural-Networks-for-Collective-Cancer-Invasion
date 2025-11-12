"""
Training module for the Cancer Invasion Simulation.

This module contains the Trainer_FNO class for training physics-informed
neural operators on the coupled species cancer invasion PDE system.
"""

import logging
import time
from typing import Dict

import torch
import torch.nn as nn


def compute_derivatives(
    coords: torch.Tensor, rho_l: torch.Tensor, rho_f: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """
    Compute spatial and temporal derivatives of density fields.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinate tensor of shape (batch_size, 2) with [x, t] coordinates
    rho_l : torch.Tensor
        Leader species density field
    rho_f : torch.Tensor
        Follower species density field

    Returns
    -------
    tuple[torch.Tensor, ...]
        Derivatives: (rho_l_t, rho_l_x, rho_l_xx, rho_f_t, rho_f_x, rho_f_xx)
    """
    # First derivatives of leader density
    rho_l_1grads = torch.autograd.grad(
        rho_l, coords, grad_outputs=torch.ones_like(rho_l), create_graph=True
    )[0]
    rho_l_2grads = torch.autograd.grad(
        rho_l_1grads,
        coords,
        grad_outputs=torch.ones_like(rho_l_1grads),
        create_graph=True,
    )[0]

    rho_l_x = rho_l_1grads[:, 0]  # ∂ρ_l/∂x
    rho_l_t = rho_l_1grads[:, 1]  # ∂ρ_l/∂t
    rho_l_xx = rho_l_2grads[:, 0]  # ∂²ρ_l/∂x²

    # First derivatives of follower density
    rho_f_1grads = torch.autograd.grad(
        rho_f, coords, grad_outputs=torch.ones_like(rho_f), create_graph=True
    )[0]
    rho_f_2grads = torch.autograd.grad(
        rho_f_1grads,
        coords,
        grad_outputs=torch.ones_like(rho_f_1grads),
        create_graph=True,
    )[0]

    rho_f_x = rho_f_1grads[:, 0]  # ∂ρ_f/∂x
    rho_f_t = rho_f_1grads[:, 1]  # ∂ρ_f/∂t
    rho_f_xx = rho_f_2grads[:, 0]  # ∂²ρ_f/∂x²

    return rho_l_t, rho_l_x, rho_l_xx, rho_f_t, rho_f_x, rho_f_xx


class Trainer_FNO:
    """
    Trainer for Physics-Informed Neural Operator (PINO) models.

    This class handles the training loop for FNO models with physics-informed
    loss functions for the coupled species cancer invasion system.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_weights: Dict[str, float],
        device: torch.device,
        constants: Dict[str, float],
        wandb_log: bool = False,
    ) -> None:
        """
        Initialize the FNO trainer.

        Parameters
        ----------
        model : nn.Module
            The neural network model to train
        optimizer : torch.optim.Optimizer
            Optimizer for training
        loss_weights : Dict[str, float]
            Weights for different loss components
        device : torch.device
            Device to run training on
        constants : Dict[str, float]
            PDE constants for the physics loss
        wandb_log : bool, optional
            Whether to log to Weights & Biases, default is False
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.device = device
        self.constants = constants
        self.wandb_log = wandb_log

        # Loss history tracking
        self.loss_history = {
            "total": [],
            "physics": [],
            "ic": [],
            "bc_left": [],
            "bc_right": [],
        }

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _compute_physics_loss(
        self, model_output: torch.Tensor, grid_x: torch.Tensor, grid_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute physics-informed loss based on PDE residuals.

        Parameters
        ----------
        model_output : torch.Tensor
            Model predictions of shape (batch, 2, n_x, n_t)
        grid_x : torch.Tensor
            Spatial grid coordinates
        grid_t : torch.Tensor
            Time grid coordinates

        Returns
        -------
        torch.Tensor
            Physics loss value
        """
        p_l = model_output[:, 0, :, :]  # Leader density
        p_f = model_output[:, 1, :, :]  # Follower density

        D_l, D_f, a_lf, K_l, K_f, X = self.constants.values()

        # Compute derivatives using automatic differentiation
        # Note: We need to create a computation graph by treating the output as a function of the inputs
        # For FNO, we need to ensure the model output depends on the coordinate inputs

        # Create coordinate tensors that require gradients for differentiation
        x_coords = grid_x.detach().requires_grad_(True)
        t_coords = grid_t.detach().requires_grad_(True)

        # Re-run forward pass with gradient-enabled coordinates to establish computation graph
        # This is a simplified approach - in practice, the model should be designed to take coordinates as inputs
        batch_size, n_x, n_t = grid_x.shape

        # For PINN-style physics loss, we need spatial-temporal derivatives
        # Let's compute them using finite differences as a simpler approach first
        # Spatial derivatives (central difference)
        dp_l_dx = torch.zeros_like(p_l)
        dp_l_dx[:, 1:-1, :] = (p_l[:, 2:, :] - p_l[:, :-2, :]) / (
            2 * (grid_x[:, 2, :] - grid_x[:, 0, :]).unsqueeze(1)
        )
        # Boundary conditions for derivatives
        dp_l_dx[:, 0, :] = (p_l[:, 1, :] - p_l[:, 0, :]) / (
            grid_x[:, 1, :] - grid_x[:, 0, :]
        )
        dp_l_dx[:, -1, :] = (p_l[:, -1, :] - p_l[:, -2, :]) / (
            grid_x[:, -1, :] - grid_x[:, -2, :]
        )

        d2p_l_dx2 = torch.zeros_like(p_l)
        d2p_l_dx2[:, 1:-1, :] = (
            p_l[:, 2:, :] - 2 * p_l[:, 1:-1, :] + p_l[:, :-2, :]
        ) / ((grid_x[:, 2, :] - grid_x[:, 0, :]).unsqueeze(1) ** 2)

        dp_f_dx = torch.zeros_like(p_f)
        dp_f_dx[:, 1:-1, :] = (p_f[:, 2:, :] - p_f[:, :-2, :]) / (
            2 * (grid_x[:, 2, :] - grid_x[:, 0, :]).unsqueeze(1)
        )
        dp_f_dx[:, 0, :] = (p_f[:, 1, :] - p_f[:, 0, :]) / (
            grid_x[:, 1, :] - grid_x[:, 0, :]
        )
        dp_f_dx[:, -1, :] = (p_f[:, -1, :] - p_f[:, -2, :]) / (
            grid_x[:, -1, :] - grid_x[:, -2, :]
        )

        d2p_f_dx2 = torch.zeros_like(p_f)
        d2p_f_dx2[:, 1:-1, :] = (
            p_f[:, 2:, :] - 2 * p_f[:, 1:-1, :] + p_f[:, :-2, :]
        ) / ((grid_x[:, 2, :] - grid_x[:, 0, :]).unsqueeze(1) ** 2)

        # Time derivatives (forward difference for simplicity)
        dp_l_dt = torch.zeros_like(p_l)
        dp_l_dt[:, :, :-1] = (p_l[:, :, 1:] - p_l[:, :, :-1]) / (
            grid_t[:, :, 1:] - grid_t[:, :, :-1]
        )
        dp_l_dt[:, :, -1] = dp_l_dt[:, :, -2]  # Extend last value

        dp_f_dt = torch.zeros_like(p_f)
        dp_f_dt[:, :, :-1] = (p_f[:, :, 1:] - p_f[:, :, :-1]) / (
            grid_t[:, :, 1:] - grid_t[:, :, :-1]
        )
        dp_f_dt[:, :, -1] = dp_f_dt[:, :, -2]  # Extend last value

        # PDE residuals for leader species
        residual_rho_l = (
            (D_l * d2p_l_dx2)
            - 2 * (K_l * a_lf) * (dp_l_dx**2)
            - 2 * (K_l * a_lf) * (p_l * d2p_l_dx2)
            - (K_l * a_lf * dp_l_dx * dp_f_dx)
            - (K_l * a_lf * p_l * d2p_f_dx2)
            - (X * dp_l_dx)
            - dp_l_dt
        )

        # PDE residuals for follower species
        residual_rho_f = (
            (D_f * d2p_f_dx2)
            - 2 * (K_f * a_lf) * (dp_f_dx**2)
            - 2 * (K_f * a_lf) * (p_f * d2p_f_dx2)
            - (K_l * a_lf * dp_f_dx * dp_l_dx)
            - (K_f * a_lf * p_f * d2p_l_dx2)
            - p_f
        )

        # Mean squared residuals
        loss_physics = torch.mean(residual_rho_l**2) + torch.mean(residual_rho_f**2)
        return loss_physics

    def _compute_ic_loss(
        self, model_output: torch.Tensor, initial_conditions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute initial condition loss.

        Parameters
        ----------
        model_output : torch.Tensor
            Model predictions
        initial_conditions : torch.Tensor
            True initial conditions

        Returns
        -------
        torch.Tensor
            Initial condition loss
        """
        pred_at_t0 = model_output[:, :, :, 0]  # Predictions at t=0
        loss_ic = torch.mean((pred_at_t0 - initial_conditions) ** 2)
        return loss_ic

    def _compute_left_bc_loss(
        self, model_output: torch.Tensor, grid_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute left boundary condition loss.

        Parameters
        ----------
        model_output : torch.Tensor
            Model predictions
        grid_x : torch.Tensor
            Spatial grid coordinates

        Returns
        -------
        torch.Tensor
            Left boundary condition loss
        """
        p_l = model_output[:, 0, :, :]  # Leader density
        p_f = model_output[:, 1, :, :]  # Follower density

        D_l, D_f, a_lf, K_l, K_f, X = self.constants.values()

        # Compute spatial derivatives at left boundary using finite differences
        # Left boundary derivatives (forward difference)
        dp_l_dx_at_x0 = (p_l[:, 1, :] - p_l[:, 0, :]) / (
            grid_x[:, 1, :] - grid_x[:, 0, :]
        )
        dp_f_dx_at_x0 = (p_f[:, 1, :] - p_f[:, 0, :]) / (
            grid_x[:, 1, :] - grid_x[:, 0, :]
        )

        # Values at left boundary
        p_l_at_x0 = p_l[:, 0, :]
        p_f_at_x0 = p_f[:, 0, :]

        # Boundary condition residuals
        residual_l = (
            (D_f * dp_l_dx_at_x0)
            - 2 * (K_l * a_lf) * (p_l_at_x0 * dp_l_dx_at_x0)
            - (K_l * a_lf * p_l_at_x0 * dp_f_dx_at_x0)
            - X * p_l_at_x0
        )
        residual_f = (
            (D_l * p_f_at_x0)
            - 2 * (K_f * a_lf) * (p_f_at_x0 * dp_f_dx_at_x0)
            - (K_f * a_lf * p_f_at_x0 * dp_l_dx_at_x0)
        )

        loss_bc_left = torch.mean(residual_l**2) + torch.mean(residual_f**2)
        return loss_bc_left

    def _compute_right_bc_loss(
        self, model_output: torch.Tensor, target_value: float = 0.0
    ) -> torch.Tensor:
        """
        Compute right boundary condition loss.

        Parameters
        ----------
        model_output : torch.Tensor
            Model predictions
        target_value : float, optional
            Target value at right boundary, default is 0.0

        Returns
        -------
        torch.Tensor
            Right boundary condition loss
        """
        pred_at_xL = model_output[:, :, -1, :]  # Predictions at right boundary
        loss_bc_right = torch.mean((pred_at_xL - target_value) ** 2)
        return loss_bc_right

    def train(
        self,
        input_tensor: torch.Tensor,
        true_initial_conditions: torch.Tensor,
        grid_x: torch.Tensor,
        grid_t: torch.Tensor,
        n_epochs: int,
    ) -> nn.Module:
        """
        Execute the training loop.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor for the model
        true_initial_conditions : torch.Tensor
            True initial conditions for loss computation
        grid_x : torch.Tensor
            Spatial grid coordinates
        grid_t : torch.Tensor
            Time grid coordinates
        n_epochs : int
            Number of training epochs

        Returns
        -------
        nn.Module
            Trained model
        """
        self.logger.info("Starting FNO training")
        self.logger.info(f"Training for {n_epochs} epochs on {self.device}")

        start_time = time.time()

        for epoch in range(n_epochs):
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(input_tensor).to(self.device)

            # Compute individual losses
            loss_physics = self._compute_physics_loss(output, grid_x, grid_t)
            loss_ic = self._compute_ic_loss(output, true_initial_conditions)
            loss_bc_left = self._compute_left_bc_loss(output, grid_x)
            loss_bc_right = self._compute_right_bc_loss(output, target_value=0.0)

            # Combine losses with weights
            total_loss = (
                self.loss_weights["physics"] * loss_physics
                + self.loss_weights["ic"] * loss_ic
                + self.loss_weights["bc_left"] * loss_bc_left
                + self.loss_weights["bc_right"] * loss_bc_right
            )

            # Store loss history
            self.loss_history["total"].append(total_loss.item())
            self.loss_history["physics"].append(loss_physics.item())
            self.loss_history["ic"].append(loss_ic.item())
            self.loss_history["bc_left"].append(loss_bc_left.item())
            self.loss_history["bc_right"].append(loss_bc_right.item())

            # Backward pass and optimization
            total_loss.backward()
            self.optimizer.step()

            # Logging
            if (epoch + 1) % 50 == 0:
                self.logger.info(
                    f"Epoch [{epoch+1}/{n_epochs}], "
                    f"Total Loss: {total_loss.item():.6f}, "
                    f"Physics: {loss_physics.item():.6f}, "
                    f"IC: {loss_ic.item():.6f}, "
                    f"BC Left: {loss_bc_left.item():.6f}, "
                    f"BC Right: {loss_bc_right.item():.6f}"
                )

        end_time = time.time()
        training_time = end_time - start_time

        self.logger.info("Training completed")
        self.logger.info(f"Total training time: {training_time:.2f} seconds")
        self.logger.info(f"Final loss: {total_loss.item():.6f}")

        return self.model
