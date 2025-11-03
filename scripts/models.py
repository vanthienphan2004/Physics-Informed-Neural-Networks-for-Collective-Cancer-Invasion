"""
Neural network models for the Cancer Invasion Simulation.

This module contains the physics-informed neural network (PINN) and
Fourier Neural Operator (FNO) models used for simulating cancer invasion
dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import TFNO
from typing import Tuple


class MLP(nn.Module):
    """
    Basic Multi-Layer Perceptron model.

    A simple feedforward neural network with configurable layers and activation.

    Parameters
    ----------
    input_dim : int
        Dimension of input features
    output_dim : int
        Dimension of output features
    hidden_dim : int
        Dimension of hidden layers
    num_layers : int
        Number of layers in the network
    activation : nn.Module, optional
        Activation function to use, default is Tanh
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: nn.Module = nn.Tanh()
    ) -> None:
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.net(x)


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for coupled species modeling.

    This model uses coordinate encoding and manifold networks to predict
    the densities of leader and follower species in cancer invasion dynamics.
    """

    def __init__(
        self,
        coord_dim: int = 2,
        output_dim: int = 1,
        coord_encoder_hidden: int = 128,
        coord_encoder_layers: int = 4,
        manifold_net_hidden: int = 128,
        manifold_net_layers: int = 4
    ) -> None:
        """
        Initialize the PINN model.

        Parameters
        ----------
        coord_dim : int, optional
            Dimension of coordinate input (x, t), default is 2
        output_dim : int, optional
            Dimension of output per species, default is 1
        coord_encoder_hidden : int, optional
            Hidden dimension for coordinate encoder, default is 128
        coord_encoder_layers : int, optional
            Number of layers in coordinate encoder, default is 4
        manifold_net_hidden : int, optional
            Hidden dimension for manifold networks, default is 128
        manifold_net_layers : int, optional
            Number of layers in manifold networks, default is 4
        """
        super(PINN, self).__init__()

        self.type_derivative = "AD"
        self.coord_dim = coord_dim
        self.output_dim = output_dim

        activation = nn.Tanh()

        # Coordinate Encoder (processes x, t)
        self.coord_encoder = MLP(
            coord_dim,
            coord_encoder_hidden,
            coord_encoder_hidden,
            coord_encoder_layers,
            activation
        )

        # Manifold Networks for Leader and Follower species
        self.manifold_net_L = MLP(
            coord_encoder_hidden,
            output_dim,
            manifold_net_hidden,
            manifold_net_layers,
            activation
        )
        self.manifold_net_F = MLP(
            coord_encoder_hidden,
            output_dim,
            manifold_net_hidden,
            manifold_net_layers,
            activation
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the PINN.

        Parameters
        ----------
        coords : torch.Tensor
            Input coordinates of shape (batch_size, coord_dim)

        Returns
        -------
        torch.Tensor
            Predicted densities for leader and follower species
            Shape: (batch_size, 2 * output_dim)
        """
        coord_embedding = self.coord_encoder(coords)

        # Predict densities for both species
        manifold_embedding_L = self.manifold_net_L(coord_embedding)
        manifold_embedding_F = self.manifold_net_F(coord_embedding)

        # Concatenate outputs: [rho_L, rho_F]
        solution = torch.cat([manifold_embedding_L, manifold_embedding_F], dim=1)
        return solution

    @staticmethod
    def get_derivative_type() -> str:
        """Return the derivative computation method."""
        return "AD"


class CoupledSpeciesTFNO(nn.Module):
    """
    Tensorized Fourier Neural Operator for coupled species system.

    This model learns to map initial conditions of two species to their
    space-time density evolution using Fourier neural operators.

    Parameters
    ----------
    modes : int, optional
        Number of Fourier modes, default is 16
    width : int, optional
        Width of hidden layers, default is 64
    n_layers : int, optional
        Number of TFNO layers, default is 4
    """

    def __init__(self, modes: int = 16, width: int = 64, n_layers: int = 4) -> None:
        super(CoupledSpeciesTFNO, self).__init__()

        self.in_channels = 4  # x, t, a(x,0), b(x,0)
        self.out_channels = 2  # p_l(x,t), p_f(x,t)

        # TFNO model for efficient PDE solving
        self.fno = TFNO(
            n_modes=(modes, modes),
            hidden_channels=width,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_layers=n_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TFNO model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, num_x_points, num_t_points)
            Channels: [x_coord, t_coord, p_l_initial, p_f_initial]

        Returns
        -------
        torch.Tensor
            Predicted densities of shape (batch_size, out_channels, num_x_points, num_t_points)
        """
        x_coord = x[:, 0:1, :, :]      # Spatial grid
        p_l_initial = x[:, 2:3, :, :]  # Leader initial condition
        p_f_initial = x[:, 3:4, :, :]  # Follower initial condition

        # Apply FNO
        fno_output = self.fno(x)

        # Extract FNO outputs for each species
        fno_output_p_l = fno_output[:, 0:1, :, :]
        fno_output_p_f = fno_output[:, 1:2, :, :]

        # Combine with initial conditions and spatial coordinate
        final_output_p_l = p_l_initial + x_coord * fno_output_p_l
        final_output_p_f = p_f_initial + x_coord * fno_output_p_f

        # Concatenate final outputs
        final_output = torch.cat([final_output_p_l, final_output_p_f], dim=1)

        return final_output
