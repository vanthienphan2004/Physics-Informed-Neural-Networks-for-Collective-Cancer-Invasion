"""
Data loading module for the Cancer Invasion Simulation.

This module contains functions for generating synthetic data and creating
training datasets for the physics-informed neural network simulation.
"""

import torch
from typing import Tuple


def rho_l_initial(x: torch.Tensor) -> torch.Tensor:
    """
    Initial condition for leader species density.

    Parameters
    ----------
    x : torch.Tensor
        Spatial coordinates

    Returns
    -------
    torch.Tensor
        Initial density values for leader species (Gaussian bump)
    """
    return torch.exp(-100 * (x - 0.5)**2)  # Gaussian bump centered at x=0.5


def rho_f_initial(x: torch.Tensor) -> torch.Tensor:
    """
    Initial condition for follower species density.

    Parameters
    ----------
    x : torch.Tensor
        Spatial coordinates

    Returns
    -------
    torch.Tensor
        Initial density values for follower species (Gaussian bump)
    """
    return torch.exp(-100 * (x - 0.3)**2)  # Gaussian bump centered at x=0.3


def generate_IC_torch(
    x: torch.Tensor,
    height: float = 0.1,
    width: float = 0.01,
    center: float = 0.1
) -> torch.Tensor:
    """
    Generate initial condition using hyperbolic tangent function.

    Parameters
    ----------
    x : torch.Tensor
        Spatial coordinates
    height : float, optional
        Maximum height of the initial condition, default is 0.1
    width : float, optional
        Width parameter for the tanh function, default is 0.01
    center : float, optional
        Center position of the initial condition, default is 0.1

    Returns
    -------
    torch.Tensor
        Initial condition values
    """
    return height * (1 - torch.tanh((x - center) / width))


def coord_loader(
    count: int,
    device: torch.device,
    type: str = 'collocation'
) -> Tuple[torch.Tensor, ...]:
    """
    Generate coordinate data for different types of training points.

    Parameters
    ----------
    count : int
        Number of points to generate
    device : torch.device
        Device to place tensors on
    type : str, optional
        Type of coordinates: 'collocation', 'ic', or 'bc'

    Returns
    -------
    Tuple[torch.Tensor, ...]
        Generated coordinate tensors depending on type
    """
    if type == 'collocation':
        # Collocation points (interior of space-time domain)
        N_f = count
        x_f = torch.rand((N_f, 1), requires_grad=True).to(device)
        t_f = torch.rand((N_f, 1), requires_grad=True).to(device)
        coords = torch.cat((x_f, t_f), dim=1)
        return coords

    elif type == 'ic':
        # Initial condition points (at t = 0)
        N_ic = count
        x_ic = torch.linspace(0, 1, N_ic).reshape(-1, 1).to(device)
        t_ic = torch.zeros_like(x_ic).to(device)
        rho_l_ic_true = rho_l_initial(x_ic).to(device)
        rho_f_ic_true = rho_f_initial(x_ic).to(device)
        ic_coords = torch.cat((x_ic, t_ic), dim=1)
        ic_densities = torch.cat((rho_l_ic_true, rho_f_ic_true), dim=1)
        return ic_coords, ic_densities

    elif type == 'bc':
        # Boundary condition points
        N_bc = count
        t_bc = torch.linspace(0, 1, N_bc, requires_grad=True).reshape(-1, 1).to(device)
        x_right = torch.ones_like(t_bc, requires_grad=True).to(device)   # x = 1 (Dirichlet)
        x_left = torch.zeros_like(t_bc, requires_grad=True).to(device)   # x = 0 (Neumann)
        right_bc_coords = torch.cat((x_right, t_bc), dim=1)
        left_bc_coords = torch.cat((x_left, t_bc), dim=1)
        return right_bc_coords, left_bc_coords

    else:
        raise ValueError(f"Unsupported coordinate type: {type}")


def fno_data_loader(
    batch_size: int = 4,
    n_spatial_points: int = 64,
    n_time_steps: int = 32,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate data for Fourier Neural Operator training.

    Creates spatial-temporal grids and initial conditions for training
    the physics-informed FNO model.

    Parameters
    ----------
    batch_size : int, optional
        Number of batch samples, default is 4
    n_spatial_points : int, optional
        Number of spatial grid points, default is 64
    n_time_steps : int, optional
        Number of time steps, default is 32
    device : torch.device, optional
        Device to place tensors on, default is CPU

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        - grid_x: Spatial grid coordinates (batch_size, n_spatial_points, n_time_steps)
        - grid_t: Time grid coordinates (batch_size, n_spatial_points, n_time_steps)
        - input_tensor: Complete input tensor for FNO (batch_size, 4, n_spatial_points, n_time_steps)
        - true_initial_conditions: True initial conditions (batch_size, 2, n_spatial_points)
    """
    # Create spatial and temporal coordinate grids
    x_coords = torch.linspace(0, 1, n_spatial_points)
    t_coords = torch.linspace(0, 1, n_time_steps)
    grid_x_base, grid_t_base = torch.meshgrid(x_coords, t_coords, indexing='ij')

    # Expand to batch dimension
    grid_x = grid_x_base.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    grid_t = grid_t_base.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    grid_x.requires_grad_(True)
    grid_t.requires_grad_(True)

    # Generate initial conditions for both species
    initial_a = (generate_IC_torch(x_coords, height=0.1, width=0.01, center=0.1)
                .unsqueeze(0).repeat(batch_size, 1).to(device))  # Leader initial condition
    initial_b = (generate_IC_torch(x_coords, height=0.4, width=0.01, center=0.1)
                .unsqueeze(0).repeat(batch_size, 1).to(device))  # Follower initial condition

    # Stack initial conditions for loss computation
    true_initial_conditions = torch.stack([initial_a, initial_b], dim=1).to(device)

    # Expand initial conditions across time dimension for input tensor
    initial_a_expanded = initial_a.unsqueeze(-1).repeat(1, 1, n_time_steps)
    initial_b_expanded = initial_b.unsqueeze(-1).repeat(1, 1, n_time_steps)

    # Create complete input tensor: [x_grid, t_grid, initial_a, initial_b]
    input_tensor = torch.stack(
        [grid_x, grid_t, initial_a_expanded, initial_b_expanded],
        dim=1
    ).to(device)

    return grid_x, grid_t, input_tensor, true_initial_conditions
