"""
Entry point for the Cancer Invasion Simulation application.

This module serves as the main entry point for the physics-informed neural
network simulation of cancer invasion dynamics. It loads the configuration,
instantiates the simulation controller, and executes the workflow.
"""

import json
import os
import sys
from controller import SimulationController


def main() -> None:
    """
    Main function to run the cancer invasion simulation.

    Loads configuration from config.json, creates the simulation controller,
    and executes the complete workflow.
    """
    # Get the directory where main.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: config.json file not found at {config_path}. Please ensure it exists in the same directory as main.py.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config.json: {e}")
        sys.exit(1)

    try:
        controller = SimulationController(config)
        controller.run()
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
