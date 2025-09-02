import logging
from datetime import datetime
import json
from mesa.batchrunner import batch_run, _collect_data
from multiprocessing import freeze_support
import pandas as pd

from model import ShipModel
from visualization import plot_simulation

# Configure logging
logging.basicConfig(
    filename="logs.log",
    filemode="w",  # Overwrites on each run; use "a" to append
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    freeze_support()
    
    config_file="config/config.json"
    with open(config_file) as f:
        config = json.load(f)

    steps = config["simulation_steps"]

    model_params = {
        "width": [config["width"]],
        "height": [config["height"]],
        "num_ships": [config["num_ships"]],
        "max_speed_range": [config["max_speed_range"]], 
        "speed_variation": [config["speed_variation"]],
        "directional_variation": [config["directional_variation"]],
        "ports": [config["ports"]],
        "speed_limit_zones": [config.get("speed_limit_zones", [])],
        "obstacles": [config["obstacles"]],
        "dwa_config": [config["dwa_config"]],
        "resolution": [config["resolution"]],
        "obstacle_threshold": [config["obstacle_threshold"]],
        "lookahead": [config["lookahead"]]
    }

    results = batch_run(
        ShipModel,
        parameters=model_params,
        iterations=1,
        max_steps=steps,
        number_processes=None,
        data_collection_period=1,
        display_progress=True,
    )

    results_df = pd.DataFrame(results)
    agent_df = results_df[["Step", "AgentID", "x", "y", "AStarPath"]].dropna()
    df = agent_df.reset_index()

    # Save to CSV
    df.to_csv("ship_simulation_output.csv", index=False)
    logging.info("Saved simulation data to ship_simulation_output.csv")

    # Run visualization
    plot_simulation(df, config)

    # model = ShipModel(
    #     width=config["width"],
    #     height=config["height"],
    #     num_ships=config["num_ships"],
    #     max_speed_range=config["max_speed_range"],
    #     speed_variation=config["speed_variation"],
    #     directional_variation=config["directional_variation"],
    #     ports=config["ports"],
    #     speed_limit_zones=config.get("speed_limit_zones", []),
    #     obstacles=config["obstacles"],
    #     dwa_config=config["dwa_config"],
    #     resolution=config["resolution"],
    #     obstacle_threshold=config["obstacle_threshold"],
    #     lookahead=config["lookahead"]
    # )

    # # Print agent counts by type
    # for agent_type, agents in model.agents_by_type.items():
    #     logging.info(f'{agent_type}: {len(agents)}')

    # # Run the simulation
    # logging.info(f"{datetime.now()} Starting ...")
    # for t in range(steps):
    #     logging.info(f"Step {t}")
    #     model.step()
    # logging.info(f"{datetime.now()} Finished.")

    # agent_df = model.datacollector.get_agent_vars_dataframe().dropna()
    # df = agent_df.reset_index()

    # # Save to CSV
    # df.to_csv("ship_simulation_output.csv", index=False)
    # logging.info("Saved simulation data to ship_simulation_output.csv")

    # # Run visualization
    # plot_simulation(df, config)