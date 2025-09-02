import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as MplPolygon

def plot_ports(ax, ports):
    """Plot ports on the map."""
    for x, y in ports:
        ax.scatter(x, y, color='blue', marker='s', label="Port" if "Port" not in ax.get_legend_handles_labels()[1] else "")

def plot_obstacles(ax, obstacles):
    """Plot islands as brown polygons."""
    for island in obstacles:
        island_shape = Polygon(island)
        x, y = island_shape.exterior.xy
        ax.fill(x, y, color="brown", alpha=0.6, label="Island" if "Island" not in ax.get_legend_handles_labels()[1] else "")

def get_distinct_colors(n):
    """Generate distinct colors for plotting speed limit zones."""
    colors = list(mcolors.TABLEAU_COLORS.values())  # Use Tableau colors for distinct shades
    if n <= len(colors):
        return colors[:n]
    else:
        return plt.cm.get_cmap("tab10", n).colors  # Use colormap for more colors if needed

def plot_speed_limit_zones(ax, speed_limit_zones):
    """Plot speed limit zones with different colors and unique legend entries."""
    colors = get_distinct_colors(len(speed_limit_zones))

    for idx, zone in enumerate(speed_limit_zones):
        zone_points = zone["zone"]
        max_speed = zone["max_speed"]
        color = colors[idx % len(colors)]  # Cycle through colors if needed

        label = f"Speed Limit {max_speed}"

        # Draw speed limit zone as a semi-transparent polygon
        polygon_patch = MplPolygon(zone_points, closed=True, color=color, alpha=0.3, label=label)
        ax.add_patch(polygon_patch)

def plot_ship_trajectories(ax, df):
    """Plot ship trajectories and A* paths."""
    a_star_legend_added = False

    for agent_id in df["AgentID"].unique():
        ship_data = df[df["AgentID"] == agent_id]

        # Extract A* path from the first entry
        a_star_path = ship_data.iloc[0]["AStarPath"]

        # Only label the first A* path
        a_star_label = "A* Path" if not a_star_legend_added else None
        a_star_legend_added = True  

        # Plot A* path as a dashed line
        a_star_x, a_star_y = zip(*a_star_path)
        ax.plot(a_star_x, a_star_y, linestyle="dashed", color="gray", alpha=0.7, label=a_star_label)

        # Plot ship trajectory as a solid line
        ax.plot(ship_data["x"], ship_data["y"], linestyle="-", alpha=0.7, label=f"Ship {agent_id}")

def setup_plot():
    """Initialize and return the plot figure and axis."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("DWA Ship Movement Trajectories and A* Paths")
    ax.grid(True)
    return fig, ax

def save_plot(fig, output_image):
    """Save the plot with a tight layout."""
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    fig.savefig(output_image, bbox_inches="tight")
    print(f"Saved simulation plot to {output_image}")

def plot_simulation(df, config, output_image="simulation_plot.png"):
    """Main function to generate the simulation plot."""
    fig, ax = setup_plot()

    # Extract configuration data
    ports = config["ports"]
    obstacles = config["obstacles"]
    speed_limit_zones = config.get("speed_limit_zones", [])

    # Plot components
    plot_ports(ax, ports)
    plot_obstacles(ax, obstacles)
    plot_speed_limit_zones(ax, speed_limit_zones)
    plot_ship_trajectories(ax, df)

    # Save the final plot
    save_plot(fig, output_image)
