import logging
import math
import random
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from agents.ship import Ship
from agents.port import Port
from agents.obstacle import Obstacle
from a_star import create_occupancy_grid

class ShipModel(Model):
    def __init__(self, width, height, num_ships, max_speed_range, speed_variation, directional_variation, ports,
                 speed_limit_zones, obstacles, dwa_config, resolution=1, obstacle_threshold=0, lookahead=3.0,
                 collect_every: int = 10):
        super().__init__()
        self.width = width
        self.height = height
        self.max_speed_range = max_speed_range
        self.speed_variation = speed_variation
        self.directional_variation = directional_variation
        self.dwa_config = dwa_config
        self.resolution = resolution
        self.obstacle_threshold = obstacle_threshold
        self.lookahead = lookahead
        self.collect_every = max(1, int(collect_every))

        if speed_variation.get("enabled", False):
            self.max_speed_variation = speed_variation["max_speed_variation"]
        if directional_variation.get("enabled", False):
            self.max_heading_deviation = math.radians(directional_variation["max_deviation_deg"])
            self.deviation_duration = directional_variation["duration_steps"]
            self.deviation_chance = directional_variation["activation_chance"]

        self.space = ContinuousSpace(self.width, self.height, torus=False)
        self.schedule = RandomActivation(self)

        self.speed_limit_zones = speed_limit_zones

        # Create obstacles & spatial index
        self.obstacle_agents = self.create_obstacles(obstacles)
        self.buffered_obstacles = [obs.shape.buffer(self.dwa_config["robot_radius"]) for obs in self.obstacle_agents]
        self.obstacle_tree = STRtree(self.buffered_obstacles)

        # Generate ports if none are provided
        if not ports:
            ports = self.generate_random_ports()

        # Create and place ports
        self.port_agents = self.create_ports(ports)

        # Generate occupancy grid
        self.occupancy_grid = create_occupancy_grid(self.width, self.height, self.resolution,
                                                    self.obstacle_agents, self.obstacle_threshold)

        # Create ships, placing them at ports
        self.create_ships(num_ships, self.port_agents)

        # Snapshot static A* paths once (don’t store every step)
        self.paths_snapshot = {
            a.unique_id: a.global_path
            for a in self.schedule.agents
            if isinstance(a, Ship) and a.global_path
        }

        # Lightweight DataCollector (fast attribute lookups; no per-step A* paths)
        self.datacollector = DataCollector(
            agent_reporters={
                "x": "x",
                "y": "y",
                "v": "v",
            }
        )
        self.datacollector.collect(self)

        # queue for ships that arrived this step (populated by Ship._queue_removal)
        self._arrived_to_remove = []

    def generate_random_ports(self, min_ports=2, max_ports=5):
        num_ports = random.randint(min_ports, max_ports)
        return [(random.uniform(0, self.width), random.uniform(0, self.height)) for _ in range(num_ports)]

    def create_ports(self, ports):
        port_agents = []
        for (x, y) in ports:
            port = Port(self)
            self.space.place_agent(port, (x, y))
            self.schedule.add(port)
            port_agents.append(port)
        return port_agents

    def create_ships(self, num_ships, port_agents):
        for id in range(num_ships):
            start_port = random.choice(port_agents)
            ship = Ship(self, id, start_port, port_agents, self.dwa_config)
            self.space.place_agent(ship, start_port.pos)
            self.schedule.add(ship)

    def create_obstacles(self, obstacles):
        obstacle_agents = []
        for corners in obstacles:
            obstacle_shape = Polygon(corners)
            obstacle = Obstacle(self, obstacle_shape)
            self.schedule.add(obstacle)
            obstacle_agents.append(obstacle)
        return obstacle_agents

    def step(self):
        """Run one step of the model (random order for scheduled agents)."""
        self.schedule.step()

        # Safely remove ships that arrived in their step
        if self._arrived_to_remove:
            for ship in self._arrived_to_remove:
                try:
                    self.schedule.remove(ship)
                except Exception:
                    pass
            self._arrived_to_remove = []

        # Collect sparsely
        if self.schedule.steps % self.collect_every == 0:
            self.datacollector.collect(self)
