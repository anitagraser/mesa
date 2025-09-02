import logging
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from shapely.geometry import Polygon
import random
from shapely.strtree import STRtree
import math

from agents.ship import Ship
from agents.port import Port
from agents.obstacle import Obstacle
from a_star import create_occupancy_grid

class ShipModel(Model):
    def __init__(self, width, height, num_ships, max_speed_range, speed_variation, directional_variation, ports, speed_limit_zones, obstacles, dwa_config, resolution=1, obstacle_threshold=0, lookahead=3.0):
        super().__init__()
        # self.step = 0
        self.width = width
        self.height = height
        self.max_speed_range = max_speed_range
        self.speed_variation = speed_variation
        self.directional_variation = directional_variation
        self.dwa_config = dwa_config
        self.resolution = resolution
        self.obstacle_threshold = obstacle_threshold
        self.lookahead = lookahead

        if speed_variation["enabled"]:
            self.max_speed_variation = speed_variation["max_speed_variation"]
        if directional_variation["enabled"]:
            self.max_heading_deviation = math.radians(directional_variation["max_deviation_deg"])
            self.deviation_duration = directional_variation["duration_steps"]
            self.deviation_chance = directional_variation["activation_chance"]

        self.space = ContinuousSpace(self.width, self.height, torus=False)
        self.schedule = RandomActivation(self)

        self.speed_limit_zones = speed_limit_zones

        # Create obstacles
        self.obstacle_agents = self.create_obstacles(obstacles)
        self.buffered_obstacles = [obs.shape.buffer(self.dwa_config["robot_radius"]) for obs in self.obstacle_agents]
        self.obstacle_tree = STRtree(self.buffered_obstacles)

        # Generate ports if none are provided
        if not ports:
            ports = self.generate_random_ports()

        # Create and place ports
        self.port_agents = self.create_ports(ports)

        # Generate occupancy grid
        self.occupancy_grid = create_occupancy_grid(self.width, self.height, self.resolution, self.obstacle_agents, self.obstacle_threshold)

        # Create ships, placing them at ports
        self.create_ships(num_ships, self.port_agents)

        self.datacollector = DataCollector(
            agent_reporters={
                "x": lambda a: a.pos[0] if isinstance(a, Ship) and a.pos[0] else None,
                "y": lambda a: a.pos[1] if isinstance(a, Ship) and a.pos[1] else None,
                "AStarPath": lambda a: a.global_path if isinstance(a, Ship) and a.global_path else None
            }
        )
        self.datacollector.collect(self) 

    def generate_random_ports(self, min_ports=2, max_ports=5):
        """Generate random ports in the simulation space."""
        num_ports = random.randint(min_ports, max_ports)
        return [(random.uniform(0, self.width), random.uniform(0, self.height)) for _ in range(num_ports)]

    def create_ports(self, ports):
        """Create and place port agents"""
        port_agents = []
        for (x, y) in ports:
            port = Port(self)
            self.space.place_agent(port, (x, y))
            self.schedule.add(port)
            port_agents.append(port)
        return port_agents
    
    def create_ships(self, num_ships, port_agents):
        """Create ship agents and assign them a start and destination port."""
        for id in range(num_ships):
            start_port = random.choice(port_agents)
            ship = Ship(self, id, start_port, port_agents, self.dwa_config)
            self.space.place_agent(ship, start_port.pos)
            self.schedule.add(ship)

    def create_obstacles(self, obstacles):
        """Create and place Polygon obstacle agents in the space"""
        obstacle_agents = []
        for corners in obstacles:
            obstacle_shape = Polygon(corners)
            obstacle = Obstacle(self, obstacle_shape)
            self.schedule.add(obstacle)
            obstacle_agents.append(obstacle)

            # Fill grid with obstacles
            # Generates many warnings
            # min_x, min_y, max_x, max_y = map(int, obstacle_shape.bounds)
            # for x in range(min_x, max_x + 1):
            #     for y in range(min_y, max_y + 1):
            #         if obstacle_shape.contains(Point(x, y)):
            #             self.space.place_agent(obstacle, (x, y))

        return obstacle_agents

    def step(self):
        """Run one step of the model.
        
        All agents are activated in random order."""
        # logging.info(f"Step {self.step}")
        self.schedule.step()
        self.datacollector.collect(self)
        # self.step += 1