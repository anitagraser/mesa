import logging
import numpy as np
from mesa import Agent
import math
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from a_star import astar
from dynamic_window_approach import dwa_control, motion

class Ship(Agent):
    def __init__(self, model, id, start_port, all_ports, dwa_config):
        super().__init__(self, model)
        self.unique_id = id
        self.destination_port = self.assign_destination(all_ports, start_port)
        self.global_path = self.calculate_global_path(start_port.pos, self.destination_port.pos)
        self.dwa_config = dwa_config
        
        self.heading_deviation = 0.0
        self.heading_drift_duration = 0
        
        # Assign a random max speed within the speed range
        dwa_config["max_speed"] = self.random.uniform(self.model.max_speed_range[0], self.model.max_speed_range[1])
        self.original_max_speed = self.dwa_config["max_speed"]
        logging.info(f"Ship {self.unique_id} has a maximum speed of {dwa_config['max_speed']}.")

        if self.global_path and len(self.global_path) > 1:
            first_waypoint = self.global_path[1]  # Ensure it doesn't use the port position
            dx = first_waypoint[0] - start_port.pos[0]
            dy = first_waypoint[1] - start_port.pos[1]
    
            # Set initial heading (theta) towards the first waypoint
            initial_theta = math.atan2(dy, dx)
        else:
            initial_theta = 0.0

        # Ship's state (x, y, theta, v, w)
        self.state = (start_port.pos[0], start_port.pos[1], initial_theta, 0.0, 0.0)

        self.current_wp_idx = 0

    def step(self):
        if self.pos[0] == self.destination_port.pos[0] and self.pos[1] == self.destination_port.pos[1]:
            return

        """Move the ship along the calculated global path."""
        if self.global_path and len(self.global_path) > 1:
            local_goal = self.get_local_goal(self.state, self.global_path, lookahead=self.model.lookahead)
            self.dwa_config["max_speed"] = self.get_speed_limit()

            if self.model.speed_variation["enabled"]:
                self.dwa_config["max_speed"] = self.get_noisy_speed()

            # if self.model.directional_variation["enabled"]:
            #     noisy_state = self.get_noisy_state()  
            # else:
            #     noisy_state = self.state                         

            if self.model.directional_variation["enabled"]:
                # Randomly trigger heading deviation
                if self.heading_drift_duration > 0:
                    # Continue existing deviation
                    noisy_theta = self.state[2] + self.heading_deviation
                    self.heading_drift_duration -= 1
                    logging.info(f"Continue deviation. Ship {self.unique_id}, Theta = {self.heading_deviation}")
                else:
                    # Random chance to start a new deviation
                    if self.random.random() < self.model.deviation_chance:
                        self.heading_deviation = self.random.uniform(-self.model.max_heading_deviation, self.model.max_heading_deviation)
                        self.heading_drift_duration = self.model.deviation_duration
                        noisy_theta = self.state[2] + self.heading_deviation
                        logging.info(f"Starting directional deviation. Ship {self.unique_id}, Theta = {self.heading_deviation}")
                    else:
                        noisy_theta = self.state[2]

                # Normalize heading
                noisy_theta = (noisy_theta + math.pi) % (2 * math.pi) - math.pi
                noisy_state = (self.state[0], self.state[1], noisy_theta, self.state[3], self.state[4])   
            else:
                noisy_state = self.state         

            control, predicted_trajectory, cost_info = dwa_control(
                noisy_state, self.dwa_config, self.model.obstacle_tree, 
                self.model.buffered_obstacles, local_goal
            )

            # Check if we should dock
            if self.should_dock(control[0]):
                self.move_to_destination()
            else:
                self.state = motion(self.state, control[0], control[1], self.dwa_config["dt"])
                self.move_position()
            
            # After motion, restore true max speed
            self.dwa_config["max_speed"] = self.original_max_speed

    def get_noisy_speed(self):
            # Assign ± speed variation as a fraction of max_speed
            max_speed_variation = self.model.max_speed_variation
            variation_amount = self.dwa_config["max_speed"] * max_speed_variation
            noisy_speed = self.dwa_config["max_speed"] + self.random.uniform(-variation_amount, variation_amount)

            # Clamp to valid range
            noisy_speed = max(self.dwa_config["min_speed"], min(noisy_speed, self.dwa_config["max_speed"]))
            return noisy_speed

    def should_dock(self, current_speed):
        """Determine if the ship should dock at its destination."""
        remaining_distance = math.hypot(
            self.state[0] - self.destination_port.pos[0], 
            self.state[1] - self.destination_port.pos[1]
        )

        # Dock if the remaining distance is smaller than what the next step would move
        if remaining_distance <= current_speed*self.dwa_config['dt']:
            return True

        return False

    def move_position(self):
        """Update the ship's position in the simulation space."""
        self.pos = (float(self.state[0]), float(self.state[1]))
        new_pos = np.array([self.pos[0], self.pos[1]])
        self.model.space.move_agent(self, new_pos)
        logging.info(
                f"Ship {self.unique_id} moving towards {self.destination_port.pos}. "
                f"Current position: {new_pos}."
                f"Speed: {self.state[3]}."
            )
    
    def move_to_destination(self):    
        """Move the ship directly to its destination."""
        self.pos = self.destination_port.pos
        self.model.space.move_agent(self, self.destination_port.pos)
        logging.info(
                f"Ship {self.unique_id} arrived at port {self.destination_port.pos}."
            )   
    
    def assign_destination(self, all_ports, start_port):
        """Select a destination port different from the starting port."""
        possible_destinations = [port for port in all_ports if port != start_port]
        return self.random.choice(possible_destinations) if possible_destinations else start_port

    def calculate_global_path(self, start, destination):
        """Calculate a path using A* algorithm."""
        grid_start = (int(start[0] / self.model.resolution), int(start[1] / self.model.resolution))
        grid_goal = (int(destination[0] / self.model.resolution), int(destination[1] / self.model.resolution))
        global_path_indices = astar(self.model.occupancy_grid, grid_start, grid_goal)

        if global_path_indices is None:
            logging.info(f"No global path for ship {self.unique_id}.")
            return
        
        global_path = [((i) * self.model.resolution, (j) * self.model.resolution)
                       for (i, j) in global_path_indices]
        return global_path
    
    def get_speed_limit(self):
        """Check if the ship is inside a speed-limited zone and return the max speed."""
        ship_position = Point(self.state[0], self.state[1])

        for zone in self.model.speed_limit_zones:
            polygon = Polygon(zone["zone"])
            if polygon.contains(ship_position):
                return zone["max_speed"]  # Apply the speed limit

        return self.original_max_speed  # Restore original speed if outside all zones
    
    def get_local_goal(self, state, global_path, lookahead=3.0):
        """
        Given the robot's current state and the global path, return a local goal.
        Returns a tuple (local_goal, new_wp_idx) where local_goal is a waypoint (x, y)
        that is at least 'lookahead' distance ahead of the current position.
        """
        x, y = state[0], state[1]
        for idx in range(self.current_wp_idx, len(global_path)):
            wp = global_path[idx]
            if math.hypot(wp[0] - x, wp[1] - y) > lookahead:
                self.current_wp_idx = idx
                return wp
        self.current_wp_idx = len(global_path) - 1
        return global_path[-1]
