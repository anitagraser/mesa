"""
ship.py — optimized
- Fast local-goal search (squared distances, incremental index)
- Lazy STRtree index for speed-limit zones (supports both index- and geometry-returns)
- Cheaper docking check (no sqrt)
- Zero-allocation move_position + DEBUG-only logging
"""
import logging
import math
from numbers import Integral
from mesa import Agent
from shapely.geometry import Point, Polygon
try:
    from shapely.strtree import STRtree
except Exception:  # STRtree may be unavailable on very old Shapely or missing GEOS
    STRtree = None

from a_star import astar
from dynamic_window_approach import dwa_control, motion

log = logging.getLogger(__name__)


def _get_rng(agent):
    """Return an RNG: prefer agent.random, then model.random, else Python's random module."""
    try:
        import random as _pr
    except Exception:  # pragma: no cover
        _pr = None
    return getattr(agent, "random", None) or getattr(agent.model, "random", None) or _pr


class Ship(Agent):
    def __init__(self, model, id, start_port, all_ports, dwa_config):
        # Mesa 2.x signature: Agent(unique_id, model)
        super().__init__(id, model)
        self.destination_port = self.assign_destination(all_ports, start_port)
        self.global_path = self.calculate_global_path(start_port.pos, self.destination_port.pos)
        # Per-ship copy so edits (speed limits/noise) don't mutate a shared dict
        self.dwa_config = dict(dwa_config)

        self.heading_deviation = 0.0
        self.heading_drift_duration = 0

        rng = _get_rng(self)
        # Assign a random max speed within the speed range
        self.dwa_config["max_speed"] = rng.uniform(self.model.max_speed_range[0], self.model.max_speed_range[1])
        self.original_max_speed = self.dwa_config["max_speed"]
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Ship %s has a maximum speed of %.3f.", self.unique_id, self.dwa_config["max_speed"])

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
        # Early exit if already docked (relies on model to have set .pos initially)
        if getattr(self, "pos", None) is not None:
            if self.pos[0] == self.destination_port.pos[0] and self.pos[1] == self.destination_port.pos[1]:
                return

        # Move the ship along the calculated global path.
        if self.global_path and len(self.global_path) > 1:
            dt = self.dwa_config["dt"]

            local_goal = self.get_local_goal(self.state, self.global_path, lookahead=self.model.lookahead)
            # Speed limit via spatial index (built lazily & cached on model)
            self.dwa_config["max_speed"] = self.get_speed_limit()

            if self.model.speed_variation["enabled"]:
                self.dwa_config["max_speed"] = self.get_noisy_speed()

            rng = _get_rng(self)
            # Directional (heading) variation
            if self.model.directional_variation["enabled"]:
                # Randomly trigger heading deviation
                if self.heading_drift_duration > 0:
                    # Continue existing deviation
                    noisy_theta = self.state[2] + self.heading_deviation
                    self.heading_drift_duration -= 1
                    if log.isEnabledFor(logging.DEBUG):
                        log.debug("Continue deviation. Ship %s, Theta=%.4f", self.unique_id, self.heading_deviation)
                else:
                    # Random chance to start a new deviation
                    if rng.random() < self.model.deviation_chance:
                        self.heading_deviation = rng.uniform(-self.model.max_heading_deviation, self.model.max_heading_deviation)
                        self.heading_drift_duration = self.model.deviation_duration
                        noisy_theta = self.state[2] + self.heading_deviation
                        if log.isEnabledFor(logging.DEBUG):
                            log.debug("Starting directional deviation. Ship %s, Theta=%.4f", self.unique_id, self.heading_deviation)
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
                self.state = motion(self.state, control[0], control[1], dt)
                self.move_position()

            # After motion, restore true max speed
            self.dwa_config["max_speed"] = self.original_max_speed

    def get_noisy_speed(self):
        # Assign ± speed variation as a fraction of max_speed
        max_speed_variation = self.model.max_speed_variation
        variation_amount = self.dwa_config["max_speed"] * max_speed_variation
        rng = _get_rng(self)
        noisy_speed = self.dwa_config["max_speed"] + rng.uniform(-variation_amount, variation_amount)

        # Clamp to valid range
        noisy_speed = max(self.dwa_config["min_speed"], min(noisy_speed, self.dwa_config["max_speed"]))
        return noisy_speed

    def should_dock(self, current_speed):
        """Determine if the ship should dock at its destination (squared-distance, no sqrt)."""
        dx = self.state[0] - self.destination_port.pos[0]
        dy = self.state[1] - self.destination_port.pos[1]
        dist2 = dx * dx + dy * dy
        step_move = current_speed * self.dwa_config['dt']
        return dist2 <= step_move * step_move

    def move_position(self):
        """Update the ship's position in the simulation space (avoids numpy allocation)."""
        self.pos = (float(self.state[0]), float(self.state[1]))
        self.model.space.move_agent(self, self.pos)
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "Ship %s moving towards %s. Current position: (%.3f, %.3f). Speed: %.3f.",
                self.unique_id, self.destination_port.pos, self.pos[0], self.pos[1], self.state[3]
            )

    def move_to_destination(self):
        """Move the ship directly to its destination."""
        self.pos = self.destination_port.pos
        self.model.space.move_agent(self, self.destination_port.pos)
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Ship %s arrived at port %s.", self.unique_id, self.destination_port.pos)

    def assign_destination(self, all_ports, start_port):
        """Select a destination port different from the starting port."""
        possible_destinations = [port for port in all_ports if port != start_port]
        rng = _get_rng(self)
        return rng.choice(possible_destinations) if possible_destinations else start_port

    def calculate_global_path(self, start, destination):
        """Calculate a path using A* algorithm."""
        grid_start = (int(start[0] / self.model.resolution), int(start[1] / self.model.resolution))
        grid_goal = (int(destination[0] / self.model.resolution), int(destination[1] / self.model.resolution))
        global_path_indices = astar(self.model.occupancy_grid, grid_start, grid_goal)

        if global_path_indices is None:
            if log.isEnabledFor(logging.DEBUG):
                log.debug("No global path for ship %s.", self.unique_id)
            return

        global_path = [((i) * self.model.resolution, (j) * self.model.resolution)
                       for (i, j) in global_path_indices]
        return global_path

    # --- Speed-limit zones: lazy-built STRtree on the model ---
    def _build_speed_zone_index_if_needed(self):
        """Build and cache a spatial index for speed-limit zones on the model (lazy)."""
        if getattr(self.model, "_speed_zone_tree", None) is not None:
            return  # already built
        zones = getattr(self.model, "speed_limit_zones", None)
        if not zones:
            # no zones configured
            self.model._speed_zone_tree = None
            self.model._speed_zone_polys = []
            self.model._speed_zone_speed_by_id = {}
            self.model._speed_zone_speeds = []
            return

        polys = [Polygon(z["zone"]) for z in zones]
        self.model._speed_zone_polys = polys
        # store speeds in parallel list for index-based lookups
        self.model._speed_zone_speeds = [z["max_speed"] for z in zones]

        if STRtree is not None and polys:
            try:
                self.model._speed_zone_tree = STRtree(polys)
            except Exception:
                self.model._speed_zone_tree = None
        else:
            self.model._speed_zone_tree = None  # fallback path will be used
        # map polygon id -> speed for geometry-based lookups
        self.model._speed_zone_speed_by_id = {id(p): z["max_speed"] for p, z in zip(polys, zones)}

    def get_speed_limit(self):
        """Return max speed at current position.
        Works with STRtree returning indices or geometries.
        """
        zones = getattr(self.model, "speed_limit_zones", None)
        if not zones:
            return self.original_max_speed

        self._build_speed_zone_index_if_needed()

        pt = Point(self.state[0], self.state[1])
        tree = getattr(self.model, "_speed_zone_tree", None)

        if tree is not None:
            # Try to use a predicate if supported (Shapely 2.x). It may return indices.
            try:
                candidates = tree.query(pt, predicate="contains")
            except TypeError:
                # Older Shapely: no predicate argument
                candidates = tree.query(pt)

            # `candidates` can be a list of indices OR a list of Polygon objects.
            for cand in candidates:
                if isinstance(cand, Integral):            # index path (e.g., numpy.int64)
                    idx = int(cand)
                    poly = self.model._speed_zone_polys[idx]
                    if poly.contains(pt):
                        return self.model._speed_zone_speeds[idx]
                else:                                     # geometry path
                    poly = cand
                    if poly.contains(pt):
                        return self.model._speed_zone_speed_by_id.get(id(poly), self.original_max_speed)

            return self.original_max_speed

        # Fallback: no STRtree available; scan the prebuilt polygons once
        for idx, poly in enumerate(self.model._speed_zone_polys):
            if poly.contains(pt):
                return self.model._speed_zone_speeds[idx]

        return self.original_max_speed

    def get_local_goal(self, state, global_path, lookahead=3.0):
        """Return the first waypoint at least 'lookahead' away (squared distance, O(1–few))."""
        x, y = state[0], state[1]
        la2 = lookahead * lookahead
        i = self.current_wp_idx
        last = len(global_path) - 1
        while i < last:
            dx = global_path[i][0] - x
            dy = global_path[i][1] - y
            if dx * dx + dy * dy > la2:
                break
            i += 1
        self.current_wp_idx = i
        return global_path[i]
