"""
ship.py — turbo (dataset-speed edition)
- Fast local-goal search (squared distances, incremental index)
- Lazy STRtree speed-zone lookup (indices or geometries)
- Cheaper docking check (no sqrt) + *early* docking precheck (skips DWA when close)
- Adds lightweight x/y/v properties for fast DataCollector
- Queues arrived ships for removal (model cleans them up after the step)
"""
import logging
import math
from numbers import Integral
from mesa import Agent
from shapely.geometry import Point, Polygon
try:
    from shapely.strtree import STRtree
except Exception:  # STRtree may be unavailable
    STRtree = None

from a_star import astar
from dynamic_window_approach import dwa_control, motion

log = logging.getLogger(__name__)


def _get_rng(agent):
    try:
        import random as _pr
    except Exception:
        _pr = None
    return getattr(agent, "random", None) or getattr(agent.model, "random", None) or _pr


class Ship(Agent):
    def __init__(self, model, id, start_port, all_ports, dwa_config):
        super().__init__(id, model)
        self.destination_port = self.assign_destination(all_ports, start_port)
        self.global_path = self.calculate_global_path(start_port.pos, self.destination_port.pos)
        self.dwa_config = dict(dwa_config)  # per-ship copy

        self.heading_deviation = 0.0
        self.heading_drift_duration = 0

        rng = _get_rng(self)
        self.dwa_config["max_speed"] = rng.uniform(self.model.max_speed_range[0], self.model.max_speed_range[1])
        self.original_max_speed = self.dwa_config["max_speed"]
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Ship %s max speed %.3f.", self.unique_id, self.dwa_config["max_speed"])

        if self.global_path and len(self.global_path) > 1:
            first_waypoint = self.global_path[1]
            dx = first_waypoint[0] - start_port.pos[0]
            dy = first_waypoint[1] - start_port.pos[1]
            initial_theta = math.atan2(dy, dx)
        else:
            initial_theta = 0.0

        # state = (x, y, theta, v, w)
        self.state = (start_port.pos[0], start_port.pos[1], initial_theta, 0.0, 0.0)
        self.current_wp_idx = 0
        self.arrived = False  # mark when we reach destination

    # --- lightweight attributes for fast DataCollector ---
    @property
    def x(self):
        p = getattr(self, "pos", None)
        return float(p[0]) if p is not None else None

    @property
    def y(self):
        p = getattr(self, "pos", None)
        return float(p[1]) if p is not None else None

    @property
    def v(self):
        return float(self.state[3])

    def step(self):
        if self.arrived:
            return
        # early exit if position equals destination
        if getattr(self, "pos", None) is not None:
            if self.pos[0] == self.destination_port.pos[0] and self.pos[1] == self.destination_port.pos[1]:
                self._queue_removal()
                return

        if self.global_path and len(self.global_path) > 1:
            dt = self.dwa_config["dt"]

            # Early docking precheck: skip DWA when we're within one max step of destination
            max_spd_here = self.get_speed_limit()
            if self.model.speed_variation.get("enabled", False):
                var = self.model.max_speed_variation
                rng = _get_rng(self)
                max_spd_here = max(self.dwa_config["min_speed"],
                                   min(max_spd_here + rng.uniform(-max_spd_here*var, max_spd_here*var),
                                       self.dwa_config["max_speed"]))
            if self._close_enough_to_dock(max_spd_here, dt):
                self.move_to_destination()
                self._queue_removal()
                return

            local_goal = self.get_local_goal(self.state, self.global_path, lookahead=self.model.lookahead)

            # Speed limit via spatial index (built lazily & cached on model)
            self.dwa_config["max_speed"] = max_spd_here

            rng = _get_rng(self)
            # Directional (heading) variation
            if self.model.directional_variation["enabled"]:
                if self.heading_drift_duration > 0:
                    noisy_theta = self.state[2] + self.heading_deviation
                    self.heading_drift_duration -= 1
                else:
                    if rng.random() < self.model.deviation_chance:
                        self.heading_deviation = rng.uniform(-self.model.max_heading_deviation, self.model.max_heading_deviation)
                        self.heading_drift_duration = self.model.deviation_duration
                        noisy_theta = self.state[2] + self.heading_deviation
                    else:
                        noisy_theta = self.state[2]
                noisy_theta = (noisy_theta + math.pi) % (2 * math.pi) - math.pi
                noisy_state = (self.state[0], self.state[1], noisy_theta, self.state[3], self.state[4])
            else:
                noisy_state = self.state

            control, _, _ = dwa_control(
                noisy_state, self.dwa_config, self.model.obstacle_tree,
                self.model.buffered_obstacles, local_goal
            )

            if self.should_dock(control[0]):
                self.move_to_destination()
                self._queue_removal()
            else:
                self.state = motion(self.state, control[0], control[1], dt)
                self.move_position()

            # restore true max speed
            self.dwa_config["max_speed"] = self.original_max_speed

    def _close_enough_to_dock(self, speed, dt):
        dx = self.state[0] - self.destination_port.pos[0]
        dy = self.state[1] - self.destination_port.pos[1]
        return (dx*dx + dy*dy) <= (speed*dt)*(speed*dt)

    def _queue_removal(self):
        """Mark ship for removal by the model at end-of-step (safe w.r.t. scheduler iteration)."""
        self.arrived = True
        q = getattr(self.model, "_arrived_to_remove", None)
        if q is None:
            self.model._arrived_to_remove = [self]
        else:
            q.append(self)

    def get_noisy_speed(self):
        max_speed_variation = self.model.max_speed_variation
        variation_amount = self.dwa_config["max_speed"] * max_speed_variation
        rng = _get_rng(self)
        noisy_speed = self.dwa_config["max_speed"] + rng.uniform(-variation_amount, variation_amount)
        return max(self.dwa_config["min_speed"], min(noisy_speed, self.dwa_config["max_speed"]))

    def should_dock(self, current_speed):
        dx = self.state[0] - self.destination_port.pos[0]
        dy = self.state[1] - self.destination_port.pos[1]
        dist2 = dx * dx + dy * dy
        step_move = current_speed * self.dwa_config['dt']
        return dist2 <= step_move * step_move

    def move_position(self):
        self.pos = (float(self.state[0]), float(self.state[1]))
        self.model.space.move_agent(self, self.pos)

    def move_to_destination(self):
        self.pos = self.destination_port.pos
        self.model.space.move_agent(self, self.destination_port.pos)

    def assign_destination(self, all_ports, start_port):
        possible_destinations = [port for port in all_ports if port != start_port]
        rng = _get_rng(self)
        return rng.choice(possible_destinations) if possible_destinations else start_port

    def calculate_global_path(self, start, destination):
        grid_start = (int(start[0] / self.model.resolution), int(start[1] / self.model.resolution))
        grid_goal = (int(destination[0] / self.model.resolution), int(destination[1] / self.model.resolution))
        global_path_indices = astar(self.model.occupancy_grid, grid_start, grid_goal)
        if global_path_indices is None:
            return
        return [((i) * self.model.resolution, (j) * self.model.resolution) for (i, j) in global_path_indices]

    # --- Speed-limit zones: lazy-built STRtree on the model ---
    def _build_speed_zone_index_if_needed(self):
        if getattr(self.model, "_speed_zone_tree", None) is not None:
            return
        zones = getattr(self.model, "speed_limit_zones", None)
        if not zones:
            self.model._speed_zone_tree = None
            self.model._speed_zone_polys = []
            self.model._speed_zone_speed_by_id = {}
            self.model._speed_zone_speeds = []
            return

        polys = [Polygon(z["zone"]) for z in zones]
        self.model._speed_zone_polys = polys
        self.model._speed_zone_speeds = [z["max_speed"] for z in zones]

        if STRtree is not None and polys:
            try:
                self.model._speed_zone_tree = STRtree(polys)
            except Exception:
                self.model._speed_zone_tree = None
        else:
            self.model._speed_zone_tree = None
        self.model._speed_zone_speed_by_id = {id(p): z["max_speed"] for p, z in zip(polys, zones)}

    def get_speed_limit(self):
        zones = getattr(self.model, "speed_limit_zones", None)
        if not zones:
            return self.original_max_speed
        self._build_speed_zone_index_if_needed()

        pt = Point(self.state[0], self.state[1])
        tree = getattr(self.model, "_speed_zone_tree", None)
        if tree is not None:
            try:
                candidates = tree.query(pt, predicate="contains")
            except TypeError:
                candidates = tree.query(pt)
            for cand in candidates:
                if isinstance(cand, Integral):
                    idx = int(cand)
                    poly = self.model._speed_zone_polys[idx]
                    if poly.contains(pt):
                        return self.model._speed_zone_speeds[idx]
                else:
                    poly = cand
                    if poly.contains(pt):
                        return self.model._speed_zone_speed_by_id.get(id(poly), self.original_max_speed)
            return self.original_max_speed

        for idx, poly in enumerate(self.model._speed_zone_polys):
            if poly.contains(pt):
                return self.model._speed_zone_speeds[idx]
        return self.original_max_speed

    def get_local_goal(self, state, global_path, lookahead=3.0):
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
