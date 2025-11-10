import math
import numpy as np
from shapely.geometry import Point, box

def motion(state, v, w, dt):
    x, y, theta, _, _ = state
    x += v * math.cos(theta) * dt
    y += v * math.sin(theta) * dt
    theta += w * dt
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    return x, y, theta, v, w

def calc_trajectory(state, v, w, config):
    """Integrate forward with constant (v,w)."""
    trajectory = []
    t = 0.0
    new_state = state
    dt = config["dt"]
    T = config.get("predict_time", 1.0)  # shorter default for speed
    while t <= T:
        new_state = motion(new_state, v, w, dt)
        trajectory.append(new_state)
        t += dt
    return trajectory

def calc_to_goal_cost(trajectory, goal, cost_gain):
    x, y, _, _, _ = trajectory[-1]
    distance = math.hypot(goal[0] - x, goal[1] - y)
    return cost_gain * distance

def calc_speed_cost(v, config):
    return config.get("speed_cost_gain", 1.0) * (config["max_speed"] - v)

def _candidate_obstacles_for_trajectory(trajectory, obstacle_tree):
    xs = [s[0] for s in trajectory]
    ys = [s[1] for s in trajectory]
    bb = box(min(xs), min(ys), max(xs), max(ys))
    try:
        cand = obstacle_tree.query(bb)
    except TypeError:
        cand = obstacle_tree.query(bb)
    return cand

def calc_obstacle_cost(trajectory, obstacle_tree, buffered_obstacles, config):
    early_hit = config.get("collision_early_exit", config.get("robot_radius", 1.0))
    min_distance = float("inf")
    nearby_indices = _candidate_obstacles_for_trajectory(trajectory, obstacle_tree)
    if len(nearby_indices) == 0:
        return 0.0
    for state in trajectory:
        x, y, _, _, _ = state
        point = Point(x, y)
        for idx in nearby_indices:
            buffered_obs = buffered_obstacles[idx]
            if buffered_obs.contains(point):
                return float("inf")
            d = buffered_obs.distance(point)
            if d < early_hit:
                return float("inf")
            if d < min_distance:
                min_distance = d
    gain = config.get("obstacle_cost_gain", 1.0)
    return gain / (min_distance + 1e-6)

def calc_combined_turn_speed_cost(state, local_goal, v, cost_gain):
    x, y, theta, _, _ = state
    dx = local_goal[0] - x
    dy = local_goal[1] - y
    desired_theta = math.atan2(dy, dx)
    turn_angle = abs(desired_theta - theta)
    turn_angle = min(turn_angle, 2 * math.pi - turn_angle)
    return cost_gain * turn_angle * v

def dwa_control(state, config, obstacle_tree, buffered_obstacles, local_goal):
    best_cost = float("inf")
    best_control = (0.0, 0.0)
    best_trajectory = []
    best_cost_info = {}

    num_v = int(config.get("num_v_samples", 3))
    num_w = int(config.get("num_w_samples", 3))

    v_lower = config["min_speed"]
    v_upper = min(config["max_speed"], state[3] + config["max_acceleration"] * config["dt"])
    v_samples = np.linspace(v_lower, v_upper, num=max(2, num_v))

    w_min = -np.deg2rad(config["max_yaw_rate"])
    w_max =  np.deg2rad(config["max_yaw_rate"])
    w_samples = np.linspace(w_min, w_max, num=max(2, num_w))

    max_evals = int(config.get("max_dwa_evals", 64))
    eval_count = 0

    for v in v_samples:
        for w in w_samples:
            trajectory = calc_trajectory(state, v, w, config)
            to_goal_cost = calc_to_goal_cost(trajectory, local_goal, config.get("to_goal_cost_gain", 1.0))
            speed_cost   = calc_speed_cost(v, config)
            obstacle_cost = calc_obstacle_cost(trajectory, obstacle_tree, buffered_obstacles, config)
            turn_cost    = calc_combined_turn_speed_cost(state, local_goal, v, config.get("turn_cost_gain", 1.0))

            total_cost = to_goal_cost + speed_cost + obstacle_cost + turn_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_control = (v, w)
                best_trajectory = trajectory
                best_cost_info = {
                    "total_cost": total_cost,
                    "to_goal_cost": to_goal_cost,
                    "speed_cost": speed_cost,
                    "obstacle_cost": obstacle_cost,
                    "turn_cost": turn_cost
                }

            eval_count += 1
            if eval_count >= max_evals:
                break
        if eval_count >= max_evals:
            break

    return best_control, best_trajectory, best_cost_info
