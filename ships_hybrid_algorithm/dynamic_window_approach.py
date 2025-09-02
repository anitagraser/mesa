import math
import numpy as np
from shapely.geometry import Point, Polygon


def motion(state, v, w, dt):
    x, y, theta, _, _ = state
    # print(f'Current state x = {x} | y = {y} | theta = {theta} | v = {v} | w = {w} | dt = {dt}')

    x += v * math.cos(theta) * dt
    y += v * math.sin(theta) * dt
    theta += w * dt

    # Normalize theta to keep it between -π and π
    theta = (theta + math.pi) % (2 * math.pi) - math.pi

    # print(f'NEW state x = {x} | y = {y} | theta = {theta} | v = {v} | w = {w} | dt = {dt}')
    # print(f'-' * 50)

    return x, y, theta, v, w


def calc_trajectory(state, v, w, config):
    trajectory = []
    t = 0.0
    new_state = state
    while t <= config["predict_time"]:
        new_state = motion(new_state, v, w, config["dt"])
        trajectory.append(new_state)
        t += config["dt"]

    return trajectory


def calc_to_goal_cost(trajectory, goal, cost_gain):
    x, y, _, _, _ = trajectory[-1]
    distance = math.hypot(goal[0] - x, goal[1] - y)
    return cost_gain * distance

def calc_speed_cost(v, config):
    return config["speed_cost_gain"] * (config["max_speed"] - v)


def calc_obstacle_cost(trajectory, obstacle_tree, buffered_obstacles, config):
    min_distance = float("inf")
    for state in trajectory:
        x, y, _, _, _ = state
        point = Point(x, y)

        nearby_indices = obstacle_tree.query(point)
        for idx in nearby_indices:
            buffered_obs = buffered_obstacles[idx]
            if buffered_obs.contains(point):
                return float("inf")
            distance = buffered_obs.distance(point)
            if distance < min_distance:
                min_distance = distance
    return config["obstacle_cost_gain"] / (min_distance + 1e-6)


def calc_combined_turn_speed_cost(state, local_goal, v, cost_gain):
    x, y, theta, _, _ = state
    dx = local_goal[0] - x
    dy = local_goal[1] - y
    desired_theta = math.atan2(dy, dx)
    turn_angle = abs(desired_theta - theta)
    turn_angle = min(turn_angle, 2 * math.pi - turn_angle)
    # If the robot is moving fast and needs to turn a lot, the cost increases.
    return cost_gain * turn_angle * v # **0.5


def dwa_control(state, config, obstacle_tree, buffered_obstacles, local_goal):
    best_cost = float("inf")
    best_control = (0.0, 0.0)
    best_trajectory = []
    best_cost_info = {}

    # v_min = max(config["min_speed"], state[3] - config["max_acceleration"] * config["dt"])
    # v_max = min(config["max_speed"], state[3] + config["max_acceleration"] * config["dt"])

    v_lower = config["min_speed"]
    v_upper = min(config["max_speed"], state[3] + config["max_acceleration"] * config["dt"])
    v_samples = np.linspace(v_lower, v_upper, num=5)

    w_min = -np.deg2rad(config["max_yaw_rate"])
    w_max = np.deg2rad(config["max_yaw_rate"])

    # v_samples = np.linspace(v_min, v_max, num=5)
    w_samples = np.linspace(w_min, w_max, num=5)

    for v in v_samples:
        for w in w_samples:
            trajectory = calc_trajectory(state, v, w, config)
            to_goal_cost = calc_to_goal_cost(trajectory, local_goal, config["to_goal_cost_gain"])
            speed_cost = calc_speed_cost(v, config)
            obstacle_cost = calc_obstacle_cost(trajectory, obstacle_tree, buffered_obstacles, config)
            # turn_cost = calc_turn_cost(state, local_goal, config["turn_cost_gain"])
            turn_cost = calc_combined_turn_speed_cost(state, local_goal, v, config["turn_cost_gain"])
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
    return best_control, best_trajectory, best_cost_info