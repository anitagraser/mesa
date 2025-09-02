import math
from heapq import heappush, heappop
import numpy as np
from shapely import vectorized

def create_occupancy_grid(map_width, map_height, resolution, obstacles, threshold=0):
    """Create a grid representation of the environment for A* pathfinding."""
    grid_width = int(map_width / resolution)
    grid_height = int(map_height / resolution)
    grid = np.zeros((grid_width, grid_height), dtype=int)

    xs = (np.arange(grid_width) + 0.5) * resolution
    ys = (np.arange(grid_height) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys, indexing='ij')

    for obs in obstacles:
        poly = obs.shape.buffer(threshold)  # Expand obstacle slightly
        mask = vectorized.contains(poly, xx, yy)
        grid[mask] = 1  # Mark as occupied

    return grid

def heuristic(a, b):
    """Calculate Euclidean distance heuristic."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def reconstruct_path(came_from, current):
    """Reconstruct the shortest path from A* search."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def astar(grid, start, goal):
    width, height = grid.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # 4-way
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal

    while open_set:
        _, current = heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                if grid[neighbor] == 1:
                    continue  # Skip obstacles

                tentative_g = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found
