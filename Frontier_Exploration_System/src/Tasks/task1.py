#!/usr/bin/env python3
import math
from typing import List, Optional, Tuple, Dict
import heapq

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan

import tf2_ros
from rclpy.time import Time


class Task1(Node):
    

    def __init__(self):
        super().__init__('task1_node')

        
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)

        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)

        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        
        self.map_msg: Optional[OccupancyGrid] = None
        self.map_data: Optional[List[int]] = None
        self.map_width: int = 0
        self.map_height: int = 0
        self.map_resolution: float = 0.0
        self.map_origin_x: float = 0.0
        self.map_origin_y: float = 0.0

        
        self.map_updated_since_last_plan: bool = False

        
        self.latest_scan: Optional[LaserScan] = None

        
        self.state: str = 'IDLE'
        self.current_path: List[Tuple[float, float]] = []
        self.current_path_index: int = 0
        self.frontier_blacklist: List[Tuple[float, float]] = []

        
        self.inflation_cells = 6          
        self.a_star_wall_weight = 0.6     

        
        self.max_linear_speed = 0.85      
        self.k_v = 1.0
        self.waypoint_tolerance = 0.1

        
        
        self.k_w_align = 2.5
        self.max_angular_speed_align = 3.5
        
        self.k_w_travel = 2.5
        self.max_angular_speed_travel = 6.5

        
        self.min_frontier_dist = 0.5        

        
        self.front_obstacle_distance = 0.45  

        
        self.emergency_backing = False
        self.emergency_back_steps = 0
        self.max_emergency_back_steps = 7   

        
        self.aligning_to_path = False

        
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info('Task1 frontier + A* exploration node started.')

    
    
    

    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.map_data = list(msg.data)
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

        
        if self.state == 'PLAN_FRONTIER':
            self.map_updated_since_last_plan = True

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    
    
    

    def timer_cb(self):
        
        if self.map_msg is None or self.map_data is None:
            self.get_logger().info(
                'Waiting for map...', throttle_duration_sec=5.0)
            return

        pose = self.get_robot_pose()
        if pose is None:
            self.get_logger().info(
                'Waiting for TF (map -> base_footprint)...',
                throttle_duration_sec=5.0)
            return

        
        if self.latest_scan is None:
            self.get_logger().info(
                'Waiting for first LaserScan before moving...',
                throttle_duration_sec=5.0
            )
            self.stop_robot()
            return

        x, y, yaw = pose

        if self.state == 'IDLE':
            
            self.state = 'PLAN_FRONTIER'
            
            self.map_updated_since_last_plan = False
            self.get_logger().info('Switching to PLAN_FRONTIER state.')
            return

        if self.state == 'PLAN_FRONTIER':
            
            if not self.map_updated_since_last_plan:
                self.get_logger().info(
                    'Waiting for new map update before frontier planning...',
                    throttle_duration_sec=5.0
                )
                return

            
            
            self.map_updated_since_last_plan = False

            
            frontier_goals = self.extract_frontier_goals((x, y))
            if not frontier_goals:
                
                self.get_logger().info(
                    'No frontier goals left. Exploration finished.')
                self.stop_robot()
                self.state = 'DONE'
                return

            goal = self.select_frontier_goal((x, y), frontier_goals)
            if goal is None:
                self.get_logger().warn('No suitable frontier goal found.')
                self.stop_robot()
                self.state = 'DONE'
                return

            self.get_logger().info(
                f'Planning A* path to frontier goal at ({goal[0]:.2f}, {goal[1]:.2f})')

            path = self.plan_a_star((x, y), goal)

            if path is None or len(path) < 2:
                self.get_logger().warn(
                    'A* failed to find a path to this frontier. '
                    'Blacklisting this goal.')
                self.frontier_blacklist.append(goal)
                
                return

            self.current_path = path
            self.current_path_index = 0
            self.publish_current_path()   

            
            self.aligning_to_path = True

            self.state = 'FOLLOW_PATH'
            self.get_logger().info(
                f'A* path found with {len(path)} waypoints. '
                'Switching to FOLLOW_PATH state.')
            return

        if self.state == 'FOLLOW_PATH':
            if not self.current_path:
                self.get_logger().warn('Empty path in FOLLOW_PATH. Replanning.')
                self.state = 'PLAN_FRONTIER'
                
                self.map_updated_since_last_plan = False
                return

            reached = self.follow_path_step(x, y, yaw)

            if reached:
                self.get_logger().info('Reached current frontier goal.')

                
                if self.current_path:
                    last_goal_x, last_goal_y = self.current_path[-1]
                    self.frontier_blacklist.append((last_goal_x, last_goal_y))
                    self.get_logger().info(
                        f'Blacklisting reached goal at '
                        f'({last_goal_x:.2f}, {last_goal_y:.2f})'
                    )

                self.state = 'PLAN_FRONTIER'
                
                self.map_updated_since_last_plan = False
            return


        if self.state == 'DONE':
            
            self.stop_robot()
            return

    
    
    

    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        
        try:
            trans = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                Time())
        except Exception as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return None

        x = trans.transform.translation.x
        y = trans.transform.translation.y

        qx = trans.transform.rotation.x
        qy = trans.transform.rotation.y
        qz = trans.transform.rotation.z
        qw = trans.transform.rotation.w

        
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return (x, y, yaw)

    
    
    

    def world_to_map(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        mx = int((x - self.map_origin_x) / self.map_resolution)
        my = int((y - self.map_origin_y) / self.map_resolution)
        if mx < 0 or my < 0 or mx >= self.map_width or my >= self.map_height:
            return None
        return mx, my

    def map_to_world(self, mx: float, my: float) -> Tuple[float, float]:
        x = self.map_origin_x + (mx + 0.5) * self.map_resolution
        y = self.map_origin_y + (my + 0.5) * self.map_resolution
        return x, y

    def cell_is_occupied(self, mx: int, my: int) -> bool:
        idx = my * self.map_width + mx
        v = self.map_data[idx]
        return v > 50  

    def cell_is_unknown(self, mx: int, my: int) -> bool:
        idx = my * self.map_width + mx
        return self.map_data[idx] == -1

    def is_blacklisted(self, wx: float, wy: float, threshold: float = 0.6) -> bool:
        for bx, by in self.frontier_blacklist:
            if math.hypot(wx - bx, wy - by) < threshold:
                return True
        return False

    
    
    

    def extract_frontier_goals(self,
                               robot_xy: Tuple[float, float]
                               ) -> List[Tuple[float, float]]:
        
        rx, ry = robot_xy

        frontier_flags = [[False for _ in range(self.map_width)]
                          for _ in range(self.map_height)]

        
        for my in range(self.map_height):
            for mx in range(self.map_width):
                idx = my * self.map_width + mx
                if self.map_data[idx] != 0:
                    continue  

                has_unknown_neighbor = False
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx = mx + dx
                        ny = my + dy
                        if nx < 0 or ny < 0 or nx >= self.map_width or ny >= self.map_height:
                            continue
                        nidx = ny * self.map_width + nx
                        if self.map_data[nidx] == -1:
                            has_unknown_neighbor = True
                            break
                    if has_unknown_neighbor:
                        break

                if has_unknown_neighbor:
                    frontier_flags[my][mx] = True

        
        visited = [[False for _ in range(self.map_width)]
                   for _ in range(self.map_height)]
        clusters: List[List[Tuple[int, int]]] = []

        for my in range(self.map_height):
            for mx in range(self.map_width):
                if not frontier_flags[my][mx] or visited[my][mx]:
                    continue

                
                queue = [(mx, my)]
                visited[my][mx] = True
                cluster_cells: List[Tuple[int, int]] = []

                while queue:
                    cx, cy = queue.pop(0)
                    cluster_cells.append((cx, cy))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx = cx + dx
                        ny = cy + dy
                        if nx < 0 or ny < 0 or nx >= self.map_width or ny >= self.map_height:
                            continue
                        if not frontier_flags[ny][nx]:
                            continue
                        if visited[ny][nx]:
                            continue
                        visited[ny][nx] = True
                        queue.append((nx, ny))

                clusters.append(cluster_cells)

        goals: List[Tuple[float, float]] = []

        
        min_sizes = [25, 20, 15, 2]
        for min_size in min_sizes:
            goals.clear()

            for cluster_cells in clusters:
                if len(cluster_cells) < min_size:
                    continue

                
                best_cell = None
                best_dist = -1.0
                for cx, cy in cluster_cells:
                    wx, wy = self.map_to_world(cx, cy)
                    d = math.hypot(wx - rx, wy - ry)
                    if d > best_dist:
                        best_dist = d
                        best_cell = (cx, cy)

                if best_cell is None:
                    continue

                wx, wy = self.map_to_world(best_cell[0], best_cell[1])

                
                if not self.is_blacklisted(wx, wy):
                    goals.append((wx, wy))

            if goals:
                self.get_logger().info(
                    f'[frontier] Using frontier clusters with size >= {min_size} '
                    f'(clusters={len(clusters)} , goals={len(goals)})'
                )
                break

        return goals

    
    
    

    def compute_unknown_ahead_score(
        self,
        robot_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        max_ahead: float = 2.0,
        step: Optional[float] = None
    ) -> float:
        
        rx, ry = robot_xy
        gx, gy = goal_xy

        dx = gx - rx
        dy = gy - ry
        dist_rg = math.hypot(dx, dy)
        if dist_rg < 1e-6:
            return 0.0

        
        ux = dx / dist_rg
        uy = dy / dist_rg

        if step is None or step <= 0.0:
            step = max(self.map_resolution, 0.05)

        total_samples = 0
        unknown_count = 0

        s = 0.0
        while s <= max_ahead:
            px = gx + ux * s
            py = gy + uy * s

            cell = self.world_to_map(px, py)
            if cell is None:
                
                break

            mx, my = cell
            idx = my * self.map_width + mx
            v = self.map_data[idx]

            total_samples += 1
            if v == -1:
                unknown_count += 1

            s += step

        if total_samples == 0:
            return 0.0

        return unknown_count / total_samples

    
    
    

    def compute_distance_map_from_robot(
        self,
        robot_xy: Tuple[float, float]
    ) -> Optional[List[List[float]]]:
        
        if self.map_data is None:
            return None

        start_cell = self.world_to_map(robot_xy[0], robot_xy[1])
        if start_cell is None:
            return None

        sx, sy = start_cell

        
        idx0 = sy * self.map_width + sx
        v0 = self.map_data[idx0]

        if v0 > 50 or v0 == -1:
            
            
            best = None
            best_dist = float('inf')

            
            search_r = max(self.inflation_cells * 2, 8)

            for dy in range(-search_r, search_r + 1):
                ny = sy + dy
                if ny < 0 or ny >= self.map_height:
                    continue
                for dx in range(-search_r, search_r + 1):
                    nx = sx + dx
                    if nx < 0 or nx >= self.map_width:
                        continue

                    idx = ny * self.map_width + nx
                    v = self.map_data[idx]
                    
                    if v > 50 or v == -1:
                        continue

                    d = math.hypot(dx, dy)
                    if d < best_dist:
                        best_dist = d
                        best = (nx, ny)

            if best is None:
                
                return None

            sx, sy = best

        INF = float('inf')
        dist_map: List[List[float]] = [
            [INF for _ in range(self.map_width)] for _ in range(self.map_height)
        ]

        
        moves = [
            (1, 0, 1.0), (-1, 0, 1.0),
            (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
        ]

        pq: List[Tuple[float, int, int]] = []
        dist_map[sy][sx] = 0.0
        heapq.heappush(pq, (0.0, sx, sy))
        
        while pq:
            cur_d, mx, my = heapq.heappop(pq)
            if cur_d > dist_map[my][mx]:
                continue

            for dx, dy, step_cells in moves:
                nx = mx + dx
                ny = my + dy
                if nx < 0 or ny < 0 or nx >= self.map_width or ny >= self.map_height:
                    continue

                idx = ny * self.map_width + nx
                v = self.map_data[idx]
                
                if v > 50 or v == -1:
                    continue

                step_dist = step_cells * self.map_resolution
                nd = cur_d + step_dist

                if nd < dist_map[ny][nx]:
                    dist_map[ny][nx] = nd
                    heapq.heappush(pq, (nd, nx, ny))

        return dist_map

    
    
    

    def select_frontier_goal(self,
                             robot_xy: Tuple[float, float],
                             goals: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        
        if not goals:
            return None

        if self.map_data is None:
            return None

        rx, ry = robot_xy
        min_d = self.min_frontier_dist

        
        dist_map = self.compute_distance_map_from_robot(robot_xy)
        if dist_map is None:
            
            
            self.get_logger().warn('[frontier] distance map computation failed.')
            return None

        best_goal = None
        best_score = float('inf')  

        
        alpha_unknown = 1.0  

        
        for gx, gy in goals:
            cell = self.world_to_map(gx, gy)
            if cell is None:
                continue
            mx, my = cell

            path_d = dist_map[my][mx]
            if not math.isfinite(path_d):
                
                continue

            
            if path_d < min_d:
                continue

            unknown_score = self.compute_unknown_ahead_score(robot_xy, (gx, gy))

            
            score = path_d - alpha_unknown * unknown_score

            if score < best_score:
                best_score = score
                best_goal = (gx, gy)

        
        if best_goal is None:
            nearest_goal = None
            nearest_dist = float('inf')

            for gx, gy in goals:
                cell = self.world_to_map(gx, gy)
                if cell is None:
                    continue
                mx, my = cell

                path_d = dist_map[my][mx]
                if not math.isfinite(path_d):
                    continue

                if path_d < nearest_dist:
                    nearest_dist = path_d
                    nearest_goal = (gx, gy)

            best_goal = nearest_goal

            if best_goal is not None:
                self.get_logger().info(
                    f'[frontier] No frontier beyond {min_d:.2f} m path-dist; '
                    f'using nearest goal at ({best_goal[0]:.2f}, {best_goal[1]:.2f}), '
                    f'path_dist={nearest_dist:.2f}'
                )
        else:
            
            cell_best = self.world_to_map(best_goal[0], best_goal[1])
            if cell_best is not None:
                mxb, myb = cell_best
                dist_best = dist_map[myb][mxb]
            else:
                dist_best = float('nan')

            self.get_logger().info(
                f'[frontier] Selected goal at '
                f'({best_goal[0]:.2f}, {best_goal[1]:.2f}), path_dist={dist_best:.2f}, '
                f'score={best_score:.2f} (min_dist={min_d:.2f})'
            )

        return best_goal

    
    
    

    def compute_wall_penalty(self,
                             x: float,
                             y: float,
                             max_search_cells: int = 8,
                             safe_dist: float = 0.35) -> float:
        
        if self.map_data is None:
            return 0.0

        cell = self.world_to_map(x, y)
        if cell is None:
            
            return 0.0

        mx, my = cell
        best_cell_dist = None

        for dy in range(-max_search_cells, max_search_cells + 1):
            ny = my + dy
            if ny < 0 or ny >= self.map_height:
                continue
            for dx in range(-max_search_cells, max_search_cells + 1):
                nx = mx + dx
                if nx < 0 or nx >= self.map_width:
                    continue
                idx = ny * self.map_width + nx
                v = self.map_data[idx]
                if v > 50:  
                    d_cells = math.hypot(dx, dy)
                    if best_cell_dist is None or d_cells < best_cell_dist:
                        best_cell_dist = d_cells

        
        if best_cell_dist is None:
            return 0.0

        d_m = best_cell_dist * self.map_resolution

        if d_m >= safe_dist:
            return 0.0

        
        return (safe_dist - d_m) / safe_dist

    
    
    

    def plan_a_star(self,
                    start: Tuple[float, float],
                    goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        
        if self.map_data is None:
            return None

        sx, sy = start
        gx, gy = goal

        coll_start = self.is_in_collision(sx, sy, allow_unknown=True)
        self.get_logger().info(f"A*: start collision? {coll_start}")

        start_cell = self.world_to_map(sx, sy)
        goal_cell = self.world_to_map(gx, gy)

        if start_cell is None or goal_cell is None:
            self.get_logger().warn('A*: start or goal is outside map.')
            return None

        sx_idx, sy_idx = start_cell
        gx_idx, gy_idx = goal_cell

        def cell_traversable(mx: int, my: int) -> bool:
            wx, wy = self.map_to_world(mx, my)
            
            return not self.is_in_collision(wx, wy, allow_unknown=True)

        
        if coll_start:
            self.get_logger().info('A*: start in collision, searching nearby free start cell...')
            best_start = None
            best_dist_start = float('inf')
            search_r_start = max(self.inflation_cells * 2, 8)

            for dy in range(-search_r_start, search_r_start + 1):
                for dx in range(-search_r_start, search_r_start + 1):
                    nx = sx_idx + dx
                    ny = sy_idx + dy
                    if nx < 0 or ny < 0 or nx >= self.map_width or ny >= self.map_height:
                        continue
                    if not cell_traversable(nx, ny):
                        continue
                    d = math.hypot(nx - sx_idx, ny - sy_idx)
                    if d < best_dist_start:
                        best_dist_start = d
                        best_start = (nx, ny)

            if best_start is None:
                self.get_logger().warn('A*: no traversable start cell found near robot.')
                return None

            sx_idx, sy_idx = best_start
            self.get_logger().info(
                f'A*: moved start to nearest free cell at ({sx_idx}, {sy_idx}), '
                f'grid_dist={best_dist_start:.2f}'
            )
        

        
        if not cell_traversable(gx_idx, gy_idx):
            best = None
            best_dist = float('inf')

            
            search_r = max(self.inflation_cells * 2, 8)  

            for dy in range(-search_r, search_r + 1):
                for dx in range(-search_r, search_r + 1):
                    nx = gx_idx + dx
                    ny = gy_idx + dy
                    if nx < 0 or ny < 0 or nx >= self.map_width or ny >= self.map_height:
                        continue
                    if not cell_traversable(nx, ny):
                        continue
                    d = math.hypot(nx - gx_idx, ny - gy_idx)
                    if d < best_dist:
                        best_dist = d
                        best = (nx, ny)

            if best is None:
                self.get_logger().warn('A*: no traversable goal cell found near frontier.')
                return None
            gx_idx, gy_idx = best

        start_key = (sx_idx, sy_idx)
        goal_key = (gx_idx, gy_idx)

        
        moves = [
            (1, 0, 1.0), (-1, 0, 1.0),
            (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
        ]

        open_heap: List[Tuple[float, float, int, int]] = []
        heapq.heappush(open_heap, (0.0, 0.0, sx_idx, sy_idx))

        g_cost: Dict[Tuple[int, int], float] = {start_key: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        closed = set()
        found = False

        while open_heap:
            f, cur_g, mx, my = heapq.heappop(open_heap)
            cur_key = (mx, my)

            if cur_key in closed:
                continue
            closed.add(cur_key)

            if cur_key == goal_key:
                found = True
                break

            for dx, dy, dist_cells in moves:
                nx = mx + dx
                ny = my + dy

                if nx < 0 or ny < 0 or nx >= self.map_width or ny >= self.map_height:
                    continue

                n_key = (nx, ny)
                if n_key in closed:
                    continue

                if not cell_traversable(nx, ny):
                    continue

                step_dist = dist_cells * self.map_resolution
                wx, wy = self.map_to_world(nx, ny)
                wall_pen = self.compute_wall_penalty(wx, wy)
                step_cost = step_dist + self.a_star_wall_weight * wall_pen

                new_g = cur_g + step_cost

                if new_g >= g_cost.get(n_key, float('inf')):
                    continue

                g_cost[n_key] = new_g

                h = self.map_resolution * math.hypot(nx - gx_idx, ny - gy_idx)
                new_f = new_g + h

                came_from[n_key] = cur_key
                heapq.heappush(open_heap, (new_f, new_g, nx, ny))

        if not found:
            self.get_logger().warn('A* planning failed (goal not reachable).')
            return None

        
        path_cells: List[Tuple[int, int]] = []
        cur = goal_key
        while cur != start_key:
            path_cells.append(cur)
            if cur not in came_from:
                self.get_logger().warn('A* path reconstruction failed.')
                return None
            cur = came_from[cur]
        path_cells.append(start_key)
        path_cells.reverse()

        
        path: List[Tuple[float, float]] = [
            self.map_to_world(mx, my) for (mx, my) in path_cells
        ]

        
        if len(path) > 2:
            thinned: List[Tuple[float, float]] = [path[0]]
            step_keep = 3  
            for i in range(step_keep, len(path) - 1, step_keep):
                thinned.append(path[i])
            thinned.append(path[-1])
            path = thinned

        return path


    def is_in_collision(self,
                        x: float,
                        y: float,
                        allow_unknown: bool = False) -> bool:
        
        cell = self.world_to_map(x, y)
        if cell is None:
            return True
        mx, my = cell

        inflation_cells = self.inflation_cells

        for dy in range(-inflation_cells, inflation_cells + 1):
            for dx in range(-inflation_cells, inflation_cells + 1):
                nx = mx + dx
                ny = my + dy
                if nx < 0 or ny < 0 or nx >= self.map_width or ny >= self.map_height:
                    return True
                idx = ny * self.map_width + nx
                v = self.map_data[idx]
                if v > 50:
                    return True
                if v == -1 and not allow_unknown:
                    return True
        return False

    def collision_free(self,
                       x1: float,
                       y1: float,
                       x2: float,
                       y2: float,
                       step: float = 0.05,
                       allow_unknown: bool = False) -> bool:
        
        dist = self.dist(x1, y1, x2, y2)
        steps = max(int(dist / step), 1)
        for i in range(steps + 1):
            t = i / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if self.is_in_collision(x, y, allow_unknown=allow_unknown):
                return False
        return True

    @staticmethod
    def dist(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    
    
    

    def publish_current_path(self):
        
        if not self.current_path:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'  

        for x, y in self.current_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  

            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    
    
    

    def follow_path_step(self,
                         x: float,
                         y: float,
                         yaw: float) -> bool:
        

        
        if self.emergency_backing:
            twist = Twist()
            twist.linear.x = -0.1
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)

            self.emergency_back_steps += 1
            if self.emergency_back_steps >= self.max_emergency_back_steps:
                
                self.emergency_backing = False
                self.stop_robot()
                self.state = 'PLAN_FRONTIER'
                
                self.map_updated_since_last_plan = False
            return False

        
        if self.latest_scan is not None:
            scan = self.latest_scan
            angle_min = scan.angle_min
            angle_inc = scan.angle_increment
            ranges = scan.ranges

            front_min = float('inf')
            for i, r in enumerate(ranges):
                if not math.isfinite(r):
                    continue
                ang = angle_min + i * angle_inc
                a = math.atan2(math.sin(ang), math.cos(ang))
                
                if -math.radians(25.0) <= a <= math.radians(25.0):
                    front_min = min(front_min, r)

            if front_min < self.front_obstacle_distance:
                self.get_logger().warn(
                    f'Obstacle too close at step start (front_min={front_min:.2f} m). '
                    'Emergency backing and replanning.'
                )
                self.emergency_backing = True
                self.emergency_back_steps = 0

                twist = Twist()
                twist.linear.x = -0.1
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                return False

        
        if not self.current_path:
            self.stop_robot()
            return True

        
        goal_x, goal_y = self.current_path[-1]
        dist_to_goal = math.hypot(goal_x - x, goal_y - y)
        if dist_to_goal < self.waypoint_tolerance:
            self.stop_robot()
            return True

        
        closest_idx = 0
        closest_dist = float('inf')
        for i, (px, py) in enumerate(self.current_path):
            d = math.hypot(px - x, py - y)
            if d < closest_dist:
                closest_dist = d
                closest_idx = i

        
        lookahead_dist = 0.5  
        target_idx = len(self.current_path) - 1  

        for i in range(closest_idx, len(self.current_path)):
            px, py = self.current_path[i]
            d = math.hypot(px - x, py - y)
            if d >= lookahead_dist:
                target_idx = i
                break

        target_x, target_y = self.current_path[target_idx]

        dx = target_x - x
        dy = target_y - y
        dist_to_target = math.hypot(dx, dy)

        
        target_yaw = math.atan2(dy, dx)
        yaw_error = self.normalize_angle(target_yaw - yaw)

        twist = Twist()

        
        if self.aligning_to_path:
            angle_thresh_start = math.radians(5.0)  

            if abs(yaw_error) > angle_thresh_start:
                twist.linear.x = 0.0
                twist.angular.z = max(
                    -self.max_angular_speed_align,
                    min(self.max_angular_speed_align, self.k_w_align * yaw_error)
                )
            else:
                
                self.aligning_to_path = False
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            self.cmd_pub.publish(twist)
            return False

        
        
        
        if dist_to_goal < 0.7:
            v = min(self.max_linear_speed, self.k_v * dist_to_goal)
        else:
            v = self.max_linear_speed

        
        angle_mid = math.radians(20.0)
        angle_slow = math.radians(45.0)

        if abs(yaw_error) > angle_slow:
            v *= 0.45   
        elif abs(yaw_error) > angle_mid:
            v *= 0.7

        
        w = max(
            -self.max_angular_speed_travel,
            min(self.max_angular_speed_travel, self.k_w_travel * yaw_error)
        )

        
        v = max(0.0, min(self.max_linear_speed, v))
        w = max(-self.max_angular_speed_travel, min(self.max_angular_speed_travel, w))

        twist.linear.x = v
        twist.angular.z = w

        self.cmd_pub.publish(twist)
        return False

    
    
    

    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = Task1()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

