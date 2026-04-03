#!/usr/bin/env python3

import math
from typing import List, Optional, Tuple, Dict, Set
import heapq
import random

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
from rclpy.time import Time
from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)


class Task2(Node):
    

    def __init__(self):
        super().__init__('task2_node')

        
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, 'global_plan', 10)

        
        self.rrt_path_marker_pub = self.create_publisher(Marker, 'rrt_path_marker', 10)
        self.rrt_tree_pub = self.create_publisher(MarkerArray, 'rrt_tree', 10)

        
        map_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, map_qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10,
        )

        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        
        self.map_msg: Optional[OccupancyGrid] = None
        self.map_data: Optional[List[int]] = None
        self.map_width: int = 0
        self.map_height: int = 0
        self.map_resolution: float = 0.0
        self.map_origin_x: float = 0.0
        self.map_origin_y: float = 0.0

        
        self.latest_scan: Optional[LaserScan] = None

        
        self.state: str = 'WAIT_GOAL'   
        self.current_goal: Optional[Tuple[float, float]] = None
        self.current_path: List[Tuple[float, float]] = []
        self.current_path_index: int = 0

        
        self.inflation_cells = 5
        self.a_star_wall_weight = 0.6

        
        self.max_linear_speed = 0.65
        self.k_v = 1.0
        self.waypoint_tolerance = 0.1

        self.k_w_align = 2.5
        self.max_angular_speed_align = 3.3
        self.k_w_travel = 2.5
        self.max_angular_speed_travel = 6.5  

        
        self.min_front_obstacle_distance = 0.3
        self.max_front_obstacle_distance = 0.60
        self.front_distance_gain = 0.5

        self.emergency_backing = False
        self.emergency_back_steps = 0
        self.max_emergency_back_steps = 7

        
        self.current_speed = 0.0

        
        self.aligning_to_path = False

        
        self.dynamic_obstacles: Set[Tuple[int, int]] = set()

        
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info(
            'Task2 BONUS node (A* global + RRT* local replanning) started.'
        )

    

    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.map_data = list(msg.data)
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

        
        self.remove_single_pixel_obstacle_noise(min_cluster_size=2)

        self.get_logger().info(
            f'Received map: {self.map_width}x{self.map_height}, '
            f'res={self.map_resolution:.3f}'
        )

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def goal_callback(self, msg: PoseStamped):
        
        self.stop_robot()

        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self.current_goal = (gx, gy)
        self.current_path = []
        self.aligning_to_path = False
        self.emergency_backing = False
        self.emergency_back_steps = 0
        self.state = 'PLAN_PATH'
        self.get_logger().info(
            f'New goal received: ({gx:.2f}, {gy:.2f}). Switching to PLAN_PATH.'
        )

    

    def timer_cb(self):
        
        if self.map_msg is None or self.map_data is None:
            self.get_logger().info('Waiting for map...', throttle_duration_sec=5.0)
            return

        
        pose = self.get_robot_pose()
        if pose is None:
            self.get_logger().info(
                'Waiting for TF (map -> base_footprint)...',
                throttle_duration_sec=5.0
            )
            return

        
        if self.latest_scan is None:
            self.get_logger().info(
                'Waiting for LaserScan...', throttle_duration_sec=5.0
            )
            self.stop_robot()
            return

        x, y, yaw = pose

        if self.state == 'WAIT_GOAL':
            self.stop_robot()
            return

        if self.state == 'PLAN_PATH':
            if self.current_goal is None:
                self.get_logger().warn(
                    'PLAN_PATH but no current_goal. Going to WAIT_GOAL.'
                )
                self.state = 'WAIT_GOAL'
                return

            gx, gy = self.current_goal
            self.get_logger().info(
                f'Planning A* path from ({x:.2f}, {y:.2f}) to ({gx:.2f}, {gy:.2f})'
            )
            path = self.plan_a_star((x, y), (gx, gy))

            if path is None or len(path) < 2:
                self.get_logger().warn('A* failed to find a path to goal.')
                self.stop_robot()
                self.state = 'WAIT_GOAL'
                return

            self.current_path = path
            self.current_path_index = 0
            self.publish_current_path()

            self.aligning_to_path = True
            self.state = 'FOLLOW_PATH'
            self.get_logger().info(
                f'A* path found with {len(path)} waypoints. '
                'Switching to FOLLOW_PATH.'
            )
            return

        if self.state == 'FOLLOW_PATH':
            if not self.current_path:
                self.get_logger().warn(
                    'FOLLOW_PATH but current_path is empty, replanning.'
                )
                self.state = 'PLAN_PATH'
                return

            reached = self.follow_path_step(x, y, yaw)

            if reached:
                self.get_logger().info(
                    'Reached current goal. Switching to WAIT_GOAL.'
                )
                self.state = 'WAIT_GOAL'
                self.current_goal = None
            return

    

    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            trans = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                Time()
            )
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

    def cell_is_occupied_static(self, mx: int, my: int) -> bool:
        idx = my * self.map_width + mx
        v = self.map_data[idx]
        return v > 50

    

    def remove_single_pixel_obstacle_noise(self, min_cluster_size: int = 2):
        if self.map_data is None:
            return

        w = self.map_width
        h = self.map_height
        visited = [[False for _ in range(w)] for _ in range(h)]
        removed = 0

        for my in range(h):
            for mx in range(w):
                if visited[my][mx]:
                    continue

                idx = my * w + mx
                if self.map_data[idx] <= 50:
                    continue

                queue = [(mx, my)]
                visited[my][mx] = True
                cluster: List[Tuple[int, int]] = []

                while queue:
                    cx, cy = queue.pop(0)
                    cluster.append((cx, cy))

                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx = cx + dx
                        ny = cy + dy
                        if nx < 0 or ny < 0 or nx >= w or ny >= h:
                            continue
                        if visited[ny][nx]:
                            continue
                        nidx = ny * w + nx
                        if self.map_data[nidx] <= 50:
                            continue

                        visited[ny][nx] = True
                        queue.append((nx, ny))

                if len(cluster) < min_cluster_size:
                    for cx, cy in cluster:
                        cidx = cy * w + cx
                        if self.map_data[cidx] > 50:
                            self.map_data[cidx] = 0
                            removed += 1

        if removed > 0:
            self.get_logger().info(
                f'Removed {removed} tiny obstacle cells (cluster size < {min_cluster_size}).'
            )

    

    def mark_dynamic_obstacles_from_scan(self,
                                         robot_x: float,
                                         robot_y: float,
                                         robot_yaw: float):
        if self.latest_scan is None or self.map_data is None:
            return

        scan = self.latest_scan
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment
        ranges = scan.ranges

        fov = math.radians(70.0)
        added_cells = 0

        for i, r in enumerate(ranges):
            if not math.isfinite(r):
                continue
            if r <= 0.05:
                continue

            ang = angle_min + i * angle_inc
            if abs(ang) > fov:
                continue

            beam_yaw = robot_yaw + ang
            ex = robot_x + r * math.cos(beam_yaw)
            ey = robot_y + r * math.sin(beam_yaw)

            cell = self.world_to_map(ex, ey)
            if cell is None:
                continue
            mx, my = cell
            idx = my * self.map_width + mx
            v = self.map_data[idx]

            if v != 0:
                continue

            if (mx, my) not in self.dynamic_obstacles:
                self.dynamic_obstacles.add((mx, my))
                added_cells += 1

        if added_cells > 0:
            self.get_logger().info(
                f'Marked {added_cells} dynamic obstacle cells from LaserScan.'
            )

    

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

                if (nx, ny) in self.dynamic_obstacles:
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
            self.get_logger().info(
                'A*: start in collision, searching nearby free start cell...'
            )
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
                self.get_logger().warn(
                    'A*: no traversable start cell found near robot.'
                )
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
                self.get_logger().warn('A*: no traversable goal cell found near goal.')
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

    def publish_rrt_path_marker(self, path: List[Tuple[float, float]]):
        if not path:
            return

        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'rrt_path'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD

        m.scale.x = 0.03

        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0

        for x, y in path:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            m.points.append(p)

        self.rrt_path_marker_pub.publish(m)

    def publish_rrt_tree(self, nodes: List["Task2._RRTStarNode"]):
        if not nodes or len(nodes) < 2:
            return

        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'rrt_tree'
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD

        m.scale.x = 0.01

        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.5

        for node in nodes:
            if node.parent < 0:
                continue
            parent = nodes[node.parent]

            p1 = Point()
            p1.x = parent.x
            p1.y = parent.y
            p1.z = 0.0

            p2 = Point()
            p2.x = node.x
            p2.y = node.y
            p2.z = 0.0

            m.points.append(p1)
            m.points.append(p2)

        ma.markers.append(m)
        self.rrt_tree_pub.publish(ma)

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

                
                self.get_logger().info('Emergency backing finished. Trying local RRT* replanning...')
                success = self.local_rrt_star_replan(x, y, yaw)

                if success:
                    self.get_logger().info('Local RRT* replanning succeeded.')
                    self.aligning_to_path = True
                    self.current_speed = 0.0
                else:
                    self.get_logger().warn(
                        'Local RRT* replanning failed. Falling back to global A* replanning.'
                    )
                    self.state = 'PLAN_PATH'
            return False

        
        v_safe = max(0.0, min(self.max_linear_speed, self.current_speed))
        dyn_dist = (
            self.min_front_obstacle_distance
            + self.front_distance_gain * v_safe
        )
        dyn_dist = max(
            self.min_front_obstacle_distance,
            min(self.max_front_obstacle_distance, dyn_dist)
        )

        
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

            if front_min < dyn_dist:
                self.get_logger().warn(
                    f'Obstacle too close in front (front_min={front_min:.2f} m, '
                    f'threshold={dyn_dist:.2f} m). '
                    'Marking dynamic obstacles and starting emergency backing.'
                )

                self.mark_dynamic_obstacles_from_scan(x, y, yaw)

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

        lookahead_dist = 0.4
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

        angle_slow = math.radians(45.0)
        angle_mid = math.radians(20.0)
        angle_fast = math.radians(10.0)

        if abs(yaw_error) > angle_slow:
            v *= 0.45
        elif abs(yaw_error) > angle_mid:
            v *= 0.7
        elif abs(yaw_error) > angle_fast:
            v *= 0.8

        
        w = max(
            -self.max_angular_speed_travel,
            min(self.max_angular_speed_travel, self.k_w_travel * yaw_error)
        )

        v = max(0.0, min(self.max_linear_speed, v))
        w = max(-self.max_angular_speed_travel, min(self.max_angular_speed_travel, w))

        twist.linear.x = v
        twist.angular.z = w

        self.current_speed = v
        self.cmd_pub.publish(twist)
        return False

    

    def local_rrt_star_replan(self, x: float, y: float, yaw: float) -> bool:
        if not self.current_path:
            self.get_logger().warn('local_rrt_star_replan: no current_path.')
            return False

        
        closest_idx = 0
        closest_dist = float('inf')
        for i, (px, py) in enumerate(self.current_path):
            d = math.hypot(px - x, py - y)
            if d < closest_dist:
                closest_dist = d
                closest_idx = i

        
        lookahead_nodes = 10
        local_goal_idx = min(closest_idx + lookahead_nodes,
                             len(self.current_path) - 1)
        local_goal = self.current_path[local_goal_idx]
        sx, sy = x, y
        gx, gy = local_goal

        
        pad = 1.0
        xmin = min(sx, gx) - pad
        xmax = max(sx, gx) + pad
        ymin = min(sy, gy) - pad
        ymax = max(sy, gy) + pad
        bounds = (xmin, xmax, ymin, ymax)

        
        local_path = self.rrt_star(
            start=(sx, sy),
            goal=(gx, gy),
            max_iter=500,
            step_size=0.25, 
            goal_radius=0.3,
            search_radius=0.7,
            bounds=bounds
        )

        if local_path is None or len(local_path) < 2:
            return False

        
        self.publish_rrt_path_marker(local_path)

        
        tail = self.current_path[local_goal_idx + 1:]
        new_path = local_path + tail

        self.current_path = new_path
        self.current_path_index = 0
        self.publish_current_path()
        return True

    

    class _RRTStarNode:
        def __init__(self, x: float, y: float, parent: int = -1, cost: float = 0.0):
            self.x = x
            self.y = y
            self.parent = parent
            self.cost = cost

    def rrt_star(self,
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 max_iter: int = 1500,
                 step_size: float = 0.25,
                 goal_radius: float = 0.3,
                 search_radius: float = 0.7,
                 bounds: Optional[Tuple[float, float, float, float]] = None
                 ) -> Optional[List[Tuple[float, float]]]:

        sx, sy = start
        gx, gy = goal

        
        if self.is_in_collision(sx, sy, allow_unknown=True):
            start_cell = self.world_to_map(sx, sy)
            if start_cell is None:
                self.get_logger().warn('RRT*: start in collision and outside map.')
                return None
            sx_idx, sy_idx = start_cell

            best_start = None
            best_dist_start = float('inf')
            search_r_start = max(self.inflation_cells * 2, 8)

            for dy in range(-search_r_start, search_r_start + 1):
                for dx in range(-search_r_start, search_r_start + 1):
                    nx = sx_idx + dx
                    ny = sy_idx + dy
                    if nx < 0 or ny < 0 or nx >= self.map_width or ny >= self.map_height:
                        continue
                    wx, wy = self.map_to_world(nx, ny)
                    if self.is_in_collision(wx, wy, allow_unknown=True):
                        continue
                    d = math.hypot(dx, dy)
                    if d < best_dist_start:
                        best_dist_start = d
                        best_start = (wx, wy)

            if best_start is None:
                self.get_logger().warn('RRT*: no traversable start found near robot.')
                return None

            sx, sy = best_start
            self.get_logger().info(
                f'RRT*: moved start to nearest free cell at '
                f'({sx:.2f}, {sy:.2f}), grid_dist={best_dist_start:.2f}'
            )

        
        if self.is_in_collision(gx, gy, allow_unknown=True):
            goal_cell = self.world_to_map(gx, gy)
            if goal_cell is None:
                self.get_logger().warn('RRT*: goal in collision and outside map.')
                return None
            gx_idx, gy_idx = goal_cell

            best_goal = None
            best_dist_goal = float('inf')
            search_r_goal = max(self.inflation_cells * 2, 8)

            for dy in range(-search_r_goal, search_r_goal + 1):
                for dx in range(-search_r_goal, search_r_goal + 1):
                    nx = gx_idx + dx
                    ny = gy_idx + dy
                    if nx < 0 or ny < 0 or nx >= self.map_width or ny >= self.map_height:
                        continue
                    wx, wy = self.map_to_world(nx, ny)
                    if self.is_in_collision(wx, wy, allow_unknown=True):
                        continue
                    d = math.hypot(dx, dy)
                    if d < best_dist_goal:
                        best_dist_goal = d
                        best_goal = (wx, wy)

            if best_goal is None:
                self.get_logger().warn('RRT*: no traversable goal found near goal.')
                return None

            gx, gy = best_goal
            self.get_logger().info(
                f'RRT*: moved goal to nearest free cell at '
                f'({gx:.2f}, {gy:.2f}), grid_dist={best_dist_goal:.2f}'
            )

        
        if bounds is None:
            bounds = (self.map_origin_x,
                      self.map_origin_x + self.map_width * self.map_resolution,
                      self.map_origin_y,
                      self.map_origin_y + self.map_height * self.map_resolution)

        xmin, xmax, ymin, ymax = bounds
        xmin = min(xmin, sx, gx)
        xmax = max(xmax, sx, gx)
        ymin = min(ymin, sy, gy)
        ymax = max(ymax, sy, gy)

        nodes: List["Task2._RRTStarNode"] = [
            self._RRTStarNode(sx, sy, parent=-1, cost=0.0)
        ]
        best_goal_idx: Optional[int] = None

        
        first_goal_it: Optional[int] = None
        extra_after_goal = 80

        for it in range(max_iter):
            
            if random.random() < 0.1:
                rx, ry = gx, gy
            else:
                rx = random.uniform(xmin, xmax)
                ry = random.uniform(ymin, ymax)

            
            nearest_idx = 0
            nearest_dist = float('inf')
            for i, node in enumerate(nodes):
                d = self.dist(node.x, node.y, rx, ry)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_idx = i
            nearest = nodes[nearest_idx]

            
            dx = rx - nearest.x
            dy = ry - nearest.y
            d = math.hypot(dx, dy)
            if d < 1e-6:
                continue
            scale = min(step_size, d) / d
            new_x = nearest.x + dx * scale
            new_y = nearest.y + dy * scale

            
            if not self.collision_free(nearest.x, nearest.y,
                                       new_x, new_y,
                                       step=0.05,
                                       allow_unknown=True):
                continue

            
            neighbors: List[int] = []
            for i, node in enumerate(nodes):
                if self.dist(node.x, node.y, new_x, new_y) <= search_radius:
                    neighbors.append(i)

            
            best_parent_idx = nearest_idx
            best_cost = nearest.cost + self.dist(nearest.x, nearest.y, new_x, new_y)

            for ni in neighbors:
                n = nodes[ni]
                edge_cost = self.dist(n.x, n.y, new_x, new_y)
                if not self.collision_free(n.x, n.y, new_x, new_y,
                                           step=0.05,
                                           allow_unknown=True):
                    continue
                cand_cost = n.cost + edge_cost
                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best_parent_idx = ni

            new_node = self._RRTStarNode(new_x, new_y,
                                         parent=best_parent_idx,
                                         cost=best_cost)
            nodes.append(new_node)
            new_idx = len(nodes) - 1

            
            for ni in neighbors:
                n = nodes[ni]
                edge_cost = self.dist(new_node.x, new_node.y, n.x, n.y)
                if not self.collision_free(new_node.x, new_node.y, n.x, n.y,
                                           step=0.05,
                                           allow_unknown=True):
                    continue
                cand_cost = new_node.cost + edge_cost
                if cand_cost < n.cost:
                    n.cost = cand_cost
                    n.parent = new_idx

            
            d_goal = self.dist(new_x, new_y, gx, gy)
            if d_goal < goal_radius:
                if best_goal_idx is None or new_node.cost < nodes[best_goal_idx].cost:
                    best_goal_idx = new_idx
                    if first_goal_it is None:
                        first_goal_it = it

            
            if first_goal_it is not None and it - first_goal_it >= extra_after_goal:
                self.get_logger().info(
                    f'RRT*: early stop at it={it}, '
                    f'first_goal_it={first_goal_it}, '
                    f'best_goal_idx={best_goal_idx}'
                )
                break

            
            if it % 50 == 0:
                self.publish_rrt_tree(nodes)

        if best_goal_idx is None:
            self.get_logger().warn('RRT*: no path found to goal region.')
            return None

        
        self.publish_rrt_tree(nodes)

        
        path: List[Tuple[float, float]] = []
        idx = best_goal_idx
        while idx != -1:
            node = nodes[idx]
            path.append((node.x, node.y))
            idx = node.parent
        path.reverse()

        
        last_x, last_y = path[-1]
        if self.collision_free(last_x, last_y, gx, gy,
                               step=0.05,
                               allow_unknown=True):
            path.append((gx, gy))
        else:
            self.get_logger().info(
                'RRT*: last segment to exact goal is blocked, '
                'using nearest node in goal region as local goal.'
            )

        return path

    

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
    node = Task2()
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

