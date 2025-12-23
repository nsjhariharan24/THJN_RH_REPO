#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def get_nearest_lane_yaw(yaw):
    candidates = [0.0, math.pi/2, -math.pi/2, math.pi, -math.pi]
    return min(candidates, key=lambda a: abs(normalize_angle(yaw - a)))


class LidarReactiveNavigator(Node):
    def __init__(self):
        super().__init__('lidar_reactive_navigation')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # ===== PARAMETERS =====
        self.front_thresh = 0.35
        self.forward_speed = 0.5
        self.angular_speed = 0.5
        self.heading_kp = 1.5

        # ===== STATES =====
        self.state = "FORWARD"   # FORWARD, TURNING, RETURN_HOME

        # ===== RETURN SUB-STATES =====
        self.return_state = "GO_X"   # GO_X, TURN_LEFT, GO_Y

        # ===== ODOM DATA =====
        self.current_yaw = None
        self.current_x = None
        self.current_y = None

        # ===== HOME POSE =====
        self.home_x = None
        self.home_y = None
        self.home_yaw = None
        self.home_recorded = False

        # ===== TURN & LANE COUNT =====
        self.target_yaw = None
        self.lane_turn_count = 0
        self.MAX_TURNS = 6

        self.get_logger().info("Lane-aligned navigation started")

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        self.current_yaw = quaternion_to_yaw(q)
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        if not self.home_recorded:
            self.home_x = self.current_x
            self.home_y = self.current_y
            self.home_yaw = get_nearest_lane_yaw(self.current_yaw)
            self.home_recorded = True

            self.get_logger().info(
                f"HOME RECORDED → x={self.home_x:.2f}, y={self.home_y:.2f}"
            )

    def scan_callback(self, msg):

        if self.current_yaw is None:
            return

        if self.state == "RETURN_HOME":
            self.return_to_home()
            return

        if self.state == "TURNING":
            self.handle_turn()
            return

        front_dist = self.get_front_distance(msg)

        if front_dist > self.front_thresh:
            self.move_forward_aligned()
        else:
            self.start_turn()

    def get_front_distance(self, msg):
        ranges = msg.ranges
        n = len(ranges)
        center = n // 2
        width = int(n * 15 / 360)

        front_ranges = ranges[center - width:center + width]
        valid = [r for r in front_ranges if not math.isinf(r) and not math.isnan(r)]

        return min(valid) if valid else float('inf')

    def move_forward_aligned(self):
        twist = Twist()
        twist.linear.x = self.forward_speed

        desired_yaw = get_nearest_lane_yaw(self.current_yaw)
        yaw_error = normalize_angle(desired_yaw - self.current_yaw)

        twist.angular.z = self.heading_kp * yaw_error
        self.cmd_pub.publish(twist)

    def start_turn(self):
        self.stop_robot()
        self.target_yaw = normalize_angle(self.current_yaw - math.pi / 2)
        self.state = "TURNING"

    def handle_turn(self):
        error = normalize_angle(self.target_yaw - self.current_yaw)

        if abs(error) > 0.03:
            twist = Twist()
            twist.angular.z = -self.angular_speed
            self.cmd_pub.publish(twist)
        else:
            self.stop_robot()
            self.state = "FORWARD"
            self.lane_turn_count += 1

            self.get_logger().info(
                f"90° turn completed → count={self.lane_turn_count}"
            )
            self.get_logger().info(f"x={self.current_x:.2f}, y={self.current_y:.2f}, yaw={self.current_yaw:.2f}")

            if self.lane_turn_count > self.MAX_TURNS:
                self.get_logger().info("7 turns done → RETURNING HOME")
                self.state = "RETURN_HOME"
                self.return_state = "GO_X"

    # ✅ GRID-BASED RETURN HOME (AS REQUESTED)
    def return_to_home(self):
        twist = Twist()

        # ---- STEP 1: GO TO HOME X ----
        if self.return_state == "GO_X":
            dx = self.home_x - self.current_x - 0.25

            if abs(dx) < 0.15:
                self.stop_robot()
                self.return_state = "TURN_LEFT"
                self.target_yaw = normalize_angle(self.current_yaw + math.pi / 2)
                return

            target_yaw = 0.0 if dx > 0 else math.pi
            yaw_error = normalize_angle(target_yaw - self.current_yaw)

            if abs(yaw_error) > 0.1:
                twist.angular.z = 0.8 * yaw_error
            else:
                twist.linear.x = self.forward_speed * 0.6

        # ---- STEP 2: TURN LEFT 90° ----
        elif self.return_state == "TURN_LEFT":
            error = normalize_angle(self.target_yaw - self.current_yaw)

            if abs(error) > 0.03:
                twist.angular.z = self.angular_speed
            else:
                self.stop_robot()
                self.return_state = "GO_Y"

        # ---- STEP 3: GO TO HOME Y ----
        elif self.return_state == "GO_Y":
            dy = self.home_y - self.current_y - 0.1

            if abs(dy) < 0.15:
                self.stop_robot()
                self.get_logger().info(
                    f"HOME REACHED → x={self.current_x:.2f}, y={self.current_y:.2f}"
                )
                return

            target_yaw = math.pi / 2 if dy > 0 else -math.pi / 2
            yaw_error = normalize_angle(target_yaw - self.current_yaw)

            if abs(yaw_error) > 0.1:
                twist.angular.z = 0.8 * yaw_error
            else:
                twist.linear.x = self.forward_speed * 0.6

        self.cmd_pub.publish(twist)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = LidarReactiveNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

