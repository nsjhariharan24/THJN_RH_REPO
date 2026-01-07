#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray

BASE_LINK = "base_link"

INITIAL_POS = [0.406, 0.010, 0.582]
WAYPOINT    = [0.0,   0.5,   0.5]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class SimpleTwoPointTest(Node):
    def __init__(self):
        super().__init__("simple_two_point_test")

        # publisher: cartesian velocity
        self.pub_twist = self.create_publisher(
            TwistStamped, "/delta_twist_cmds", 10
        )

        # subscriber: tcp pose
        self.tcp_pose = None
        self.create_subscription(
            Float64MultiArray,
            "/tcp_pose_raw",
            self.cb_tcp_pose,
            10
        )

        # control params
        self.lin_gain = 0.8
        self.max_lin  = 0.05   # safe speed
        self.pos_tol  = 0.02

        self.phase = "INIT"  # INIT -> GO_INIT -> GO_WP -> DONE
        self.timer = self.create_timer(0.1, self.loop)

        self.get_logger().info("Two-point hardware test started")

    def cb_tcp_pose(self, msg: Float64MultiArray):
        self.tcp_pose = msg.data

    def get_tool_pos(self):
        if self.tcp_pose is None or len(self.tcp_pose) < 3:
            return None
        return [self.tcp_pose[0], self.tcp_pose[1], self.tcp_pose[2]]

    def publish_twist(self, lx, ly, lz):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_LINK
        msg.twist.linear.x = lx
        msg.twist.linear.y = ly
        msg.twist.linear.z = lz
        self.pub_twist.publish(msg)

    def stop(self):
        self.publish_twist(0.0, 0.0, 0.0)

    def go_to_pos(self, target):
        pos = self.get_tool_pos()
        if pos is None:
            self.stop()
            return False

        err = [target[i] - pos[i] for i in range(3)]
        dist = math.sqrt(sum(e * e for e in err))
        if dist < self.pos_tol:
            self.stop()
            return True

        vx = clamp(err[0] * self.lin_gain, -self.max_lin, self.max_lin)
        vy = clamp(err[1] * self.lin_gain, -self.max_lin, self.max_lin)
        vz = clamp(err[2] * self.lin_gain, -self.max_lin, self.max_lin)
        self.publish_twist(vx, vy, vz)
        return False

    def loop(self):
        if self.phase == "INIT":
            # wait for tcp_pose_raw
            if self.get_tool_pos() is not None:
                self.get_logger().info("TCP pose available, going to INITIAL_POS")
                self.phase = "GO_INIT"
            return

        if self.phase == "GO_INIT":
            if self.go_to_pos(INITIAL_POS):
                self.get_logger().info("Reached INITIAL_POS, going to WAYPOINT")
                self.phase = "GO_WP"
            return

        if self.phase == "GO_WP":
            if self.go_to_pos(WAYPOINT):
                self.get_logger().info("Reached WAYPOINT, test DONE")
                self.phase = "DONE"
            return

        if self.phase == "DONE":
            self.stop()
            return


def main():
    rclpy.init()
    node = SimpleTwoPointTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
