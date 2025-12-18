#!/usr/bin/env python3

import math
import time
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32, Float64MultiArray
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener

TEAM_ID = 4784


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Task3BArmHardware(Node):

    def __init__(self):
        super().__init__("task3b_fruit_drop_hardware")

        self.pub_twist = self.create_publisher(
            TwistStamped, "/delta_twist_cmds", 10
        )

        self.create_subscription(
            Float64MultiArray, "/tcp_pose_raw", self.cb_tcp_pose, 10
        )

        self.create_subscription(
            Float32, "/net_wrench", self.cb_force, 10
        )

        self.magnet = self.create_client(SetBool, "/magnet")
        while not self.magnet.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for magnet service...")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # SAFE SPEEDS
        self.lin_gain = 0.5
        self.max_lin = 0.08
        self.pos_tol = 0.04
        self.force_pick_threshold = 12.0

        self.initial_pos = [0.406, 0.010, 0.582]
        self.waypoint = [0.0, 0.5, 0.55]
        self.wp_z = self.waypoint[2]
        self.lift_height = 0.15

        self.tcp_pos = None
        self.force_z = 0.0

        self.fruit_pos = {}
        self.drop_pos = None

        self.current_fruit = 1
        self.state = "WAIT_TF"

        self.timer = self.create_timer(0.1, self.loop)

    def cb_tcp_pose(self, msg):
        self.tcp_pos = msg.data[:3]

    def cb_force(self, msg):
        self.force_z = msg.data

    def get_tf_pos(self, frame):
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", frame, rclpy.time.Time()
            )
            t = tf.transform.translation
            return [t.x, t.y, t.z]
        except Exception:
            return None

    def stop_motion(self):
        stop = TwistStamped()
        stop.header.stamp = self.get_clock().now().to_msg()
        self.pub_twist.publish(stop)

    def set_magnet(self, state):
        req = SetBool.Request()
        req.data = state
        self.magnet.call_async(req)

    def go_to_pos(self, target):
        if self.tcp_pos is None:
            return False

        err = [target[i] - self.tcp_pos[i] for i in range(3)]
        dist = math.sqrt(sum(e * e for e in err))

        if dist < self.pos_tol:
            self.stop_motion()
            return True

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.twist.linear.x = clamp(err[0] * self.lin_gain, -self.max_lin, self.max_lin)
        cmd.twist.linear.y = clamp(err[1] * self.lin_gain, -self.max_lin, self.max_lin)
        cmd.twist.linear.z = clamp(err[2] * self.lin_gain, -self.max_lin, self.max_lin)

        self.pub_twist.publish(cmd)
        return False

    def loop(self):
        s = self.state

        if s == "WAIT_TF":
            if not self.fruit_pos:
                for i in range(1, 4):
                    p = self.get_tf_pos(f"{TEAM_ID}_bad_fruit_{i}")
                    if p:
                        self.fruit_pos[i] = p
                self.drop_pos = self.get_tf_pos(f"{TEAM_ID}_drop_zone")

            if len(self.fruit_pos) == 3 and self.drop_pos:
                self.state = "MOVE_INIT"
            return

        if s == "MOVE_INIT":
            if self.go_to_pos(self.initial_pos):
                self.state = "MOVE_WP"
            return

        if s == "MOVE_WP":
            if self.go_to_pos(self.waypoint):
                self.state = "MOVE_FRUIT"
            return

        if s == "MOVE_FRUIT":
            p = self.fruit_pos[self.current_fruit]
            self.go_to_pos(p)
            if self.force_z > self.force_pick_threshold:
                self.stop_motion()
                self.set_magnet(True)
                time.sleep(0.2)
                self.state = "LIFT_FRUIT"
            return

        if s == "LIFT_FRUIT":
            p = self.fruit_pos[self.current_fruit]
            if self.go_to_pos([p[0], p[1], p[2] + self.lift_height]):
                self.state = "MOVE_DROP_XY"
            return

        if s == "MOVE_DROP_XY":
            tgt = [self.drop_pos[0], self.drop_pos[1], self.tcp_pos[2]]
            if self.go_to_pos(tgt):
                self.state = "DROP_Z"
            return

        if s == "DROP_Z":
            tgt = [self.drop_pos[0], self.drop_pos[1], self.drop_pos[2] + 0.15]
            if self.go_to_pos(tgt):
                self.state = "DROP_FRUIT"
            return

        if s == "DROP_FRUIT":
            self.set_magnet(False)
            time.sleep(0.2)
            self.current_fruit += 1
            if self.current_fruit > 3:
                self.state = "FINISH"
            else:
                self.state = "MOVE_WP"
            return

        if s == "FINISH":
            self.stop_motion()
            self.get_logger().info("🎉 All fruits dropped")
            return


def main():
    rclpy.init()
    node = Task3BArmHardware()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
