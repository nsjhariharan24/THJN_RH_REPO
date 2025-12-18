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


class PickPlaceCartesian(Node):
    def __init__(self):
        super().__init__("task3b_pick_place_cartesian")

        self.twist_pub = self.create_publisher(
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

        self.tcp_pos = None
        self.force_z = 0.0

        self.bad_fruits = {}
        self.fertilizer_pos = None
        self.drop_pos = None

        self.current_fruit = 1
        self.state = "WAIT_TF"

        self.timer = self.create_timer(0.1, self.loop)

        self.get_logger().info("✅ Pick & Place node started")

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
        self.twist_pub.publish(stop)

    def set_magnet(self, state):
        req = SetBool.Request()
        req.data = state
        self.magnet.call_async(req)

    def go_to_pos(self, target):
        if self.tcp_pos is None:
            return False

        err = [target[i] - self.tcp_pos[i] for i in range(3)]
        dist = math.sqrt(sum(e * e for e in err))

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()

        if dist < self.pos_tol:
            self.stop_motion()
            return True

        cmd.twist.linear.x = clamp(err[0] * self.lin_gain, -self.max_lin, self.max_lin)
        cmd.twist.linear.y = clamp(err[1] * self.lin_gain, -self.max_lin, self.max_lin)
        cmd.twist.linear.z = clamp(err[2] * self.lin_gain, -self.max_lin, self.max_lin)

        self.twist_pub.publish(cmd)
        return False

    def loop(self):
        s = self.state

        if s == "WAIT_TF":
            if not self.bad_fruits:
                for i in range(1, 4):
                    p = self.get_tf_pos(f"{TEAM_ID}_bad_fruit_{i}")
                    if p:
                        self.bad_fruits[i] = p

                self.fertilizer_pos = self.get_tf_pos(f"{TEAM_ID}_fertilizer_1")
                self.drop_pos = self.get_tf_pos(f"{TEAM_ID}_drop_zone")

            if len(self.bad_fruits) == 3 and self.fertilizer_pos and self.drop_pos:
                self.state = "PICK_FRUIT"
            return

        if s == "PICK_FRUIT":
            p = self.bad_fruits[self.current_fruit]

            self.go_to_pos(p)

            if self.force_z > self.force_pick_threshold:
                self.stop_motion()
                self.set_magnet(True)
                time.sleep(0.2)
                self.state = "LIFT_FRUIT"
            return

        if s == "LIFT_FRUIT":
            p = self.bad_fruits[self.current_fruit]
            if self.go_to_pos([p[0], p[1], p[2] + 0.15]):
                self.state = "DROP_FRUIT"
            return

        if s == "DROP_FRUIT":
            if self.go_to_pos([self.drop_pos[0], self.drop_pos[1], self.drop_pos[2] + 0.15]):
                self.set_magnet(False)
                time.sleep(0.2)
                if self.current_fruit < 3:
                    self.current_fruit += 1
                    self.state = "PICK_FRUIT"
                else:
                    self.state = "PICK_CAN"
            return

        if s == "PICK_CAN":
            self.go_to_pos(self.fertilizer_pos)
            if self.force_z > self.force_pick_threshold:
                self.stop_motion()
                self.set_magnet(True)
                time.sleep(0.2)
                self.state = "LIFT_CAN"
            return

        if s == "LIFT_CAN":
            if self.go_to_pos([self.fertilizer_pos[0],
                               self.fertilizer_pos[1],
                               self.fertilizer_pos[2] + 0.2]):
                self.state = "DROP_CAN"
            return

        if s == "DROP_CAN":
            if self.go_to_pos([self.drop_pos[0], self.drop_pos[1], self.drop_pos[2] + 0.2]):
                self.set_magnet(False)
                self.state = "FINISH"
            return

        if s == "FINISH":
            self.stop_motion()
            self.get_logger().info("✅ TASK FINISHED")
            return


def main():
    rclpy.init()
    node = PickPlaceCartesian()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
