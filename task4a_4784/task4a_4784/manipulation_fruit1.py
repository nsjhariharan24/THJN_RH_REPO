#!/usr/bin/env python3
'''
# Team ID:          4784
# Theme:            Krishi coBot
# Author List:      Hariharan S, Thilakraj S, Jeevan Uday Alexander, Niranjan R
# Filename:         task4a_arm_fruit_only_hw_tcp.py
#
# Hardware version for Task 4A:
#  - Uses /delta_twist_cmds (TwistStamped) for Cartesian servoing
#  - Uses /tcp_pose_raw (Float64MultiArray) for TCP pose
#  - Uses /magnet (std_srvs/SetBool) for electromagnet
#  - Monitors /net_wrench for safety
#  - FSM sequence reused from simulation (2 bad fruits)
'''

import math
import time
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32, Float64MultiArray
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener  # only for fruit TFs
import tf_transformations

TEAM_ID = 4784
BASE_LINK = "base_link"

NUM_FRUITS = 2               # HARDWARE: only 2 bad fruits
FORCE_LIMIT = 10.0           # tune based on /net_wrench scale


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Task4AArmFruitHW(Node):

    def __init__(self):
        super().__init__("task4a_arm_fruit_only_hw_tcp")

        # --- Publishers & clients ---
        self.pub_twist = self.create_publisher(
            TwistStamped, "/delta_twist_cmds", 10
        )

        # net force monitor
        self.net_force = 0.0
        self.create_subscription(
            Float32, "/net_wrench", self.net_force_cb, 10
        )

        # TCP pose from /tcp_pose_raw
        self.tcp_pose = None
        self.create_subscription(
            Float64MultiArray, "/tcp_pose_raw", self.tcp_pose_cb, 10
        )

        # magnet
        self.magnet_cli = self.create_client(SetBool, "/magnet")
        while not self.magnet_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /magnet service...")

        # --- TF (only for fruit positions from perception node) ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------- Positions (tune for hardware) ----------------
        self.initial_pos = [0.406, 0.010, 0.582]
        self.waypoint    = [0.0,   0.5,   0.5  ]
        self.pre_bin_pos = [-0.806, 0.010, 0.402]

        # ---------------- Control ----------------
        self.lin_gain = 0.8
        self.max_lin = 0.15
        self.pos_tol = 0.04

        # Orientation: we only enforce fixed pitch approximately.
        # /tcp_pose_raw gives [x, y, z, rx, ry, rz] (UR-style axis-angle),
        # but for simplicity we only control position here.
        self.q_pitch_fruit = tf_transformations.quaternion_from_euler(
            0, math.pi, 0
        )
        self.ang_gain = 0.6
        self.max_ang = 1.0
        self.ori_tol = 0.15

        # ---------------- State ----------------
        self.fruit_pos = {}
        self.current_fruit = 1

        self.state = "WAIT_TF"
        self.timer = self.create_timer(0.1, self.loop)

        self.get_logger().info("🍎 HW Fruit-only FSM started (2 fruits, tcp_pose_raw)")

    # ---------------- Callbacks ----------------

    def net_force_cb(self, msg: Float32):
        self.net_force = msg.data

    def tcp_pose_cb(self, msg: Float64MultiArray):
        # msg.data = [x, y, z, rx, ry, rz] in base frame
        if len(msg.data) >= 6:
            self.tcp_pose = msg.data

    # ---------------- Magnet helpers ----------------

    def set_magnet(self, on: bool):
        req = SetBool.Request()
        req.data = on
        future = self.magnet_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if future.result() is None or not future.result().success:
            self.get_logger().warn("Magnet command may have failed")

    # ---------------- Pose helpers ----------------
    def get_tool_pose(self):
        """
        Returns (position, None) using /tcp_pose_raw.
        Position: [x, y, z] in base frame.
        Orientation from tcp_pose_raw is ignored in this simple version.
        """
        if self.tcp_pose is None:
            return None, None
        x, y, z = self.tcp_pose[0], self.tcp_pose[1], self.tcp_pose[2]
        return [x, y, z], None

    def get_tf_pos(self, frame):
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_LINK, frame, rclpy.time.Time()
            )
            t = tf.transform.translation
            return [t.x, t.y, t.z]
        except Exception:
            return None

    # ---------------- Motion ----------------
    def publish_zero_twist(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_LINK
        self.pub_twist.publish(msg)

    def go_to_pos(self, target):
        # Safety: stop if force too high
        if self.net_force > FORCE_LIMIT:
            self.get_logger().warn(
                f"High force {self.net_force:.2f}, stopping motion"
            )
            self.publish_zero_twist()
            return False

        pos, _ = self.get_tool_pose()
        if pos is None:
            self.publish_zero_twist()
            return False

        err = [target[i] - pos[i] for i in range(3)]
        if math.sqrt(sum(e * e for e in err)) < self.pos_tol:
            self.publish_zero_twist()
            return True

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = BASE_LINK
        cmd.twist.linear.x = clamp(err[0] * self.lin_gain, -self.max_lin, self.max_lin)
        cmd.twist.linear.y = clamp(err[1] * self.lin_gain, -self.max_lin, self.max_lin)
        cmd.twist.linear.z = clamp(err[2] * self.lin_gain, -self.max_lin, self.max_lin)
        # No angular motion in this simplified version
        self.pub_twist.publish(cmd)
        return False

    # (Optional) orientation control using tcp_pose_raw could be added,
    # but Task 4A does not force you to track a specific orientation as long
    # as you pick/drop correctly.

    # ---------------- FSM (same sequence, now 2 fruits) ----------------
    def loop(self):

        s = self.state

        # ----- WAIT FOR FRUIT TFs -----
        if s == "WAIT_TF":
            self.fruit_pos.clear()
            for i in range(1, NUM_FRUITS + 1):   # 1..2
                self.fruit_pos[i] = self.get_tf_pos(f"{TEAM_ID}_bad_fruit_{i}")

            if all(self.fruit_pos.values()):
                # we skip explicit orientation control here and go directly to INIT pose
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

        # ----- FRUIT PICK & DROP -----
        if s == "MOVE_FRUIT":
            p = self.fruit_pos[self.current_fruit]
            if self.go_to_pos(p):
                # AT_FRUIT → magnet ON
                self.set_magnet(True)
                time.sleep(0.3)     # small wait while velocities are zero
                self.state = "LIFT_FRUIT"
            return

        if s == "LIFT_FRUIT":
            p = self.fruit_pos[self.current_fruit]
            if self.go_to_pos([p[0], p[1], p[2] + 0.15]):
                self.state = "DROP_FRUIT"
            return

        if s == "DROP_FRUIT":
            if self.go_to_pos(self.pre_bin_pos):
                # AT_BIN → magnet OFF
                self.set_magnet(False)
                time.sleep(0.3)

                if self.current_fruit < NUM_FRUITS:
                    self.current_fruit += 1
                    self.state = "MOVE_WP"
                else:
                    self.state = "FINISH"
            return

        if s == "FINISH":
            self.publish_zero_twist()
            self.get_logger().info(
                "✅ All fruits picked & dropped successfully (HW, 2 fruits, tcp_pose_raw)"
            )
            return


def main():
    rclpy.init()
    node = Task4AArmFruitHW()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_zero_twist()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
