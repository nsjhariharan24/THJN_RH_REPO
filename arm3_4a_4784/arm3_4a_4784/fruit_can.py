#!/usr/bin/env python3
'''
Team ID:    4784
Theme:      Krishi coBot
Filename:   task4a_fruit_then_can_hw_final.py

Hardware controller for Task 4A:
- Fruits (bad_fruit_1..NUM_FRUITS) → fertilizer can
- Uses:
  - /delta_twist_cmds (geometry_msgs/TwistStamped)
  - /tcp_pose_raw (std_msgs/Float64MultiArray)
  - /magnet (std_srvs/SetBool)
  - TF frames from perception: 4784_bad_fruit_i, 4784_fertilizer_1
'''

import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener

TEAM_ID = 4784
BASE_LINK = "base_link"
NUM_FRUITS = 3  # as per your plan


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Task4AArmFruitThenCanHW(Node):

    def __init__(self):
        super().__init__("task4a_arm_fruit_then_can_hw_final")

        # ------------- Publishers -------------
        self.pub_twist = self.create_publisher(
            TwistStamped, "/delta_twist_cmds", 10
        )
        self.pub_arm_done = self.create_publisher(Bool, "/arm_done_drop", 10)

        # ------------- Subscribers -------------
        self.tcp_pose = None  # [x, y, z, rx, ry, rz]
        self.create_subscription(
            Float64MultiArray,
            "/tcp_pose_raw",
            self.cb_tcp_pose,
            10
        )

        # ------------- Magnet client -------------
        self.magnet_cli = self.create_client(SetBool, "/magnet")
        while not self.magnet_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /magnet service...")

        # ------------- TF for object frames -------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ------------- Positions (tune as needed) -------------
        self.initial_pos = [0.406, 0.010, 0.582]
        self.waypoint    = [0.0,   0.5,   0.5]
        self.pre_bin_pos = [-0.806, 0.010, 0.402]

        # For can sequence, reuse your P1 point
        self.P1_POS  = [-0.214, -0.332, 0.557]

        # ------------- Control params -------------
        self.lin_gain = 0.8
        self.max_lin  = 0.05   # reduced for hardware safety
        self.pos_tol  = 0.02

        # ------------- State variables -------------
        self.fruit_pos = {}
        self.current_fruit = 1

        self.can_pos = None

        self.state = "WAIT_TF_FRUIT"
        self.timer = self.create_timer(0.1, self.loop)

        self.get_logger().info("🍎 Fruits → 🟡 Can FSM started (HW, tcp_pose_raw, magnet)")

    # ================= TCP pose =================
    def cb_tcp_pose(self, msg: Float64MultiArray):
        self.tcp_pose = msg.data

    def get_tool_pos(self):
        """
        Returns current TCP position [x, y, z] from /tcp_pose_raw.
        """
        if self.tcp_pose is None or len(self.tcp_pose) < 3:
            return None
        return [self.tcp_pose[0], self.tcp_pose[1], self.tcp_pose[2]]

    # ================= TF helpers =================
    def get_tf_pos(self, frame):
        """
        Returns [x, y, z] of a TF frame in base_link.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_LINK, frame, rclpy.time.Time()
            )
            t = tf.transform.translation
            return [t.x, t.y, t.z]
        except Exception:
            return None

    # ================= Motion helpers =================
    def publish_twist(self, lx, ly, lz, ax=0.0, ay=0.0, az=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_LINK
        msg.twist.linear.x = lx
        msg.twist.linear.y = ly
        msg.twist.linear.z = lz
        msg.twist.angular.x = ax
        msg.twist.angular.y = ay
        msg.twist.angular.z = az
        self.pub_twist.publish(msg)

    def stop_cartesian(self):
        self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def go_to_pos(self, target):
        """
        Move TCP toward target [x,y,z] using P-control on /delta_twist_cmds.
        Returns True when within pos_tol.
        """
        pos = self.get_tool_pos()
        if pos is None:
            self.stop_cartesian()
            return False

        err = [target[i] - pos[i] for i in range(3)]
        dist = math.sqrt(sum(e * e for e in err))
        if dist < self.pos_tol:
            self.stop_cartesian()
            return True

        vx = clamp(err[0] * self.lin_gain, -self.max_lin, self.max_lin)
        vy = clamp(err[1] * self.lin_gain, -self.max_lin, self.max_lin)
        vz = clamp(err[2] * self.lin_gain, -self.max_lin, self.max_lin)

        self.publish_twist(vx, vy, vz)
        return False

    # ================= Magnet =================
    def set_magnet(self, on: bool):
        req = SetBool.Request()
        req.data = on
        future = self.magnet_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if future.result() is None or not future.result().success:
            self.get_logger().warn("Magnet command may have failed")

    # ================= FSM =================
    def loop(self):
        s = self.state
        # self.get_logger().info(f"State: {s}")

        # ---------- FRUITS ----------
        if s == "WAIT_TF_FRUIT":
            self.fruit_pos.clear()
            for i in range(1, NUM_FRUITS + 1):
                self.fruit_pos[i] = self.get_tf_pos(f"{TEAM_ID}_bad_fruit_{i}")
            if all(self.fruit_pos.values()):
                self.get_logger().info("Bad fruit TFs found, starting fruit sequence")
                self.state = "FRUIT_INIT"
            return

        if s == "FRUIT_INIT":
            if self.go_to_pos(self.initial_pos):
                self.state = "FRUIT_WP"
            return

        if s == "FRUIT_WP":
            if self.go_to_pos(self.waypoint):
                self.state = "MOVE_FRUIT_ABOVE"
            return

        if s == "MOVE_FRUIT_ABOVE":
            p = self.fruit_pos[self.current_fruit]
            if p is None:
                self.get_logger().warn("Fruit TF missing in MOVE_FRUIT_ABOVE")
                return
            target_above = [p[0], p[1], p[2] + 0.10]
            if self.go_to_pos(target_above):
                self.state = "MOVE_FRUIT_DOWN"
            return

        if s == "MOVE_FRUIT_DOWN":
            p = self.fruit_pos[self.current_fruit]
            if p is None:
                self.get_logger().warn("Fruit TF missing in MOVE_FRUIT_DOWN")
                return
            # Slightly above exact TF to avoid crushing
            target_touch = [p[0], p[1], p[2] + 0.01]
            if self.go_to_pos(target_touch):
                self.set_magnet(True)
                time.sleep(0.3)
                self.state = "LIFT_FRUIT"
            return

        if s == "LIFT_FRUIT":
            p = self.fruit_pos[self.current_fruit]
            if p is None:
                self.get_logger().warn("Fruit TF missing in LIFT_FRUIT")
                return
            if self.go_to_pos([p[0], p[1], p[2] + 0.15]):
                self.state = "DROP_FRUIT"
            return

        if s == "DROP_FRUIT":
            if self.go_to_pos(self.pre_bin_pos):
                self.set_magnet(False)
                time.sleep(0.3)
                if self.current_fruit < NUM_FRUITS:
                    self.current_fruit += 1
                    self.state = "FRUIT_WP"
                else:
                    self.state = "WAIT_TF_CAN"
            return

        # ---------- CAN ----------
        if s == "WAIT_TF_CAN":
            self.can_pos = self.get_tf_pos(f"{TEAM_ID}_fertilizer_1")
            if self.can_pos:
                self.get_logger().info("Fertilizer TF found, starting can sequence")
                self.state = "CAN_INIT"
            return

        if s == "CAN_INIT":
            if self.go_to_pos(self.initial_pos):
                self.state = "CAN_P1"
            return

        if s == "CAN_P1":
            if self.go_to_pos(self.P1_POS):
                self.state = "MOVE_CAN_ABOVE"
            return

        if s == "MOVE_CAN_ABOVE":
            p = self.can_pos
            if p is None:
                self.get_logger().warn("Can TF missing in MOVE_CAN_ABOVE")
                return
            target_above = [p[0], p[1], p[2] + 0.10]
            if self.go_to_pos(target_above):
                self.state = "MOVE_CAN_DOWN"
            return

        if s == "MOVE_CAN_DOWN":
            p = self.can_pos
            if p is None:
                self.get_logger().warn("Can TF missing in MOVE_CAN_DOWN")
                return
            target_touch = [p[0], p[1], p[2] + 0.01]
            if self.go_to_pos(target_touch):
                self.set_magnet(True)
                time.sleep(0.3)
                self.state = "SHIFT_CAN_Y"
            return

        if s == "SHIFT_CAN_Y":
            tgt = [self.can_pos[0], self.can_pos[1] + 0.3, self.can_pos[2] + 0.10]
            if self.go_to_pos(tgt):
                self.state = "MOVE_CAN_TO_BIN"
            return

        if s == "MOVE_CAN_TO_BIN":
            if self.go_to_pos(self.pre_bin_pos):
                self.state = "DROP_CAN"
            return

        if s == "DROP_CAN":
            self.set_magnet(False)
            time.sleep(0.3)
            self.pub_arm_done.publish(Bool(data=True))
            self.state = "FINISH"
            return

        if s == "FINISH":
            self.stop_cartesian()
            self.get_logger().info("✅ Fruits + Can Task 4A completed (HW)")
            return


def main():
    rclpy.init()
    node = Task4AArmFruitThenCanHW()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.stop_cartesian()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
