#!/usr/bin/env python3
'''
Team ID:    4784
Theme:      Krishi coBot
Filename:   task4a_fruit_then_can_hw_final_pitch.py

Hardware controller for Task 4A:
- Fruits (bad_fruit_1..NUM_FRUITS) → fertilizer can
- Uses:
  - /delta_twist_cmds (geometry_msgs/TwistStamped)
  - /tcp_pose_raw (std_msgs/Float64MultiArray) for position
  - TF base_link -> tool0 for orientation
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
import tf_transformations


TEAM_ID = 4784
BASE_LINK = "base_link"
TCP_FRAME = "tool0"       # change if your EE frame name is different
NUM_FRUITS = 3


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Task4AArmFruitThenCanHW(Node):

    def __init__(self):
        super().__init__("task4a_arm_fruit_then_can_hw_final_pitch")

        # ---------- Publishers ----------
        self.pub_twist = self.create_publisher(
            TwistStamped, "/delta_twist_cmds", 10
        )
        self.pub_arm_done = self.create_publisher(Bool, "/arm_done_drop", 10)

        # ---------- Subscribers ----------
        self.tcp_pose = None  # [x, y, z, rx, ry, rz]
        self.create_subscription(
            Float64MultiArray,
            "/tcp_pose_raw",
            self.cb_tcp_pose,
            10
        )

        # ---------- Magnet client ----------
        self.magnet_cli = self.create_client(SetBool, "/magnet")
        while not self.magnet_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /magnet service...")

        # ---------- TF (for orientation + object frames) ----------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------- Positions ----------
        self.initial_pos = [0.406, 0.010, 0.582]
        self.waypoint    = [0.0,   0.5,   0.5]
        self.pre_bin_pos = [-0.806, 0.010, 0.402]
        self.P1_POS      = [-0.214, -0.332, 0.557]

        # ---------- Control ----------
        self.lin_gain = 0.8
        self.max_lin  = 0.05
        self.pos_tol  = 0.02

        self.ang_gain = 0.6
        self.max_ang  = 0.5
        self.ori_tol  = 0.15

        # ---------- Orientations ----------
        # Initial pitch: (0, pi, -pi/2)
        self.q_init_pitch = tf_transformations.quaternion_from_euler(
            0.0, math.pi, -math.pi / 2.0
        )
        # Can orientation: given quaternion
        self.q_can = [0.707, 0.028, 0.034, 0.707]

        # ---------- State ----------
        self.fruit_pos = {}
        self.current_fruit = 1
        self.can_pos = None

        self.state = "WAIT_TF_FRUIT"
        self.timer = self.create_timer(0.1, self.loop)

        self.get_logger().info("🍎 Fruits → 🟡 Can FSM (HW, pitch + can quat) started")

    # ================= TCP pose =================
    def cb_tcp_pose(self, msg: Float64MultiArray):
        self.tcp_pose = msg.data

    def get_tool_pos(self):
        if self.tcp_pose is None or len(self.tcp_pose) < 3:
            return None
        return [self.tcp_pose[0], self.tcp_pose[1], self.tcp_pose[2]]

    # ================= TF helpers =================
    def get_tool_quat(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_LINK, TCP_FRAME, rclpy.time.Time()
            )
            q = tf.transform.rotation
            return [q.x, q.y, q.z, q.w]
        except Exception:
            return None

    def get_tf_pos(self, frame):
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

    def go_to_ori(self, q_target):
        q = self.get_tool_quat()
        if q is None:
            self.stop_cartesian()
            return False

        q_err = tf_transformations.quaternion_multiply(
            q_target, tf_transformations.quaternion_inverse(q)
        )
        ang = 2 * math.acos(clamp(q_err[3], -1.0, 1.0))

        if ang < self.ori_tol:
            self.stop_cartesian()
            return True

        s = math.sin(ang / 2.0)
        if abs(s) < 1e-6:
            self.stop_cartesian()
            return True

        axis = [q_err[i] / s for i in range(3)]
        wx = clamp(axis[0] * ang * self.ang_gain, -self.max_ang, self.max_ang)
        wy = clamp(axis[1] * ang * self.ang_gain, -self.max_ang, self.max_ang)
        wz = clamp(axis[2] * ang * self.ang_gain, -self.max_ang, self.max_ang)
        self.publish_twist(0.0, 0.0, 0.0, wx, wy, wz)
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

        # ---------- FRUITS ----------
        if s == "WAIT_TF_FRUIT":
            self.fruit_pos.clear()
            for i in range(1, NUM_FRUITS + 1):
                self.fruit_pos[i] = self.get_tf_pos(f"{TEAM_ID}_bad_fruit_{i}")
            if all(self.fruit_pos.values()):
                self.get_logger().info("Bad fruit TFs found")
                self.state = "INIT_POS"
            return

        if s == "INIT_POS":
            if self.go_to_pos(self.initial_pos):
                self.state = "INIT_PITCH"
            return

        if s == "INIT_PITCH":
            # Set orientation at initial pose: (0, pi, -pi/2)
            if self.go_to_ori(self.q_init_pitch):
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
                self.get_logger().info("Fertilizer TF found")
                self.state = "CAN_INIT"
            return

        if s == "CAN_INIT":
            if self.go_to_pos(self.initial_pos):
                self.state = "CAN_P1_POS"
            return

        if s == "CAN_P1_POS":
            if self.go_to_pos(self.P1_POS):
                self.state = "CAN_P1_ORI"
            return

        if s == "CAN_P1_ORI":
            # Enforce given can orientation quaternion at P1
            if self.go_to_ori(self.q_can):
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
            self.get_logger().info("✅ Fruits + Can Task 4A completed (HW, pitch + can quat)")
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
