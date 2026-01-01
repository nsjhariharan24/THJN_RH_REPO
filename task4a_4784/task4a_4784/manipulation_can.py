#!/usr/bin/env python3
'''
# Team ID:          4784
# Theme:            Krishi coBot
# Filename:         task4a_can_only_hw.py
#
# Hardware version for CAN pick & place:
#  - Uses /delta_twist_cmds (TwistStamped) for Cartesian servoing
#  - Uses /magnet (std_srvs/SetBool) instead of linkattacher + /grip_control
#  - Optionally uses /net_wrench for safety (can be added like in fruit code)
#  - FSM sequence reused from simulation
'''

import math
import time
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener
import tf_transformations

TEAM_ID = 4784
BASE_LINK = "base_link"
TCP_FRAME = "tool0"          # change if EE frame name is different

FORCE_LIMIT = 10.0           # optional; only if you add /net_wrench later


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ========================= ARM NODE =========================
class Task2BArmHW(Node):

    def __init__(self):
        super().__init__("task2b_arm_can_only_hw")

        # ---- Publishers ----
        self.pub_twist = self.create_publisher(
            TwistStamped, "/delta_twist_cmds", 10
        )
        self.pub_arm_done = self.create_publisher(Bool, "/arm_done_drop", 10)

        # ---- Magnet client ----
        self.magnet_cli = self.create_client(SetBool, "/magnet")
        while not self.magnet_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /magnet service...")

        # ---- TF ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------------- Fixed Positions ----------------
        self.initial_pos = [0.406, 0.010, 0.582]

        # Approach pose near rack (P1)
        self.P1_POS = [-0.214, -0.332, 0.557]
        self.P1_QUAT = [0.707, 0.028, 0.034, 0.707]  # fixed orientation near rack

        # Pre-bin pose (drop zone approach)
        self.pre_bin_pos = [-0.806, 0.010, 0.402]

        # ---------------- Control ----------------
        self.lin_gain = 0.8
        self.max_lin = 0.25
        self.pos_tol = 0.02

        self.ang_gain = 0.6
        self.ori_tol = 0.15

        # Fixed pitch for can
        self.q_pitch_can = tf_transformations.quaternion_from_euler(
            0, math.pi, 0
        )

        # ---------------- State ----------------
        self.can_pos = None
        self.state = "WAIT_TF"

        self.timer = self.create_timer(0.1, self.loop)

        self.get_logger().info("🟡 CAN FSM (hardware, magnet) started")

    # ---------------- Magnet helpers ----------------

    def set_magnet(self, on: bool):
        req = SetBool.Request()
        req.data = on
        future = self.magnet_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if future.result() is None or not future.result().success:
            self.get_logger().warn("Magnet command may have failed")

    # ---------------- TF helpers ----------------
    def get_tool_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_LINK, TCP_FRAME, rclpy.time.Time()
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            return [t.x, t.y, t.z], [q.x, q.y, q.z, q.w]
        except Exception:
            return None, None

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
        pos, _ = self.get_tool_pose()
        if pos is None:
            self.publish_zero_twist()
            return False

        err = [target[i] - pos[i] for i in range(3)]
        dist = math.sqrt(sum(e * e for e in err))

        if dist < self.pos_tol:
            self.publish_zero_twist()
            return True

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = BASE_LINK
        cmd.twist.linear.x = clamp(err[0] * self.lin_gain, -self.max_lin, self.max_lin)
        cmd.twist.linear.y = clamp(err[1] * self.lin_gain, -self.max_lin, self.max_lin)
        cmd.twist.linear.z = clamp(err[2] * self.lin_gain, -self.max_lin, self.max_lin)
        self.pub_twist.publish(cmd)
        return False

    def go_to_ori(self, q_target):
        _, q = self.get_tool_pose()
        if q is None:
            self.publish_zero_twist()
            return False

        q_err = tf_transformations.quaternion_multiply(
            q_target, tf_transformations.quaternion_inverse(q)
        )
        ang = 2 * math.acos(clamp(q_err[3], -1, 1))

        if ang < self.ori_tol:
            self.publish_zero_twist()
            return True

        s = math.sin(ang / 2)
        if abs(s) < 1e-6:
            self.publish_zero_twist()
            return True

        axis = [q_err[i] / s for i in range(3)]

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = BASE_LINK
        cmd.twist.angular.x = axis[0] * ang * self.ang_gain
        cmd.twist.angular.y = axis[1] * ang * self.ang_gain
        cmd.twist.angular.z = axis[2] * ang * self.ang_gain
        self.pub_twist.publish(cmd)
        return False

    # ---------------- FSM (same sequence) ----------------
    def loop(self):

        s = self.state

        # ----- WAIT FOR CAN TF -----
        if s == "WAIT_TF":
            self.can_pos = self.get_tf_pos(f"{TEAM_ID}_fertilizer_1")

            if self.can_pos:
                self.state = "FIXED_PITCH"
            return

        if s == "FIXED_PITCH":
            if self.go_to_ori(self.q_pitch_can):
                self.state = "MOVE_INIT"
            return

        if s == "MOVE_INIT":
            if self.go_to_pos(self.initial_pos):
                self.state = "MOVE_P1"
            return

        if s == "MOVE_P1":
            if self.go_to_pos(self.P1_POS) and self.go_to_ori(self.P1_QUAT):
                self.state = "MOVE_CAN"
            return

        if s == "MOVE_CAN":
            if self.go_to_pos(self.can_pos):
                # AT_CAN → magnet ON
                self.set_magnet(True)
                time.sleep(0.3)
                self.state = "SHIFT_CAN_Y"
            return

        if s == "SHIFT_CAN_Y":
            tgt = [
                self.can_pos[0],
                self.can_pos[1] + 0.3,
                self.can_pos[2]
            ]
            if self.go_to_pos(tgt):
                self.state = "MOVE_TO_BIN"
            return

        if s == "MOVE_TO_BIN":
            if self.go_to_pos(self.pre_bin_pos):
                self.state = "DROP_CAN"
            return

        if s == "DROP_CAN":
            # AT_DROP → magnet OFF
            self.set_magnet(False)
            time.sleep(0.3)
            # Notify other nodes that CAN drop is done
            self.pub_arm_done.publish(Bool(data=True))
            self.state = "FINISH"
            return

        if s == "FINISH":
            self.publish_zero_twist()
            self.get_logger().info("✅ CAN pick & place completed (HW, magnet)")
            return


# ========================= MAIN =========================
def main():
    rclpy.init()

    arm = Task2BArmHW()

    rclpy.spin(arm)

    arm.publish_zero_twist()
    arm.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
