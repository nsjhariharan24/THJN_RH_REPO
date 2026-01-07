#!/usr/bin/env python3
'''
# Team ID:          4784
# Theme:            Krishi coBot
# Filename:         task4a_can_only_hardware.py
# Description:      CAN-only hardware controller using /delta_twist_cmds,
#                   /tcp_pose_raw, /magnet and TF from perception.
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
BASE_FRAME = "base_link"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Task4AArmCanOnly(Node):

    def __init__(self):
        super().__init__("task4a_arm_can_only")

        # ---------- Publishers ----------
        self.pub_twist = self.create_publisher(
            TwistStamped, "/delta_twist_cmds", 10
        )
        self.pub_arm_done = self.create_publisher(Bool, "/arm_done_drop", 10)

        # ---------- Subscribers ----------
        self.sub_tcp = self.create_subscription(
            Float64MultiArray, "/tcp_pose_raw", self.cb_tcp_pose, 10
        )

        # ---------- TF (for fertilizer frame only) ----------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------- Magnet service ----------
        self.magnet_client = self.create_client(SetBool, "/magnet")
        while not self.magnet_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /magnet service...")

        # ---------- Fixed Positions ----------
        # You may need to retune these for hardware
        self.initial_pos = [0.406, 0.010, 0.582]
        self.P1_POS = [-0.214, -0.332, 0.557]
        self.pre_bin_pos = [-0.806, 0.010, 0.402]

        # ---------- Control ----------
        self.lin_gain = 0.8
        self.max_lin = 0.05   # safer for hardware
        self.pos_tol = 0.02

        # ---------- State ----------
        self.can_pos = None
        self.state = "WAIT_TF"

        self.tcp_pose = None  # [x, y, z, rx, ry, rz]

        self.timer = self.create_timer(0.1, self.loop)

        self.get_logger().info("🟡 CAN-only FSM started (hardware, tcp_pose_raw)")


    # ---------------- TCP Pose callback ----------------
    def cb_tcp_pose(self, msg: Float64MultiArray):
        # [x, y, z, rx, ry, rz]
        self.tcp_pose = msg.data


    # ---------------- Pose helpers ----------------
    def get_tool_pos(self):
        """Return current TCP position [x, y, z] from /tcp_pose_raw."""
        if self.tcp_pose is None or len(self.tcp_pose) < 3:
            return None
        x, y, z = self.tcp_pose[0], self.tcp_pose[1], self.tcp_pose[2]
        return [x, y, z]

    def get_tf_pos(self, frame):
        """Return [x, y, z] of a TF frame in base_link."""
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, frame, rclpy.time.Time()
            )
            t = tf.transform.translation
            return [t.x, t.y, t.z]
        except Exception:
            return None


    # ---------------- Motion ----------------
    def publish_twist(self, lx, ly, lz, ax=0.0, ay=0.0, az=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
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
        Move TCP toward target [x, y, z] using P-control on /delta_twist_cmds.
        Returns True when within pos_tol.
        """
        pos = self.get_tool_pos()
        if pos is None:
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


    # ---------------- Magnet ----------------
    def set_magnet(self, state: bool):
        req = SetBool.Request()
        req.data = state
        future = self.magnet_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.result() is not None:
            if not future.result().success:
                self.get_logger().warn(f"Magnet call failed: {future.result().message}")
        else:
            self.get_logger().warn("No response from /magnet service")


    # ---------------- FSM ----------------
    def loop(self):

        s = self.state
        # self.get_logger().debug(f"State: {s}")

        # ----- WAIT FOR CAN TF -----
        if s == "WAIT_TF":
            self.can_pos = self.get_tf_pos(f"{TEAM_ID}_fertilizer_1")
            if self.can_pos:
                self.get_logger().info("Fertilizer TF found, starting CAN sequence")
                self.state = "MOVE_INIT"
            return

        if s == "MOVE_INIT":
            if self.go_to_pos(self.initial_pos):
                self.state = "MOVE_P1"
            return

        if s == "MOVE_P1":
            # In hardware version we are not rotating by P1_QUAT (no orientation control here),
            # just using this point as an intermediate safe waypoint.
            if self.go_to_pos(self.P1_POS):
                self.state = "MOVE_CAN"
            return

        if s == "MOVE_CAN":
            # Approach from above for safety
            target_above = [self.can_pos[0], self.can_pos[1], self.can_pos[2] + 0.10]
            if self.go_to_pos(target_above):
                # Move down a bit to make contact and grip (no force check here,
                # you can add /net_wrench logic if needed)
                down_target = [self.can_pos[0], self.can_pos[1], self.can_pos[2]]
                if self.go_to_pos(down_target):
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
            # Small descent before release if you want:
            # down_tgt = [self.pre_bin_pos[0], self.pre_bin_pos[1], self.pre_bin_pos[2] - 0.05]
            # self.go_to_pos(down_tgt)
            self.set_magnet(False)
            self.pub_arm_done.publish(Bool(data=True))
            self.state = "FINISH"
            return

        if s == "FINISH":
            self.stop_cartesian()
            self.get_logger().info("✅ CAN pick & place completed (hardware)")
            return


def main():
    rclpy.init()

    arm = Task4AArmCanOnly()

    rclpy.spin(arm)

    arm.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
