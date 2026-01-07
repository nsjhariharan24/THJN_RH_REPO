#!/usr/bin/env python3
# Team ID: 4784 – Krishi coBot – Task 4A hardware controller (using /tcp_pose_raw)

import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool, Float32, Float64MultiArray
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener
import tf_transformations

TEAM_ID = 4784
BASE_FRAME = "base_link"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Task4AArm(Node):
    def __init__(self):
        super().__init__("task4a_arm_controller")

        # ---------- Publishers ----------
        self.pub_twist = self.create_publisher(
            TwistStamped, "/delta_twist_cmds", 10
        )
        self.pub_arm_done = self.create_publisher(Bool, "/arm_done_drop", 10)

        # ---------- Subscribers ----------
        self.sub_docked = self.create_subscription(
            Bool, "/ebot_docked", self.cb_docked, 10
        )
        self.sub_force = self.create_subscription(
            Float32, "/net_wrench", self.cb_force, 10
        )
        self.sub_tcp = self.create_subscription(
            Float64MultiArray, "/tcp_pose_raw", self.cb_tcp_pose, 10
        )

        # ---------- TF (for object frames only) ----------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------- Magnet service ----------
        self.magnet_client = self.create_client(SetBool, "/magnet")
        while not self.magnet_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /magnet service...")

        # ---------- Positions / waypoints ----------
        self.initial_pos = [0.406, 0.010, 0.582]
        self.waypoint = [0.0, 0.5, 0.5]
        self.pre_bin_pos = [-0.806, 0.010, 0.402]
        self.P1_POS = [-0.214, -0.332, 0.557]
        self.P1_QUAT = [0.707, 0.028, 0.034, 0.707]

        # ---------- Control gains ----------
        self.lin_gain = 0.8
        self.max_lin = 0.05
        self.pos_tol = 0.02

        self.ang_gain = 0.6
        self.max_ang = 0.5
        self.ori_tol = 0.15

        # Pitch orientations (used if you later do orientation via TF)
        self.q_pitch_fruit = tf_transformations.quaternion_from_euler(
            0, math.pi, 0
        )
        self.q_pitch_can = tf_transformations.quaternion_from_euler(
            0, math.pi, 0
        )

        # ---------- State ----------
        self.fruit_pos = {}
        self.can_pos = None
        self.marker6_pos = None
        self.current_fruit = 1
        self.ebot_docked = False
        self.can_done = False

        self.wait_start_time = None
        self.force_z = 0.0

        self.tcp_pose = None  # [x, y, z, rx, ry, rz]

        self.state = "WAIT_TF"
        self.timer = self.create_timer(0.1, self.loop)

        self.get_logger().info("Task 4A arm controller started (hardware, tcp_pose_raw)")

    # --------- Callbacks ---------
    def cb_docked(self, msg: Bool):
        self.ebot_docked = msg.data

    def cb_force(self, msg: Float32):
        self.force_z = msg.data

    def cb_tcp_pose(self, msg: Float64MultiArray):
        self.tcp_pose = msg.data

    # --------- Pose helpers using /tcp_pose_raw ---------
    def get_tool_pose(self):
        """
        Returns ([x, y, z], None) using /tcp_pose_raw.
        Orientation from rx, ry, rz is ignored here.
        """
        if self.tcp_pose is None or len(self.tcp_pose) < 3:
            return None, None
        x, y, z = self.tcp_pose[0], self.tcp_pose[1], self.tcp_pose[2]
        return [x, y, z], None

    # --------- Object TF helpers ---------
    def get_tf_pos(self, frame):
        """Return translation [x,y,z] of a TF frame in base_link."""
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, frame, rclpy.time.Time()
            )
            t = tf.transform.translation
            return [t.x, t.y, t.z]
        except Exception:
            return None

    # --------- Motion helpers (Cartesian velocity) ---------
    def publish_twist(self, lx, ly, lz, ax, ay, az):
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
        """Move end-effector toward target [x,y,z] using P-control on /delta_twist_cmds."""
        pos, _ = self.get_tool_pose()
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

        self.publish_twist(vx, vy, vz, 0.0, 0.0, 0.0)
        return False

    def descend_until_contact(self, step_v=-0.01, force_thresh=4.0, timeout=5.0):
        """Slowly move down until net force exceeds threshold."""
        start = time.time()
        while time.time() - start < timeout:
            if abs(self.force_z) > force_thresh:
                self.get_logger().info(f"Contact detected: Fz={self.force_z:.2f}")
                self.stop_cartesian()
                return True
            self.publish_twist(0.0, 0.0, step_v, 0.0, 0.0, 0.0)
            time.sleep(0.05)
        self.stop_cartesian()
        self.get_logger().warn("Timeout in descend_until_contact")
        return False

    # --------- Magnet control ---------
    def set_magnet(self, state: bool):
        req = SetBool.Request()
        req.data = state
        future = self.magnet_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.result() is not None:
            if not future.result().success:
                self.get_logger().warn(f"Magnet call failed: {future.result().message}")
        else:
            self.get_logger().warn("No response from magnet service")

    # --------- FSM loop ---------
    def loop(self):
        s = self.state
        now = time.time()

        # Debug: print state occasionally
        self.get_logger().debug(f"State: {s}")

        # --------- Wait for perception TFs ---------
        if s == "WAIT_TF":
            for i in range(1, 4):
                self.fruit_pos[i] = self.get_tf_pos(f"{TEAM_ID}_bad_fruit_{i}")
            self.can_pos = self.get_tf_pos(f"{TEAM_ID}_fertilizer_1")

            if all(self.fruit_pos.values()) and self.can_pos:
                self.get_logger().info("All TF targets found, starting sequence")
                self.state = "MOVE_INIT"
            return

        # Move to initial safe pose
        if s == "MOVE_INIT":
            if self.go_to_pos(self.initial_pos):
                self.state = "MOVE_WP"
            return

        # Move to waypoint
        if s == "MOVE_WP":
            if self.go_to_pos(self.waypoint):
                self.wait_start_time = now
                self.state = "CHECK_DOCK_1S"
            return

        # Wait for docking or timeout → go to fruits
        if s == "CHECK_DOCK_1S":
            if self.ebot_docked and not self.can_done:
                self.marker6_pos = self.get_tf_pos("aruco_marker_6")
                self.state = "MOVE_INIT_BEFORE_CAN"
                return
            if now - self.wait_start_time >= 1.0:
                self.state = "MOVE_FRUIT"
            return

        # --------- Fruit sequence ---------
        if s == "MOVE_FRUIT":
            p = self.fruit_pos[self.current_fruit]
            if p is None:
                self.get_logger().warn("Fruit TF missing, staying in MOVE_FRUIT")
                return
            # Approach above fruit
            target_above = [p[0], p[1], p[2] + 0.10]
            if self.go_to_pos(target_above):
                # descend until contact
                self.descend_until_contact()
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
                # descend a bit and release
                self.descend_until_contact(step_v=-0.01, force_thresh=2.0, timeout=2.0)
                self.set_magnet(False)
                time.sleep(0.3)
                # lift back up
                p = self.pre_bin_pos
                self.go_to_pos([p[0], p[1], p[2] + 0.10])

                if self.current_fruit < 3:
                    self.current_fruit += 1
                    self.state = "MOVE_WP"
                else:
                    self.state = (
                        "WAIT_DOCK_AFTER_FRUITS" if not self.can_done else "FINISH"
                    )
            return

        if s == "WAIT_DOCK_AFTER_FRUITS":
            if self.go_to_pos(self.waypoint) and self.ebot_docked:
                self.marker6_pos = self.get_tf_pos("aruco_marker_6")
                self.state = "MOVE_INIT_BEFORE_CAN"
            return

        # --------- Can sequence ---------
        if s == "MOVE_INIT_BEFORE_CAN":
            if self.go_to_pos(self.initial_pos):
                self.state = "MOVE_P1"
            return

        if s == "MOVE_P1":
            if self.go_to_pos(self.P1_POS):
                self.state = "MOVE_CAN"
            return

        if s == "MOVE_CAN":
            p = self.can_pos
            if p is None:
                self.get_logger().warn("Can TF missing, staying in MOVE_CAN")
                return
            target_above = [p[0], p[1], p[2] + 0.10]
            if self.go_to_pos(target_above):
                self.descend_until_contact()
                self.set_magnet(True)
                time.sleep(0.3)
                self.marker6_pos = self.get_tf_pos("aruco_marker_6")
                self.state = "SHIFT_CAN_Y"
            return

        if s == "SHIFT_CAN_Y":
            if self.go_to_pos(
                [self.can_pos[0], self.can_pos[1] + 0.3, self.can_pos[2]]
            ):
                self.state = "SHIFT_CAN_X"
            return

        if s == "SHIFT_CAN_X":
            if self.go_to_pos(
                [self.can_pos[0] + 0.5, self.can_pos[1] + 0.3, self.can_pos[2]]
            ):
                self.state = "MOVE_INIT_AFTER_CAN"
            return

        if s == "MOVE_INIT_AFTER_CAN":
            if self.go_to_pos(self.initial_pos):
                self.state = "DROP_CAN"
            return

        if s == "DROP_CAN":
            if self.marker6_pos is None:
                self.get_logger().warn("marker6_pos missing in DROP_CAN")
                return
            tgt = [
                self.marker6_pos[0],
                self.marker6_pos[1],
                self.marker6_pos[2] + 0.2,
            ]
            if self.go_to_pos(tgt):
                # descend and drop can
                self.descend_until_contact(step_v=-0.01, force_thresh=2.0, timeout=2.0)
                self.set_magnet(False)
                self.pub_arm_done.publish(Bool(data=True))
                self.can_done = True
                self.state = "MOVE_INIT"
            return

        if s == "FINISH":
            self.stop_cartesian()
            self.get_logger().info("TASK 4A FINISHED")
            return


def main():
    rclpy.init()
    node = Task4AArm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
