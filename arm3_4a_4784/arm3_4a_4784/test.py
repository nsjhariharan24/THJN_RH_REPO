#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from control_msgs.msg import JointJog
from std_srvs.srv import SetBool
from std_msgs.msg import Float32

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import tf_transformations
import math
import time


TEAM_PREFIX = "4784"
DROP_FRAME = f"{TEAM_PREFIX}_drop_zone"   # define drop zone TF in URDF or via static tf
BASE_FRAME = "base_link"


class UR5TaskController(Node):
    def __init__(self):
        super().__init__("ur5_task_controller")

        # ---- Publishers ----
        self.twist_pub = self.create_publisher(TwistStamped, "/delta_twist_cmds", 10)
        self.joint_pub = self.create_publisher(JointJog, "/delta_joint_cmds", 10)

        # ---- Magnet client ----
        self.magnet_client = self.create_client(SetBool, "/magnet")
        while not self.magnet_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /magnet service...")

        # ---- Net force subscriber ----
        self.force_z = 0.0
        self.create_subscription(Float32, "/net_wrench", self.force_cb, 10)

        # ---- TF buffer/listener ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info("UR5 task controller started")

        # small timer to start high-level logic
        self.timer = self.create_timer(1.0, self.main_loop)
        self.step_done = False

    # ------------- Callbacks -------------

    def force_cb(self, msg):
        self.force_z = msg.data

    # ------------- High-level loop -------------

    def main_loop(self):
        if self.step_done:
            return

        # Example: pick first bad fruit then fertilizer
        try:
            # 1) pick bad fruit 1
            self.pick_and_place(f"{TEAM_PREFIX}_bad_fruit_1")

            # 2) pick fertilizer can
            self.pick_and_place(f"{TEAM_PREFIX}_fertilizer_1")

            self.step_done = True
            self.get_logger().info("Task finished")
        except Exception as e:
            self.get_logger().error(f"Error in main_loop: {e}")
            self.step_done = True

    # ------------- Core task: pick & place -------------

    def pick_and_place(self, target_frame):
        self.get_logger().info(f"Handling target: {target_frame}")

        # 1) Move above target (approach from top)
        target_t = self.lookup(target_frame)
        approach_z = target_t.transform.translation.z + 0.10  # 10 cm above
        self.move_cartesian_to_xy_z(
            x=target_t.transform.translation.x,
            y=target_t.transform.translation.y,
            z=approach_z,
        )

        # 2) Move down slowly until contact
        self.descend_until_contact()

        # 3) Magnet ON
        self.set_magnet(True)
        self.get_logger().info("Magnet ON")
        self.sleep(0.5)

        # 4) Lift up
        self.move_cartesian_delta(0.0, 0.0, 0.10)

        # 5) Move to drop zone
        drop_t = self.lookup(DROP_FRAME)
        self.move_cartesian_to_xy_z(
            x=drop_t.transform.translation.x,
            y=drop_t.transform.translation.y,
            z=drop_t.transform.translation.z + 0.10,
        )

        # 6) Move down a bit and release
        self.move_cartesian_delta(0.0, 0.0, -0.08)
        self.set_magnet(False)
        self.get_logger().info("Magnet OFF")
        self.sleep(0.5)

        # 7) Lift up again
        self.move_cartesian_delta(0.0, 0.0, 0.10)

    # ------------- Motion helpers -------------

    def move_cartesian_to_xy_z(self, x, y, z, speed=0.05, tol=0.01):
        """
        Simple P-type servo in XYZ using /delta_twist_cmds.
        """
        self.get_logger().info(f"Move to x={x:.3f}, y={y:.3f}, z={z:.3f}")
        for _ in range(300):  # ~6 s at 20 Hz
            now_t = self.lookup_tcp()
            cx = now_t.transform.translation.x
            cy = now_t.transform.translation.y
            cz = now_t.transform.translation.z

            ex, ey, ez = x - cx, y - cy, z - cz
            dist = math.sqrt(ex * ex + ey * ey + ez * ez)
            if dist < tol:
                self.stop_cartesian()
                return

            vx = self.clamp(0.8 * ex, -speed, speed)
            vy = self.clamp(0.8 * ey, -speed, speed)
            vz = self.clamp(0.8 * ez, -speed, speed)

            self.publish_twist(vx, vy, vz, 0.0, 0.0, 0.0)
            self.sleep(0.05)

        self.stop_cartesian()

    def move_cartesian_delta(self, dx, dy, dz, speed=0.05):
        now_t = self.lookup_tcp()
        x = now_t.transform.translation.x + dx
        y = now_t.transform.translation.y + dy
        z = now_t.transform.translation.z + dz
        self.move_cartesian_to_xy_z(x, y, z, speed=speed)

    def descend_until_contact(self, speed=0.02, force_thresh=5.0):
        self.get_logger().info("Descending until contact...")
        for _ in range(400):  # timeout
            if abs(self.force_z) > force_thresh:
                self.get_logger().info(f"Contact detected: Fz={self.force_z:.2f}")
                self.stop_cartesian()
                return
            self.publish_twist(0.0, 0.0, -speed, 0.0, 0.0, 0.0)
            self.sleep(0.05)
        self.stop_cartesian()

    def publish_twist(self, lx, ly, lz, ax, ay, az):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = lx
        msg.twist.linear.y = ly
        msg.twist.linear.z = lz
        msg.twist.angular.x = ax
        msg.twist.angular.y = ay
        msg.twist.angular.z = az
        self.twist_pub.publish(msg)

    def stop_cartesian(self):
        self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # ------------- TF helpers -------------

    def lookup(self, child_frame):
        return self.tf_buffer.lookup_transform(
            BASE_FRAME, child_frame, rclpy.time.Time()
        )

    def lookup_tcp(self):
        # use /tcp_pose_raw instead if you prefer
        return self.tf_buffer.lookup_transform(
            BASE_FRAME, "tool0", rclpy.time.Time()
        )

    # ------------- Magnet helper -------------

    def set_magnet(self, state: bool):
        req = SetBool.Request()
        req.data = state
        future = self.magnet_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    # ------------- Utility -------------

    def clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def sleep(self, sec):
        # non-blocking-like sleep in ROS2 node
        end = self.get_clock().now().nanoseconds + int(sec * 1e9)
        while self.get_clock().now().nanoseconds < end:
            rclpy.spin_once(self, timeout_sec=0.01)


def main():
    rclpy.init()
    node = UR5TaskController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
