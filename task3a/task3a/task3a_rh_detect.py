#!/usr/bin/env python3
"""
Team ID:          4784
Theme:            Krishi coBot
Task:             3A - Object Detection & TF Publishing on Hardware
Authors:          Hariharan S., Thilak Raj S., Jeevan Uday Alexander, Niranjan R.
Filename:         task3a_rh_detect.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import numpy as np
import cv2
import math
import tf_transformations


# ---------------- ROTATION UTILITIES (if needed later) ---------------- #
def rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa, ca]])

def rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, 0, sa],
                     [0, 1, 0],
                     [-sa, 0, ca]])

def rot_z(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0],
                     [sa,  ca, 0],
                     [0,   0, 1]])


# ---------------- MAIN NODE ---------------- #
class FruitAndCanDetectorHW(Node):
    """
    Node to:
    - Subscribe to hardware camera topics
    - Detect bad fruits (by color) in ROI
    - Detect fertilizer can (ArUco ID 3)
    - Compute 3D positions using depth
    - Transform to base_link using TF
    - Publish TF frames:
        4784_bad_fruit_1, 4784_bad_fruit_2, 4784_fertilizer_1
    """

    def __init__(self):
        super().__init__("team_4784_hardware_detector")

        # CV bridge
        self.bridge = CvBridge()

        # Camera data placeholders
        self.color = None
        self.depth = None
        self.camera_matrix = None
        self.caminfo_ready = False

        # Camera frame used in TF lookup (change if needed)
        # For RealSense, often "camera_color_optical_frame". If TF errors, try "camera_link".
        self.camera_frame = "camera_color_optical_frame"

        # ------------ Subscribers: HARDWARE TOPICS ------------ #
        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",      # RGB from hardware
            self.color_cb,
            10
        )

        self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",  # aligned depth
            self.depth_cb,
            10
        )

        self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",   # intrinsics
            self.caminfo_cb,
            10
        )

        # ------------ TF Broadcaster + Buffer ------------ #
        self.tf_pub = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ------------ Fruit detection ROI ------------ #
        # Same ROI as your previous working sim code
        self.box_x1, self.box_y1 = 5, 210
        self.box_x2, self.box_y2 = 380, 400

        # ------------ ArUco setup ------------ #
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Fertilizer can fixed orientation (you can tune if needed)
        q_raw = np.array([0.707, 0.028, 0.034, 0.707])
        self.can_quat = (q_raw / np.linalg.norm(q_raw)).tolist()

        self.get_logger().info("Team 4784 | Hardware Detector Node ACTIVE")


    # ---------------- CAMERA INFO CALLBACK ---------------- #
    def caminfo_cb(self, msg):
        """
        Receives camera intrinsics (K matrix) from hardware.
        """
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.caminfo_ready = True


    # ---------------- COLOR CALLBACK ---------------- #
    def color_cb(self, msg):
        """
        Receives color image and triggers processing.
        """
        try:
            self.color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"Color image conversion failed: {e}")
            self.color = None
            return

        self.process()


    # ---------------- DEPTH CALLBACK ---------------- #
    def depth_cb(self, msg):
        """
        Receives depth image.
        """
        try:
            self.depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            self.get_logger().warn(f"Depth image conversion failed: {e}")
            self.depth = None


    # ---------------- DEPTH HELPER ---------------- #
    def median_patch(self, u, v, half=4):
        """
        Take a small square patch around pixel (u, v) in depth image,
        return median finite depth.
        """
        if self.depth is None:
            return None

        h, w = self.depth.shape[:2]
        u0, u1 = max(0, u-half), min(w-1, u+half)
        v0, v1 = max(0, v-half), min(h-1, v+half)

        patch = self.depth[v0:v1+1, u0:u1+1]
        valid = np.isfinite(patch) & (patch > 0)

        if np.count_nonzero(valid) == 0:
            return None

        return float(np.median(patch[valid]))


    # ---------------- MAIN PIPELINE ---------------- #
    def process(self):
        """
        Runs every time a new color frame arrives
        (but only if depth + camera info are ready).
        """
        if self.color is None or self.depth is None or not self.caminfo_ready:
            return

        self.detect_bad_fruits()
        self.detect_fertilizer_can()


    # ---------------- BAD FRUIT DETECTION ---------------- #
    def detect_bad_fruits(self):
        """
        Detect bad fruits in ROI using color threshold in HSV,
        compute 3D positions using depth, transform to base_link
        using TF, and publish TF frames:
            4784_bad_fruit_1
            4784_bad_fruit_2
        """

        # 1. Crop ROI from full image
        roi = self.color[self.box_y1:self.box_y2, self.box_x1:self.box_x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 2. Threshold for bad fruit color range (same as your older working code)
        lower = np.array([0, 0, 70])
        upper = np.array([180, 40, 200])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 5)

        # 3. Find blobs (contours)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        fruit_id = 1

        for c in contours:
            if fruit_id > 2:    # limit to 2 bad fruits as per rules
                break

            if cv2.contourArea(c) < 300:
                continue

            # bounding box in ROI
            x, y, w, h = cv2.boundingRect(c)
            # shift to full image coordinates
            x += self.box_x1
            y += self.box_y1

            # center pixel
            u = x + w // 2
            v = y + h // 2

            depth = self.median_patch(u, v)
            if depth is None:
                continue

            # Back-project to 3D in camera frame
            X = depth
            Y = -(u - cx) * depth / fx
            Z = -(v - cy) * depth / fy
            P_cam = np.array([X, Y, Z])

            # Lookup TF from camera frame to base_link
            try:
                tf = self.tf_buffer.lookup_transform(
                    "base_link",
                    self.camera_frame,
                    rclpy.time.Time()
                )
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed for fruits: {e}")
                return

            T = tf.transform.translation
            Q = tf.transform.rotation
            R = tf_transformations.quaternion_matrix([Q.x, Q.y, Q.z, Q.w])[:3, :3]

            # Transform point to base_link
            P_base = R @ P_cam + np.array([T.x, T.y, T.z])

            # Orientation for fruits – identity (can be changed if needed)
            q = tf_transformations.quaternion_from_euler(0.0, 0.0, 0.0)

            # Publish TF
            tfmsg = TransformStamped()
            tfmsg.header.stamp = self.get_clock().now().to_msg()
            tfmsg.header.frame_id = "base_link"
            tfmsg.child_frame_id = f"4784_bad_fruit_{fruit_id}"
            tfmsg.transform.translation.x = float(P_base[0])
            tfmsg.transform.translation.y = float(P_base[1])
            tfmsg.transform.translation.z = float(P_base[2])
            tfmsg.transform.rotation.x = q[0]
            tfmsg.transform.rotation.y = q[1]
            tfmsg.transform.rotation.z = q[2]
            tfmsg.transform.rotation.w = q[3]

            self.tf_pub.sendTransform(tfmsg)
            fruit_id += 1


    # ---------------- FERTILIZER CAN DETECTION (ARUCO ID 3) ---------------- #
    def detect_fertilizer_can(self):
        """
        Detect ArUco marker ID 3 on the fertilizer can,
        compute 3D position using depth, transform to base_link
        using TF, and publish TF frame:
            4784_fertilizer_1
        """

        gray = cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.aruco_params
        )

        if ids is None:
            return

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        for i, corner in enumerate(corners):
            marker_id = int(ids[i][0])
            if marker_id != 3:
                continue

            pts = corner.reshape((4, 2)).astype(int)
            u = int(np.mean(pts[:, 0]))
            v = int(np.mean(pts[:, 1]))

            depth = self.median_patch(u, v)
            if depth is None:
                continue

            # Back-project to 3D in camera frame
            X = depth
            Y = -(u - cx) * depth / fx
            Z = -(v - cy) * depth / fy
            P_cam = np.array([X, Y, Z])

            # Lookup TF from camera frame to base_link
            try:
                tf = self.tf_buffer.lookup_transform(
                    "base_link",
                    self.camera_frame,
                    rclpy.time.Time()
                )
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed for fertilizer: {e}")
                return

            T = tf.transform.translation
            Q = tf.transform.rotation
            R = tf_transformations.quaternion_matrix([Q.x, Q.y, Q.z, Q.w])[:3, :3]

            # Transform to base_link
            P_base = R @ P_cam + np.array([T.x, T.y, T.z])

            # Use predefined can orientation (tune if required)
            msg = TransformStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.child_frame_id = "4784_fertilizer_1"
            msg.transform.translation.x = float(P_base[0])
            msg.transform.translation.y = float(P_base[1])
            msg.transform.translation.z = float(P_base[2])
            msg.transform.rotation.x = self.can_quat[0]
            msg.transform.rotation.y = self.can_quat[1]
            msg.transform.rotation.z = self.can_quat[2]
            msg.transform.rotation.w = self.can_quat[3]

            self.tf_pub.sendTransform(msg)
            # Only one fertilizer can
            break


# ---------------- MAIN ---------------- #
def main():
    rclpy.init()
    node = FruitAndCanDetectorHW()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()