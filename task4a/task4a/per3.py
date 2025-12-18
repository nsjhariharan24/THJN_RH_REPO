#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge

from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import tf_transformations

import cv2
import numpy as np
import math

# ---------------- CONFIG ----------------
TEAM_PREFIX = "4784"
CAN_MARKER_ID = 3          # Fertilizer marker (KNOWN)
DROP_ZONE_MAX_Z = 0.25     # meters (drop zone is LOW)
MAX_BAD_FRUITS = 3
SHOW_GUI = True


class CombinedPerception(Node):

    def __init__(self):
        super().__init__("combined_perception")

        self.bridge = CvBridge()
        self.color = None
        self.depth = None
        self.K = None

        # Fixed fertilizer orientation
        q = np.array([0.707, 0.028, 0.034, 0.707])
        self.q_fertilizer = (q / np.linalg.norm(q)).tolist()

        # Subscribers
        self.create_subscription(Image,
                                 "/camera/camera/color/image_raw",
                                 self.color_cb, 10)
        self.create_subscription(Image,
                                 "/camera/camera/aligned_depth_to_color/image_raw",
                                 self.depth_cb, 10)
        self.create_subscription(CameraInfo,
                                 "/camera/camera/color/camera_info",
                                 self.caminfo_cb, 10)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_pub = TransformBroadcaster(self)

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.get_logger().info("✅ Combined perception started")

    # ---------------- Callbacks ----------------
    def caminfo_cb(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)

    def color_cb(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process()

    def depth_cb(self, msg):
        try:
            self.depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception:
            self.depth = self.bridge.imgmsg_to_cv2(msg)

    # ---------------- Main ----------------
    def process(self):
        if self.color is None or self.depth is None or self.K is None:
            return

        frame = self.color.copy()

        self.detect_arucos(frame)
        self.detect_bad_fruits(frame)

        if SHOW_GUI:
            cv2.imshow("Perception", frame)
            cv2.waitKey(1)

    # ---------------- ArUco Detection ----------------
    def detect_arucos(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        if ids is None:
            return

        for i, mid in enumerate(ids.flatten()):
            pts = corners[i].reshape(4, 2)
            u = int(np.mean(pts[:, 0]))
            v = int(np.mean(pts[:, 1]))

            d = self.get_depth(self.depth, u, v)
            if d is None:
                continue

            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]

            X = d
            Y = -(u - cx) * d / fx
            Z = -(v - cy) * d / fy
            P_cam = np.array([X, Y, Z])

            try:
                t = self.tf_buffer.lookup_transform(
                    "base_link", "camera_link", rclpy.time.Time()
                )
                R = tf_transformations.quaternion_matrix([
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w
                ])[:3, :3]
                T = np.array([
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z
                ])
                P_base = R @ P_cam + T
            except Exception:
                continue

            tfm = TransformStamped()
            tfm.header.stamp = self.get_clock().now().to_msg()
            tfm.header.frame_id = "base_link"

            # ---- DIFFERENTIATION LOGIC ----
            if mid == CAN_MARKER_ID:
                tfm.child_frame_id = f"{TEAM_PREFIX}_fertilizer_1"
                tfm.transform.rotation.x = self.q_fertilizer[0]
                tfm.transform.rotation.y = self.q_fertilizer[1]
                tfm.transform.rotation.z = self.q_fertilizer[2]
                tfm.transform.rotation.w = self.q_fertilizer[3]

            else:
                # Drop zone is LOW in Z
                if P_base[2] < DROP_ZONE_MAX_Z:
                    tfm.child_frame_id = f"{TEAM_PREFIX}_drop_zone"
                    tfm.transform.rotation.w = 1.0
                else:
                    continue

            tfm.transform.translation.x = float(P_base[0])
            tfm.transform.translation.y = float(P_base[1])
            tfm.transform.translation.z = float(P_base[2])

            self.tf_pub.sendTransform(tfm)

    # ---------------- Bad Fruit Detection ----------------
    def detect_bad_fruits(self, frame):
        h, _, _ = frame.shape
        img = frame[h // 2 :, :]
        depth = self.depth[h // 2 :, :]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, np.array([35, 40, 40]), np.array([95, 255, 255])
        )

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

        fid = 0
        for c in cnts:
            if cv2.contourArea(c) < 300:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            d = self.get_depth(depth, cx, cy)
            if d is None:
                continue

            fx, fy = self.K[0, 0], self.K[1, 1]
            cx0, cy0 = self.K[0, 2], self.K[1, 2]

            X = d
            Y = -(cx - cx0) * d / fx
            Z = -((cy + h // 2) - cy0) * d / fy
            P_cam = np.array([X, Y, Z])

            try:
                t = self.tf_buffer.lookup_transform(
                    "base_link", "camera_link", rclpy.time.Time()
                )
                R = tf_transformations.quaternion_matrix([
                    t.transform.rotation.x,
                    t.transform.rotation.y,
                    t.transform.rotation.z,
                    t.transform.rotation.w
                ])[:3, :3]
                T = np.array([
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z
                ])
                P_base = R @ P_cam + T
            except Exception:
                continue

            fid += 1
            if fid > MAX_BAD_FRUITS:
                break

            tfm = TransformStamped()
            tfm.header.stamp = self.get_clock().now().to_msg()
            tfm.header.frame_id = "base_link"
            tfm.child_frame_id = f"{TEAM_PREFIX}_bad_fruit_{fid}"

            tfm.transform.translation.x = float(P_base[0])
            tfm.transform.translation.y = float(P_base[1])
            tfm.transform.translation.z = float(P_base[2])
            tfm.transform.rotation.w = 1.0

            self.tf_pub.sendTransform(tfm)

    def get_depth(self, depth_img, u, v):
        h, w = depth_img.shape
        if u < 0 or v < 0 or u >= w or v >= h:
            return None
        patch = depth_img[max(0, v-2):min(h, v+2),
                          max(0, u-2):min(w, u+2)]
        valid = np.isfinite(patch) & (patch > 0)
        if valid.sum() == 0:
            return None
        return float(np.median(patch[valid]))


def main():
    rclpy.init()
    node = CombinedPerception()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
