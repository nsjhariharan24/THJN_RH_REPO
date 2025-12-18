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
CAN_MARKER_ID = 3
MAX_BAD_FRUITS = 3
SHOW_GUI = True   # disable in remote access


class CombinedPerception(Node):

    def __init__(self):
        super().__init__("combined_fruit_fertilizer_perception")

        self.bridge = CvBridge()
        self.color = None
        self.depth = None
        self.K = None

        # ---- Fertilizer fixed orientation (UNCHANGED) ----
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

        # ArUco setup (UNCHANGED)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.get_logger().info("Combined perception node started")

    # ---------------- Callbacks ----------------

    def caminfo_cb(self, msg):
        self.K = np.array(msg.k).reshape(3, 3)

    def color_cb(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process()

    def depth_cb(self, msg):
        try:
            self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception:
            self.depth = self.bridge.imgmsg_to_cv2(msg)

    # ---------------- Main Processing ----------------

    def process(self):
        if self.color is None or self.depth is None or self.K is None:
            return

        frame = self.color.copy()

        # 1️⃣ Fertilizer + extra ArUco detection
        self.detect_fertilizer_and_other_arucos(frame)

        # 2️⃣ Bad fruit detection (UNCHANGED)
        self.detect_bad_fruits(frame)

        if SHOW_GUI:
            cv2.imshow("Combined Perception", frame)
            cv2.waitKey(1)

    # ---------------- Fertilizer + OTHER ArUco Detection ----------------

    def detect_fertilizer_and_other_arucos(self, frame):
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

            # -------- Fertilizer (UNCHANGED) --------
            if mid == CAN_MARKER_ID:
                tfm.child_frame_id = f"{TEAM_PREFIX}_fertilizer_1"
                tfm.transform.rotation.x = self.q_fertilizer[0]
                tfm.transform.rotation.y = self.q_fertilizer[1]
                tfm.transform.rotation.z = self.q_fertilizer[2]
                tfm.transform.rotation.w = self.q_fertilizer[3]

            # -------- ONLY ADDITION: OTHER ARUCO --------
            else:
                tfm.child_frame_id = f"{TEAM_PREFIX}_aruco_{mid}"
                tfm.transform.rotation.w = 1.0
            # -----------------------------------------

            tfm.transform.translation.x = float(P_base[0])
            tfm.transform.translation.y = float(P_base[1])
            tfm.transform.translation.z = float(P_base[2])

            self.tf_pub.sendTransform(tfm)

            if SHOW_GUI:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.circle(frame, (u, v), 6, (255, 0, 0), -1)

    # ---------------- Bad Fruit Detection (UNCHANGED) ----------------

    def detect_bad_fruits(self, frame):
        h, w, _ = frame.shape

        img = frame[h // 2 :, :]
        depth = self.depth[h // 2 :, :]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        green_lower = np.array([35, 40, 40])
        green_upper = np.array([95, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_mask = cv2.medianBlur(green_mask, 5)

        cnts, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        fid = 0

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 300:
                continue

            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue

            circularity = 4 * math.pi * area / (peri * peri)
            if circularity < 0.6:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            x, y, bw, bh = cv2.boundingRect(c)

            body = hsv[
                min(y + bh, img.shape[0]-1):min(y + bh + 40, img.shape[0]-1),
                max(x - 20, 0):min(x + bw + 20, img.shape[1]-1)
            ]

            if body.size == 0:
                continue

            sat = body[:, :, 1]
            grey_ratio = np.mean(sat < 60)
            if grey_ratio < 0.10:
                continue

            hue = body[:, :, 0]
            sat_body = body[:, :, 1]
            purple_ratio = np.mean((hue > 125) & (hue < 165) & (sat_body > 50))
            if purple_ratio > 0.10:
                continue

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

            if SHOW_GUI:
                cv2.circle(frame, (cx, cy + h // 2), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"BAD_{fid}",
                            (cx - 25, cy + h // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

    # ---------------- Depth Helper ----------------

    def get_depth(self, depth_img, u, v):
        h, w = depth_img.shape
        if u < 0 or v < 0 or u >= w or v >= h:
            return None

        patch = depth_img[max(0, v-3):min(h, v+3),
                          max(0, u-3):min(w, u+3)]
        valid = np.isfinite(patch) & (patch > 0)
        if valid.sum() == 0:
            return None

        d = np.median(patch[valid])
        if d > 10.0:
            d /= 1000.0
        return float(d)


def main():
    rclpy.init()
    node = CombinedPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if SHOW_GUI:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()