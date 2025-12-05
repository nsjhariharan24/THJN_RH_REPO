#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, PointStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import numpy as np
import cv2
import math
import tf_transformations

"""
Merged node (adjusted to ensure fertilizer TF is published wrt base_link
by using the camera frame from CameraInfo and a robust TF lookup).

⚙ Hardware topics:
- Color  : /camera/camera/color/image_raw
- Depth  : /camera/camera/aligned_depth_to_color/image_raw
- Info   : /camera/camera/color/camera_info
"""

CAN_MARKER_ID = 3  # marker id for the fertiliser can alias


class ArucoCenterFixedOri(Node):
    """Publish base_link -> aruco_marker_X with fixed orientation.
       ID 3 → fixed quaternion (0.707, 0.028, 0.034, 0.707)
       ID 6 → orientation = Euler (0, π, 0)
       Also: detect "bad fruits" and publish team_4784_bad_fruit_N and /bad_fruit_points.
    """

    def __init__(self):
        super().__init__("aruco_center_fixed_ori")
        self.bridge = CvBridge()
        self.color = None
        self.depth = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = None   # <- will be set from CameraInfo

        # ---------- Subscribers (UPDATED TOPICS FOR HARDWARE) ----------
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.color_cb, 10)
        self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw", self.depth_cb, 10)
        self.create_subscription(CameraInfo, "/camera/camera/color/camera_info", self.caminfo_cb, 10)

        # TF System
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_pub = TransformBroadcaster(self)

        # Bad-fruit point publisher
        self.point_pub = self.create_publisher(PointStamped, '/bad_fruit_points', 10)

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # detector parameters: support both APIs
        try:
            self.params = cv2.aruco.DetectorParameters()
        except Exception:
            self.params = cv2.aruco.DetectorParameters_create()

        # Marker 3 → fixed quaternion
        q3_raw = np.array([0.707, 0.028, 0.034, 0.707], dtype=float)
        self.q_marker3 = (q3_raw / np.linalg.norm(q3_raw)).tolist()

        # Marker 6 → Euler (0, π, 0)
        self.q_marker6 = tf_transformations.quaternion_from_euler(0, math.pi, 0)

        # --- Bad fruit detection params (from reference) ---
        # ROI for fruit detection (pixels)
        self.box_x1, self.box_y1 = 5, 210
        self.box_x2, self.box_y2 = 380, 400

        # Camera pose from URDF (reference) — used for bad-fruit points
        self.cam_translation = np.array([-1.08, 0.007, 1.09], dtype=float)
        roll, pitch, yaw = 0.0, 0.7330383, 0.0
        self.cam_rot = self._rot_z(yaw) @ self._rot_y(pitch) @ self._rot_x(roll)

        self.get_logger().info("Merged node started ✔ (Aruco + Bad-fruit detection)")

    # ---------- rotation helpers ----------
    def _rot_x(self, a):
        ca, sa = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0],
                         [0, ca, -sa],
                         [0, sa, ca]])

    def _rot_y(self, a):
        ca, sa = math.cos(a), math.sin(a)
        return np.array([[ca, 0, sa],
                         [0, 1, 0],
                         [-sa, 0, ca]])

    def _rot_z(self, a):
        ca, sa = math.cos(a), math.sin(a)
        return np.array([[ca, -sa, 0],
                         [sa, ca, 0],
                         [0, 0, 1]])

    # ---------------- Camera Info ----------------
    def caminfo_cb(self, msg: CameraInfo):
        # store intrinsics and the camera frame id (very important)
        try:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            # set camera frame from camera_info header (use this for TF lookup)
            self.camera_frame = msg.header.frame_id if msg.header.frame_id != "" else "camera_link"
            self.get_logger().info(f"camera_info received; camera_frame = '{self.camera_frame}'")
        except Exception as e:
            self.get_logger().warn(f"Unable to parse camera_info: {e}")

    # ---------------- Image Callbacks ----------------
    def color_cb(self, msg):
        try:
            self.color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            self.color = None
        self.process()

    def depth_cb(self, msg):
        try:
            self.depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception:
            # fallback for 16UC1
            try:
                depth_u16 = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                depth = depth_u16.astype(np.float32)
                if np.nanmax(depth) > 1000.0:
                    depth = depth / 1000.0
                self.depth = depth
            except Exception:
                self.depth = None

    # ---------------- Utility ----------------
    def _depth_patch(self, u, v, half=4):
        if self.depth is None:
            return None

        h, w = self.depth.shape[:2]
        u0 = max(0, u - half)
        u1 = min(w - 1, u + half)
        v0 = max(0, v - half)
        v1 = min(h - 1, v + half)

        patch = self.depth[v0:v1 + 1, u0:u1 + 1]
        valid = np.isfinite(patch) & (patch > 0)

        if np.count_nonzero(valid) == 0:
            return None

        return float(np.median(patch[valid]))

    # ---------------- Main Processing ----------------
    def process(self):
        # require color, depth, camera intrinsics
        if self.color is None or self.depth is None or self.camera_matrix is None:
            return

        img = self.color.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)

            for i, corner in enumerate(corners):
                marker_id = int(ids[i][0])
                pts = corner.reshape((4, 2)).astype(int)

                # Center pixel
                u = int(np.mean(pts[:, 0]))
                v = int(np.mean(pts[:, 1]))

                depth = self._depth_patch(u, v)
                if depth is None:
                    continue

                # Camera coordinates (kept as in original ArUco code)
                X = depth
                Y = -(u - cx) * depth / fx
                Z = -(v - cy) * depth / fy
                P_cam = np.array([X, Y, Z], float)

                # Transform cam->base using live TF lookup
                # Use camera_frame determined from CameraInfo to be robust
                if self.camera_frame is None:
                    # fallback to previous logic if camera_frame not known yet
                    camera_frame_lookup = "camera_link"
                else:
                    camera_frame_lookup = self.camera_frame

                # check transform availability before lookup
                try:
                    # wait up to 0.2s for transform (non-blocking mostly)
                    can_tf = self.tf_buffer.can_transform(
                        "base_link",
                        camera_frame_lookup,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.2)
                    )
                except Exception:
                    can_tf = False

                if not can_tf:
                    # skip publishing this marker now — TF not ready
                    self.get_logger().warn(
                        f"TF from '{camera_frame_lookup}' to 'base_link' not available yet; skipping marker {marker_id}"
                    )
                    continue

                try:
                    tf = self.tf_buffer.lookup_transform("base_link", camera_frame_lookup, rclpy.time.Time())
                except Exception as e:
                    # if lookup fails skip this marker
                    self.get_logger().warn(f"lookup_transform failed: {e}")
                    continue

                T = tf.transform.translation
                Rq = tf.transform.rotation
                R = tf_transformations.quaternion_matrix(
                    [Rq.x, Rq.y, Rq.z, Rq.w]
                )[:3, :3]

                P_base = R @ P_cam + np.array([T.x, T.y, T.z])

                # Orientation selection for specific IDs
                if marker_id == 3:
                    q = self.q_marker3
                elif marker_id == 6:
                    q = self.q_marker6
                else:
                    q = self.q_marker3

                # Normalize quaternion
                q_arr = np.array(q, dtype=float)
                nrm = np.linalg.norm(q_arr)
                if not np.isfinite(nrm) or nrm < 1e-8:
                    q_arr = np.array([0.0, 0.0, 0.0, 1.0])
                else:
                    q_arr = q_arr / nrm

                # Publish TF for aruco marker (frame = base_link -> aruco_marker_X)
                tf_msg = TransformStamped()
                tf_msg.header.stamp = self.get_clock().now().to_msg()
                tf_msg.header.frame_id = "base_link"
                tf_msg.child_frame_id = f"aruco_marker_{marker_id}"

                tf_msg.transform.translation.x = float(P_base[0])
                tf_msg.transform.translation.y = float(P_base[1])
                tf_msg.transform.translation.z = float(P_base[2])

                tf_msg.transform.rotation.x = float(q_arr[0])
                tf_msg.transform.rotation.y = float(q_arr[1])
                tf_msg.transform.rotation.z = float(q_arr[2])
                tf_msg.transform.rotation.w = float(q_arr[3])

                self.tf_pub.sendTransform(tf_msg)

                # ALSO publish alias expected by Task2B for the can (if this marker is the can)
                if marker_id == CAN_MARKER_ID:
                    alias = TransformStamped()
                    alias.header.stamp = tf_msg.header.stamp
                    alias.header.frame_id = "base_link"
                    alias.child_frame_id = "team_4784_fertiliser_can"
                    alias.transform = tf_msg.transform
                    self.tf_pub.sendTransform(alias)

                # Draw center on image
                cv2.circle(img, (u, v), 6, (0, 255, 0), -1)
                cv2.putText(img, f"ID {marker_id}", (u + 5, v - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                self.get_logger().info(f"[ID {marker_id}] XYZ={P_base.round(4)}  Q={np.round(q_arr,4)}")

        # ---------------- Bad-fruit detection (from reference) ----------------
        # Run regardless of whether ArUco markers were found so both features are available
        self._process_bad_fruits(img)

        cv2.imshow("Aruco Center Fixed Ori", img)
        cv2.waitKey(1)

    # ---------------- Bad fruit detection (from reference, publishing only team_4784_bad_fruit_N) ----------------
    def _process_bad_fruits(self, frame):
        # ensure required data present
        if self.camera_matrix is None or self.depth is None:
            return

        roi = frame[self.box_x1:self.box_x2, self.box_y1:self.box_y2] if False else frame[self.box_y1:self.box_y2, self.box_x1:self.box_x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Detect greyish / pale region (bad fruit)
        lower_bad = np.array([0, 0, 70])
        upper_bad = np.array([180, 40, 200])
        mask_bad = cv2.inRange(hsv_roi, lower_bad, upper_bad)
        mask_bad = cv2.medianBlur(mask_bad, 5)

        contours, _ = cv2.findContours(mask_bad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx_cam = self.camera_matrix[0, 2]
        cy_cam = self.camera_matrix[1, 2]
        bad_count = 1

        # Optionally sort contours by area and limit to 2 (if desired); code kept as original (publishes all)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:
                continue

            x_roi, y_roi, w, h = cv2.boundingRect(contour)
            x, y = int(x_roi + self.box_x1), int(y_roi + self.box_y1)
            h_img, w_img = self.depth.shape[:2]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w_img - 1, x + w), min(h_img - 1, y + h)
            if x1 <= x0 or y1 <= y0:
                continue

            depth_roi = self.depth[y0:y1, x0:x1]
            valid = np.isfinite(depth_roi) & (depth_roi > 0.0)
            valid_depths = depth_roi[valid]
            if valid_depths.size == 0:
                continue
            depth = float(np.median(valid_depths))

            # Draw bounding box & label on the main frame
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame, "bad_fruit", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Center pixel of fruit
            cx_pixel = x + w // 2
            cy_pixel = y + h // 2
            cv2.circle(frame, (cx_pixel, cy_pixel), 5, (0, 255, 0), -1)  # green dot

            # --- 3D point in camera frame (kept same convention as reference) ---
            X_cam = depth
            Y_cam = -(cx_pixel - cx_cam) * depth / fx
            Z_cam = -(cy_pixel - cy_cam) * depth / fy
            point_cam = np.array([X_cam, Y_cam, Z_cam], dtype=float)

            # --- Transform to base_link using URDF-based cam_rot & cam_translation (reference) ---
            point_base = self.cam_rot @ point_cam + self.cam_translation

            # Publish transform + point (only team_4784_bad_fruit_N)
            self._publish_bad_fruit_transform(point_base, bad_count)
            self._publish_bad_fruit_point(point_base)
            bad_count += 1

    def _publish_bad_fruit_transform(self, point, fruit_id):
        # publish only the alias expected by Task2B
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = f"team_4784_bad_fruit_{fruit_id}"
        t.transform.translation.x = float(point[0])
        t.transform.translation.y = float(point[1])
        t.transform.translation.z = float(point[2])
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_pub.sendTransform(t)

        self.get_logger().info(
            f"TF -> {t.child_frame_id}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"
        )

    def _publish_bad_fruit_point(self, point):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = float(point[2])
        self.point_pub.publish(msg)

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = ArucoCenterFixedOri()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

