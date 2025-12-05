#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage, CameraInfo
from geometry_msgs.msg import TransformStamped, PointStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import numpy as np
import cv2
import math
import tf_transformations

"""
Task-3 node (ready):
 - Subscribes (defaults):
    color: /camera/camera/color/image_raw
    depth: /camera/camera/aligned_depth_to_color/image_raw
    camera_info: /camera/camera/color/camera_info
 - Publishes TFs (all relative to base_link):
    4784_fertilizer_1   (when ArUco CAN_MARKER_ID seen)
    4784_bad_fruit_1..2 (up to 2 bad fruits per frame)
 - Publishes: /bad_fruit_points (PointStamped) and /task3/annotated_image (bgr8)
 - Shows a local CV window if `show_gui` param is true; otherwise publishes annotated image only.
"""

# ---------- CONFIG ----------
CAN_MARKER_ID = 3
TEAM_PREFIX = "4784"       # exactly as requested (no "team_" prefix)
MAX_BAD_FRUITS = 2        # only 2 bad fruits

class ArucoCenterFixedOri(Node):
    def __init__(self):
        super().__init__("aruco_center_fixed_ori_task3")
        self.bridge = CvBridge()
        self.color = None
        self.depth = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # parameters (defaults tuned for typical remote / sim camera topics)
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("caminfo_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("show_gui", True)

        color_topic = self.get_parameter("color_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        caminfo_topic = self.get_parameter("caminfo_topic").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.show_gui = self.get_parameter("show_gui").get_parameter_value().bool_value

        # subscribers
        self.create_subscription(RosImage, color_topic, self.color_cb, 5)
        self.create_subscription(RosImage, depth_topic, self.depth_cb, 5)
        self.create_subscription(CameraInfo, caminfo_topic, self.caminfo_cb, 5)

        # TF system
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_pub = TransformBroadcaster(self)

        # bad-fruit point publisher
        self.point_pub = self.create_publisher(PointStamped, '/bad_fruit_points', 10)

        # annotated image publisher (for RViz / rqt_image_view)
        self.img_pub = self.create_publisher(RosImage, '/task3/annotated_image', 6)

        # ArUco detector setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.params = cv2.aruco.DetectorParameters_create()
        except Exception:
            self.params = cv2.aruco.DetectorParameters()

        # Marker 3 → fixed quaternion (normalized)
        q3_raw = np.array([0.707, 0.028, 0.034, 0.707], dtype=float)
        self.q_marker3 = (q3_raw / np.linalg.norm(q3_raw)).tolist()

        # Marker 6 → Euler (0, π, 0) — convert to list to avoid .tolist() issues
        self.q_marker6 = list(tf_transformations.quaternion_from_euler(0, math.pi, 0))

        # ROI for fruit detection (pixels) — change if image resolution differs
        self.box_x1, self.box_y1 = 5, 210
        self.box_x2, self.box_y2 = 380, 400

        # URDF-based camera pose fallback (cam -> base)
        self.cam_translation = np.array([-1.08, 0.007, 1.09], dtype=float)
        roll, pitch, yaw = 0.0, 0.7330383, 0.0
        self.cam_rot = self._rot_z(yaw) @ self._rot_y(pitch) @ self._rot_x(roll)

        self.get_logger().info(f"Task3 node started — TEAM={TEAM_PREFIX} MAX_BAD_FRUITS={MAX_BAD_FRUITS} show_gui={self.show_gui}")

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

    # ---------- callbacks ----------
    def caminfo_cb(self, msg: CameraInfo):
        try:
            self.camera_matrix = np.array(msg.k).reshape(3,3)
            self.dist_coeffs = np.array(msg.d)
        except Exception as e:
            self.get_logger().warning(f"caminfo parse failed: {e}")

    def color_cb(self, msg: RosImage):
        try:
            self.color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            self.color = None
        self.process()

    def depth_cb(self, msg: RosImage):
        try:
            self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception:
            try:
                self.depth = self.bridge.imgmsg_to_cv2(msg)
            except Exception:
                self.depth = None

    # small depth patch (median)
    def _depth_patch(self, u, v, half=4):
        if self.depth is None:
            return None
        h, w = self.depth.shape[:2]
        u0 = max(0, u-half)
        u1 = min(w-1, u+half)
        v0 = max(0, v-half)
        v1 = min(h-1, v+half)
        patch = self.depth[v0:v1+1, u0:u1+1]
        valid = np.isfinite(patch) & (patch > 0)
        if np.count_nonzero(valid) == 0:
            return None
        return float(np.median(patch[valid]))

    # ---------- main processing ----------
    def process(self):
        # require color, depth, camera intrinsics
        if self.color is None or self.depth is None or self.camera_matrix is None:
            return

        img = self.color.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect aruco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

        fx = self.camera_matrix[0,0]
        fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]
        cy = self.camera_matrix[1,2]

        # If ArUco detected, publish ONLY fertilizer alias for CAN_MARKER_ID
        if ids is not None:
            for i, corner in enumerate(corners):
                try:
                    marker_id = int(ids[i][0])
                except Exception:
                    continue

                pts = corner.reshape((4,2)).astype(int)
                u = int(np.mean(pts[:,0]))
                v = int(np.mean(pts[:,1]))

                depth = self._depth_patch(u, v)
                if depth is None:
                    continue

                # camera-frame 3D point (kept same sign convention)
                X = depth
                Y = -(u - cx) * depth / fx
                Z = -(v - cy) * depth / fy
                P_cam = np.array([X, Y, Z], float)

                # prefer live TF (camera_link -> base_link) and fallback to URDF pose
                try:
                    tf = self.tf_buffer.lookup_transform("base_link", self.camera_frame, rclpy.time.Time())
                    T = tf.transform.translation
                    Rq = tf.transform.rotation
                    R = tf_transformations.quaternion_matrix([Rq.x, Rq.y, Rq.z, Rq.w])[:3,:3]
                    P_base = R @ P_cam + np.array([T.x, T.y, T.z])
                except Exception:
                    P_base = self.cam_rot @ P_cam + self.cam_translation

                # If this is the CAN marker, publish fertilizer alias TF only
                if marker_id == CAN_MARKER_ID:
                    alias = TransformStamped()
                    alias.header.stamp = self.get_clock().now().to_msg()
                    alias.header.frame_id = "base_link"
                    alias.child_frame_id = f"{TEAM_PREFIX}_fertilizer_1"
                    alias.transform.translation.x = float(P_base[0])
                    alias.transform.translation.y = float(P_base[1])
                    alias.transform.translation.z = float(P_base[2])
                    # use fixed quaternion for marker 3
                    alias.transform.rotation.x = self.q_marker3[0]
                    alias.transform.rotation.y = self.q_marker3[1]
                    alias.transform.rotation.z = self.q_marker3[2]
                    alias.transform.rotation.w = self.q_marker3[3]
                    self.tf_pub.sendTransform(alias)
                    self.get_logger().info(f"TF -> {alias.child_frame_id}: ({P_base[0]:.3f}, {P_base[1]:.3f}, {P_base[2]:.3f})")

                # draw center and id text for visualization
                cv2.circle(img, (u, v), 6, (0,255,0), -1)
                cv2.putText(img, f"ID {marker_id}", (u+5, v-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # process bad fruits (publish up to MAX_BAD_FRUITS)
        self._process_bad_fruits(img)

        # draw ROI box for visualization
        cv2.rectangle(img, (self.box_x1, self.box_y1), (self.box_x2, self.box_y2), (255, 0, 0), 2)
        cv2.putText(img, "ROI - Fruit Detection", (self.box_x1, max(10, self.box_y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # publish annotated image for RViz/rqt
        try:
            ros_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            ros_img.header.stamp = self.get_clock().now().to_msg()
            ros_img.header.frame_id = self.camera_frame
            self.img_pub.publish(ros_img)
        except Exception as e:
            self.get_logger().warning(f"Failed to publish annotated image: {e}")

        # optionally show GUI locally (if display available)
        if self.show_gui:
            try:
                cv2.imshow("Task3 - ArUco + BadFruit (annotated)", img)
                cv2.waitKey(1)
            except Exception as e:
                # if imshow fails (headless), disable GUI so we don't spam errors
                self.get_logger().warning(f"cv2.imshow failed — disabling GUI: {e}")
                self.show_gui = False

    # ---------- bad-fruit detection ----------
    def _process_bad_fruits(self, frame):
        if self.camera_matrix is None or self.depth is None:
            return

        # crop ROI
        roi = frame[self.box_y1:self.box_y2, self.box_x1:self.box_x2]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # thresholds for greyish / pale (bad fruit)
        lower_bad = np.array([0, 0, 70])
        upper_bad = np.array([180, 40, 200])
        mask_bad = cv2.inRange(hsv_roi, lower_bad, upper_bad)
        mask_bad = cv2.medianBlur(mask_bad, 5)

        contours, _ = cv2.findContours(mask_bad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fx = self.camera_matrix[0,0]
        fy = self.camera_matrix[1,1]
        cx_cam = self.camera_matrix[0,2]
        cy_cam = self.camera_matrix[1,2]

        bad_count = 1
        # sort largest-first for ID stability
        contours_sorted = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

        for contour in contours_sorted:
            if bad_count > MAX_BAD_FRUITS:
                break

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

            # bounding box on frame
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame, "bad_fruit", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cx_pixel = x + w // 2
            cy_pixel = y + h // 2
            cv2.circle(frame, (cx_pixel, cy_pixel), 5, (0, 255, 0), -1)

            # 3D point in camera frame (same convention)
            X_cam = depth
            Y_cam = -(cx_pixel - cx_cam) * depth / fx
            Z_cam = -(cy_pixel - cy_cam) * depth / fy
            point_cam = np.array([X_cam, Y_cam, Z_cam], dtype=float)

            # prefer live TF for transform to base, fallback to URDF pose
            try:
                tf = self.tf_buffer.lookup_transform("base_link", self.camera_frame, rclpy.time.Time())
                T = tf.transform.translation
                Rq = tf.transform.rotation
                R = tf_transformations.quaternion_matrix([Rq.x, Rq.y, Rq.z, Rq.w])[:3,:3]
                point_base = R @ point_cam + np.array([T.x, T.y, T.z])
            except Exception:
                point_base = self.cam_rot @ point_cam + self.cam_translation

            # publish transform & point
            child_name = f"{TEAM_PREFIX}_bad_fruit_{bad_count}"
            self._publish_bad_fruit_transform(point_base, child_name)
            self._publish_bad_fruit_point(point_base)
            bad_count += 1

    def _publish_bad_fruit_transform(self, point, child_frame_name):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = child_frame_name
        t.transform.translation.x = float(point[0])
        t.transform.translation.y = float(point[1])
        t.transform.translation.z = float(point[2])
        # neutral orientation
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_pub.sendTransform(t)
        self.get_logger().info(f"TF -> {child_frame_name}: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})")

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
