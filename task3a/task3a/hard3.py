#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image as RosImage, CameraInfo
from geometry_msgs.msg import TransformStamped, PointStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import numpy as np
import cv2
import math
import tf_transformations

"""
Hardware-ready Task3 node:
 - Default topics match remote hardware (color/depth/camera_info)
 - Publishes TFs (parent = base_link):
     <TEAM>_fertilizer_1   (ArUco ID CAN_MARKER_ID)  <-- uses real ArUco orientation
     <TEAM>_bad_fruit_N    (N up to MAX_BAD_FRUITS)
 - Publishes /task3/annotated_image (bgr8) and /bad_fruit_points (PointStamped)
 - Params allow tuning without file edits.
"""

# Default configuration (override via ros2 params)
CAN_MARKER_ID_DEFAULT = 3

class Task3ANode(Node):
    def __init__(self):
        super().__init__("task3a_detector")

        # --- parameters (can be changed with ros2 param set) ---
        self.declare_parameter("team_prefix", "4784")
        self.declare_parameter("can_marker_id", CAN_MARKER_ID_DEFAULT)
        self.declare_parameter("marker_length_m", 0.05)   # ArUco marker side in meters (tune)
        self.declare_parameter("max_bad_fruits", 3)
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("caminfo_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("show_gui", False)  # headless remote default
        self.declare_parameter("roi_x1", 5)
        self.declare_parameter("roi_y1", 210)
        self.declare_parameter("roi_x2", 380)
        self.declare_parameter("roi_y2", 400)
        self.declare_parameter("min_contour_area", 300)

        # read params
        self.TEAM_PREFIX = self.get_parameter("team_prefix").get_parameter_value().string_value
        self.CAN_MARKER_ID = int(self.get_parameter("can_marker_id").get_parameter_value().integer_value)
        self.marker_length = float(self.get_parameter("marker_length_m").get_parameter_value().double_value)
        self.MAX_BAD_FRUITS = int(self.get_parameter("max_bad_fruits").get_parameter_value().integer_value)
        color_topic = self.get_parameter("color_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        caminfo_topic = self.get_parameter("caminfo_topic").get_parameter_value().string_value
        self.show_gui = bool(self.get_parameter("show_gui").get_parameter_value().bool_value)
        self.box_x1 = int(self.get_parameter("roi_x1").get_parameter_value().integer_value)
        self.box_y1 = int(self.get_parameter("roi_y1").get_parameter_value().integer_value)
        self.box_x2 = int(self.get_parameter("roi_x2").get_parameter_value().integer_value)
        self.box_y2 = int(self.get_parameter("roi_y2").get_parameter_value().integer_value)
        self.min_contour_area = int(self.get_parameter("min_contour_area").get_parameter_value().integer_value)

        # cv bridge + images
        self.bridge = CvBridge()
        self.color = None
        self.depth = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = None

        # subscribers (use the hardware topics by default)
        self.create_subscription(RosImage, color_topic, self.color_cb, 5)
        self.create_subscription(RosImage, depth_topic, self.depth_cb, 5)
        self.create_subscription(CameraInfo, caminfo_topic, self.caminfo_cb, 5)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # publishers
        self.point_pub = self.create_publisher(PointStamped, '/bad_fruit_points', 10)
        self.annot_pub = self.create_publisher(RosImage, '/task3/annotated_image', 6)

        # aruco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        try:
            self.detector_params = cv2.aruco.DetectorParameters_create()
        except Exception:
            self.detector_params = cv2.aruco.DetectorParameters()

        # fallback cam pose (from URDF/reference) if live TF not present
        self.cam_translation = np.array([-1.08, 0.007, 1.09], dtype=float)
        roll, pitch, yaw = 0.0, 0.7330383, 0.0
        self.cam_rot = self._rot_z(yaw) @ self._rot_y(pitch) @ self._rot_x(roll)

        self.get_logger().info(f"Task3A node started — TEAM={self.TEAM_PREFIX} CAN_ID={self.CAN_MARKER_ID} MAX_BAD_FRUITS={self.MAX_BAD_FRUITS}")

    # rotation helpers
    def _rot_x(self, a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def _rot_y(self, a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    def _rot_z(self, a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    # camera info callback
    def caminfo_cb(self, msg: CameraInfo):
        try:
            self.camera_matrix = np.array(msg.k).reshape((3,3))
            self.dist_coeffs = np.array(msg.d)
            self.camera_frame = msg.header.frame_id if msg.header.frame_id else "camera_link"
            self.get_logger().info(f"camera_info received; frame_id='{self.camera_frame}'")
        except Exception as e:
            self.get_logger().warning(f"caminfo parse failed: {e}")

    # image callbacks
    def color_cb(self, msg: RosImage):
        try:
            self.color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            self.color = None
        self.process()

    def depth_cb(self, msg: RosImage):
        # handle 32FC1 and 16UC1 robustly
        try:
            self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception:
            try:
                depth_u16 = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
                depth = depth_u16.astype(np.float32)
                # convert mm to meters if needed
                if np.nanmax(depth) > 1000.0:
                    depth = depth / 1000.0
                self.depth = depth
            except Exception:
                self.depth = None

    # small local median of depth
    def _depth_patch(self, u, v, half=4):
        if self.depth is None:
            return None
        h, w = self.depth.shape[:2]
        u0 = max(0, u-half); u1 = min(w-1, u+half)
        v0 = max(0, v-half); v1 = min(h-1, v+half)
        patch = self.depth[v0:v1+1, u0:u1+1]
        valid = np.isfinite(patch) & (patch > 0)
        if np.count_nonzero(valid) == 0:
            return None
        return float(np.median(patch[valid]))

    # main processing
    def process(self):
        # need color + depth + intrinsics
        if self.color is None or self.depth is None or self.camera_matrix is None:
            return

        img = self.color.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        fx = float(self.camera_matrix[0,0])
        fy = float(self.camera_matrix[1,1])
        cx = float(self.camera_matrix[0,2])
        cy = float(self.camera_matrix[1,2])

        # ---- ArUco detection & CAN pose (use estimatePoseSingleMarkers) ----
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.detector_params)
        if ids is not None and len(ids) > 0:
            # estimate pose in camera frame (rvecs, tvecs)
            try:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_length, self.camera_matrix, self.dist_coeffs
                )
            except Exception:
                rvecs, tvecs = None, None

            for idx, id_arr in enumerate(ids):
                marker_id = int(id_arr[0])
                pts = corners[idx].reshape((4,2)).astype(int)
                u = int(np.mean(pts[:,0])); v = int(np.mean(pts[:,1]))

                # if tvecs available, use it (marker position in camera frame)
                if tvecs is not None:
                    tvec = tvecs[idx][0]   # (x,y,z) in meters (camera frame)
                    rvec = rvecs[idx][0]
                    # rotation matrix and quaternion of marker in camera frame
                    R_cam_marker, _ = cv2.Rodrigues(rvec)
                    marker_quat_cam = tf_transformations.quaternion_from_matrix(
                        np.vstack((np.hstack((R_cam_marker, np.zeros((3,1)))), [0,0,0,1]))
                    )
                    P_cam = np.array([float(tvec[0]), float(tvec[1]), float(tvec[2])], dtype=float)
                else:
                    # fallback: compute from depth patch + projection (less accurate)
                    depth = self._depth_patch(u, v)
                    if depth is None:
                        continue
                    X = depth
                    Y = -(u - cx) * depth / fx
                    Z = -(v - cy) * depth / fy
                    P_cam = np.array([X, Y, Z], dtype=float)
                    marker_quat_cam = np.array([0.0, 0.0, 0.0, 1.0])

                # transform marker pose to base_link using live TF if available
                camera_frame_lookup = self.camera_frame if self.camera_frame else "camera_link"
                use_live_tf = True
                try:
                    # try lookup; many hardware setups publish camera->base transform
                    tf = self.tf_buffer.lookup_transform("base_link", camera_frame_lookup, rclpy.time.Time())
                    T = tf.transform.translation
                    Rq = [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
                    R_cam_to_base = tf_transformations.quaternion_matrix(Rq)[:3,:3]
                    P_base = R_cam_to_base @ P_cam + np.array([T.x, T.y, T.z])
                    # rotate marker quaternion into base: q_base = q_cam2base * q_marker_cam
                    q_base = tf_transformations.quaternion_multiply(Rq, marker_quat_cam)
                except Exception:
                    # fallback: use URDF-based cam_rot + cam_translation
                    use_live_tf = False
                    P_base = self.cam_rot @ P_cam + self.cam_translation
                    q_base = marker_quat_cam

                # Publish only the fertilizer alias when this marker is the can ID
                if marker_id == self.CAN_MARKER_ID:
                    tf_msg = TransformStamped()
                    tf_msg.header.stamp = self.get_clock().now().to_msg()
                    tf_msg.header.frame_id = "base_link"
                    tf_msg.child_frame_id = f"{self.TEAM_PREFIX}_fertilizer_1"
                    tf_msg.transform.translation.x = float(P_base[0])
                    tf_msg.transform.translation.y = float(P_base[1])
                    tf_msg.transform.translation.z = float(P_base[2])
                    # assign quaternion (ensure normalized)
                    qb = np.array(q_base, dtype=float)
                    n = np.linalg.norm(qb)
                    if n < 1e-8 or not np.isfinite(n):
                        qb = np.array([0.0,0.0,0.0,1.0])
                    else:
                        qb = qb / n
                    tf_msg.transform.rotation.x = float(qb[0])
                    tf_msg.transform.rotation.y = float(qb[1])
                    tf_msg.transform.rotation.z = float(qb[2])
                    tf_msg.transform.rotation.w = float(qb[3])
                    self.tf_broadcaster.sendTransform(tf_msg)
                    self.get_logger().info(f"Published {tf_msg.child_frame_id} at ({P_base[0]:.3f},{P_base[1]:.3f},{P_base[2]:.3f}) using {'live TF' if use_live_tf else 'fallback'}")

                # draw detections on annotated image
                cv2.circle(img, (u,v), 5, (0,255,0), -1)
                cv2.putText(img, f"ID {marker_id}", (u+6, v-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # ---- Bad-fruit detection (ROI) ----
        self._process_bad_fruits(img, fx, fy, cx, cy)

        # draw ROI for visualization
        cv2.rectangle(img, (self.box_x1, self.box_y1), (self.box_x2, self.box_y2), (255,0,0), 2)
        cv2.putText(img, "ROI - Fruit Detection", (self.box_x1, max(10, self.box_y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        # publish annotated image for rqt_image_view
        try:
            ros_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            ros_img.header.stamp = self.get_clock().now().to_msg()
            ros_img.header.frame_id = self.camera_frame if self.camera_frame else "camera_link"
            self.annot_pub.publish(ros_img)
        except Exception as e:
            self.get_logger().warning(f"Failed to publish annotated image: {e}")

        # optionally show GUI (remote typically headless => usually False)
        if self.show_gui:
            try:
                cv2.imshow("Task3 - annotated", img)
                cv2.waitKey(1)
            except Exception:
                self.get_logger().warning("cv2.imshow failed (likely headless); disabling GUI")
                self.show_gui = False

    def _process_bad_fruits(self, frame, fx, fy, cx, cy):
        if self.camera_matrix is None or self.depth is None:
            return

        roi = frame[self.box_y1:self.box_y2, self.box_x1:self.box_x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_bad = np.array([0, 0, 70])
        upper_bad = np.array([180, 40, 200])
        mask_bad = cv2.inRange(hsv, lower_bad, upper_bad)
        mask_bad = cv2.medianBlur(mask_bad, 5)

        contours, _ = cv2.findContours(mask_bad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

        bad_count = 1
        for c in contours_sorted:
            if bad_count > self.MAX_BAD_FRUITS:
                break
            area = cv2.contourArea(c)
            if area < self.min_contour_area:
                continue

            x_roi, y_roi, w, h = cv2.boundingRect(c)
            x_global = x_roi + self.box_x1
            y_global = y_roi + self.box_y1

            # safety checks
            h_img, w_img = self.depth.shape[:2]
            x0, y0 = max(0, x_global), max(0, y_global)
            x1, y1 = min(w_img - 1, x_global + w), min(h_img - 1, y_global + h)
            if x1 <= x0 or y1 <= y0:
                continue

            depth_roi = self.depth[y0:y1, x0:x1]
            valid = np.isfinite(depth_roi) & (depth_roi > 0.0)
            valid_depths = depth_roi[valid]
            if valid_depths.size == 0:
                continue
            depth = float(np.median(valid_depths))

            # draw
            cv2.rectangle(frame, (x0,y0), (x1,y1), (0,0,255), 2)
            cv2.putText(frame, "bad_fruit", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            cx_pixel = x_global + w//2
            cy_pixel = y_global + h//2
            cv2.circle(frame, (cx_pixel, cy_pixel), 5, (0,255,0), -1)

            # 3D point in camera frame (projection, same convention)
            X_cam = depth
            Y_cam = -(cx_pixel - cx) * depth / fx
            Z_cam = -(cy_pixel - cy) * depth / fy
            point_cam = np.array([X_cam, Y_cam, Z_cam], dtype=float)

            # transform to base_link (prefer live TF)
            camera_frame_lookup = self.camera_frame if self.camera_frame else "camera_link"
            try:
                tf = self.tf_buffer.lookup_transform("base_link", camera_frame_lookup, rclpy.time.Time())
                T = tf.transform.translation
                Rq = [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
                R_cam_to_base = tf_transformations.quaternion_matrix(Rq)[:3,:3]
                point_base = R_cam_to_base @ point_cam + np.array([T.x, T.y, T.z])
            except Exception:
                point_base = self.cam_rot @ point_cam + self.cam_translation

            # publish transform alias and point
            child = f"{self.TEAM_PREFIX}_bad_fruit_{bad_count}"
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "base_link"
            t.child_frame_id = child
            t.transform.translation.x = float(point_base[0])
            t.transform.translation.y = float(point_base[1])
            t.transform.translation.z = float(point_base[2])
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(t)
            self.get_logger().info(f"Published {child} at ({point_base[0]:.3f},{point_base[1]:.3f},{point_base[2]:.3f})")

            # publish PointStamped
            pmsg = PointStamped()
            pmsg.header.stamp = self.get_clock().now().to_msg()
            pmsg.header.frame_id = "base_link"
            pmsg.point.x = float(point_base[0])
            pmsg.point.y = float(point_base[1])
            pmsg.point.z = float(point_base[2])
            self.point_pub.publish(pmsg)

            bad_count += 1

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = Task3ANode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
