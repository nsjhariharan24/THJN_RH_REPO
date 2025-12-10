#!/usr/bin/env python3
'''
# Team ID:          4784
# Theme:            Krishi coBot
# Author List:      Hariharan S, Thilakraj S, Jeevan Uday Alexander, Niranjan R
# Filename:         task3_fruit_and_aruco.py
# Functions:        caminfo_cb, color_cb, depth_cb, process, _process_fruits,
#                   _depth_to_m, _depth_patch, _rot_x, _rot_y, _rot_z, main
# Global variables: CAN_MARKER_ID, TEAM_PREFIX, MAX_BAD_FRUITS
'''

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

# -------------------------- CONFIGURATION CONSTANTS --------------------------
CAN_MARKER_ID = 3                   # ID of ArUco marker on fertilizer can
TEAM_PREFIX = "4784"                # Prefix for TF frame names
MAX_BAD_FRUITS = 2                  # Maximum number of bad fruits to track


class Task3ADetection(Node):
    '''
    Purpose:
    ---
    This ROS node detects:
    1. A specific ArUco marker (fertilizer can) and publishes its TF and 3D pose
    2. Bad fruits inside a fixed ROI, estimates their 3D position, 
       assigns stable IDs, and publishes TF + PointStamped
    
    It subscribes to RGB, depth, and camera info topics and publishes:
    - TF frames for fertilizer marker and fruit centroids
    - /bad_fruit_points topic (PointStamped)
    - /task3/annotated_image (visualization)
    '''

    def __init__(self):
        '''
        Purpose:
        ---
        Constructor for the node. Initializes subscribers, publishers, TF, 
        ArUco parameters, and fruit-detection variables.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        node = Task3ADetection()
        '''
        super().__init__("aruco_center_fixed_ori_task3")

        # ------------------- CV Bridge for ROS <-> OpenCV -------------------
        self.bridge = CvBridge()
        self.color = None             # stores latest RGB frame
        self.depth = None             # stores latest depth frame
        self.camera_matrix = None     # intrinsic matrix from CameraInfo

        # ------------------- Declare ROS Parameters -------------------
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("caminfo_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("show_gui", False)

        # Retrieve parameter values
        color_topic = self.get_parameter("color_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        caminfo_topic = self.get_parameter("caminfo_topic").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.show_gui = self.get_parameter("show_gui").get_parameter_value().bool_value

        # ------------------- Subscribers -------------------
        self.create_subscription(RosImage, color_topic, self.color_cb, 5)
        self.create_subscription(RosImage, depth_topic, self.depth_cb, 5)
        self.create_subscription(CameraInfo, caminfo_topic, self.caminfo_cb, 5)

        # ------------------- TF Setup -------------------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_pub = TransformBroadcaster(self)

        # ------------------- Publishers -------------------
        self.point_pub = self.create_publisher(PointStamped, '/bad_fruit_points', 10)
        self.img_pub = self.create_publisher(RosImage, '/task3/annotated_image', 6)

        # ------------------- ArUco Initialization -------------------
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        except Exception:
            self.get_logger().warn("Failed to load ArUco dictionary.")

        try:
            self.params = cv2.aruco.DetectorParameters()
        except AttributeError:
            self.params = cv2.aruco.DetectorParameters_create()

        # Orientation of fertilizer marker (pre-calibrated quaternion)
        q = np.array([0.707, 0.028, 0.034, 0.707])
        self.q_marker3 = (q / np.linalg.norm(q)).tolist()

        # ------------------- Region of Interest for Fruits -------------------
        self.box_x1, self.box_y1 = 60, 430
        self.box_x2, self.box_y2 = 375, 600

        # ------------------- Fallback Camera-to-Base Transform -------------------
        self.cam_translation = np.array([-1.08, 0.007, 1.09])
        pitch = 0.7330383
        self.cam_rot = (
            self._rot_y(pitch)
        )

        # Tracking previous fruit locations for stable IDs
        self.prev_fruit_positions = {i: None for i in range(1, MAX_BAD_FRUITS + 1)}

        self.get_logger().info("Task3 node started — Fertilizer + Fruit TF publisher initialized.")

    # ============================================================================
    #                          ROTATION MATRIX HELPERS
    # ============================================================================

    def _rot_x(self, angle):
        '''
        Purpose:
        ---
        Computes rotation matrix for rotation about the X-axis.

        Input Arguments:
        ---
        `angle` : [float]
            Angle in radians

        Returns:
        ---
        rot_mat : [numpy.ndarray]
            3×3 rotation matrix

        Example call:
        ---
        self._rot_x(1.57)
        '''
        ca, sa = math.cos(angle), math.sin(angle)
        return np.array([[1, 0, 0],
                         [0, ca, -sa],
                         [0, sa, ca]])

    def _rot_y(self, angle):
        '''
        Purpose:
        ---
        Computes rotation matrix for rotation about the Y-axis.

        Input Arguments:
        ---
        `angle` : [float]
            Angle in radians

        Returns:
        ---
        rot_mat : [numpy.ndarray]
            3×3 rotation matrix
        '''
        ca, sa = math.cos(angle), math.sin(angle)
        return np.array([[ca, 0, sa],
                         [0, 1, 0],
                         [-sa, 0, ca]])

    def _rot_z(self, angle):
        '''
        Purpose:
        ---
        Computes rotation matrix for rotation about the Z-axis.
        '''
        ca, sa = math.cos(angle), math.sin(angle)
        return np.array([[ca, -sa, 0],
                         [sa, ca, 0],
                         [0, 0, 1]])

    # ============================================================================
    #                                 CALLBACKS
    # ============================================================================

    def caminfo_cb(self, msg):
        '''
        Purpose:
        ---
        Stores camera intrinsic matrix from CameraInfo topic.

        Input Arguments:
        ---
        `msg` : [CameraInfo]
            ROS CameraInfo message

        Returns:
        ---
        None
        '''
        try:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
        except Exception as e:
            self.get_logger().error(f"Error parsing camera matrix: {e}")

    def color_cb(self, msg):
        '''
        Purpose:
        ---
        Receives RGB frame and triggers fruit + ArUco processing.

        Input Arguments:
        ---
        `msg` : [sensor_msgs/Image]
            RGB image message

        Returns:
        ---
        None
        '''
        try:
            self.color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Color CV bridge error: {e}")
            return

        self.process()

    def depth_cb(self, msg):
        '''
        Purpose:
        ---
        Receives depth frame and stores it.

        Input Arguments:
        ---
        `msg` : [sensor_msgs/Image]
            Depth image message

        Returns:
        ---
        None
        '''
        try:
            self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception:
            try:
                self.depth = self.bridge.imgmsg_to_cv2(msg)
            except Exception as e:
                self.get_logger().error(f"Depth CV bridge error: {e}")
                self.depth = None

    # ============================================================================
    #                          DEPTH HELPER FUNCTIONS
    # ============================================================================

    def _depth_to_m(self, d):
        '''
        Purpose:
        ---
        Converts depth to meters (supports both mm & meter streams)

        Input Arguments:
        ---
        `d` : [float]
            Raw depth value

        Returns:
        ---
        depth_m : [float or None]
            Depth in meters OR None if invalid
        '''
        if d is None:
            return None
        if not np.isfinite(d) or d <= 0:
            return None

        d = float(d)
        if d > 10.0:       # mm stream
            return d / 1000.0
        return d           # already meters

    def _depth_patch(self, u, v):
        '''
        Purpose:
        ---
        Extracts a 9×9 depth patch around pixel (u,v) and returns median valid depth.

        Input Arguments:
        ---
        `u` : [int] pixel x
        `v` : [int] pixel y

        Returns:
        ---
        depth_m : [float or None]
        '''
        if self.depth is None:
            return None

        h, w = self.depth.shape
        if not (0 <= u < w and 0 <= v < h):
            return None

        patch = self.depth[max(0, v - 4):min(h, v + 5),
                           max(0, u - 4):min(w, u + 5)]

        ok = np.isfinite(patch) & (patch > 0)
        if ok.sum() == 0:
            return None

        return self._depth_to_m(np.median(patch[ok]))

    # ============================================================================
    #                         MAIN PROCESSING FUNCTION
    # ============================================================================

    def process(self):
        '''
        Purpose:
        ---
        Central processing function. Performs:
        - ArUco detection
        - Fruit detection
        - Depth-based 3D coordinate estimation
        - TF and PointStamped publishing

        Input Arguments:
        ---
        None

        Returns:
        ---
        None
        '''
        if self.color is None or self.depth is None or self.camera_matrix is None:
            return

        img = self.color.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ---------------- ARUCO DETECTION ----------------
        try:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
        except Exception:
            res = cv2.aruco.detectMarkers(gray, self.aruco_dict)
            corners, ids = res[:2] if isinstance(res, tuple) else (None, None)

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # ---------------- HANDLE ARUCO ----------------
        if ids is not None:
            for i, crn in enumerate(corners):
                ID = int(ids[i][0])
                pts = crn.reshape(4, 2).astype(int)
                u = int(np.mean(pts[:, 0]))
                v = int(np.mean(pts[:, 1]))

                d = self._depth_patch(u, v)
                if d is None:
                    continue

                # Project to camera frame
                X = d
                Y = -(u - cx) * d / fx
                Z = -(v - cy) * d / fy
                P_cam = np.array([X, Y, Z])

                # Transform to base frame
                try:
                    tf = self.tf_buffer.lookup_transform("base_link", self.camera_frame, rclpy.time.Time())
                    R = tf_transformations.quaternion_matrix([
                        tf.transform.rotation.x,
                        tf.transform.rotation.y,
                        tf.transform.rotation.z,
                        tf.transform.rotation.w])[:3, :3]
                    T = np.array([
                        tf.transform.translation.x,
                        tf.transform.translation.y,
                        tf.transform.translation.z])
                    P_base = R @ P_cam + T
                except Exception:
                    P_base = self.cam_rot @ P_cam + self.cam_translation

                # Only publish if fertilizer marker detected
                if ID == CAN_MARKER_ID:
                    msg = TransformStamped()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = "base_link"
                    msg.child_frame_id = f"{TEAM_PREFIX}_fertilizer_1"

                    msg.transform.translation.x = float(P_base[0])
                    msg.transform.translation.y = float(P_base[1])
                    msg.transform.translation.z = float(P_base[2])

                    msg.transform.rotation.x = self.q_marker3[0]
                    msg.transform.rotation.y = self.q_marker3[1]
                    msg.transform.rotation.z = self.q_marker3[2]
                    msg.transform.rotation.w = self.q_marker3[3]

                    self.tf_pub.sendTransform(msg)

        # ---------------- FRUIT PROCESSING ----------------
        self._process_fruits(img)

        # Draw ROI for visualization
        cv2.rectangle(img, (self.box_x1, self.box_y1), (self.box_x2, self.box_y2), (255, 0, 0), 2)

        # Publish annotated frame
        try:
            out = self.bridge.cv2_to_imgmsg(img, "bgr8")
            out.header.frame_id = self.camera_frame
            self.img_pub.publish(out)
        except Exception as e:
            self.get_logger().error(f"Annotated image publish error: {e}")

        if self.show_gui:
            cv2.imshow("Task3 Fruits", img)
            cv2.waitKey(1)

    # ============================================================================
    #                        FRUIT PROCESSING FUNCTION
    # ============================================================================

    def _process_fruits(self, frame):
        '''
        Purpose:
        ---
        Detects bad fruits in ROI using HSV green masking.
        Computes depth, assigns stable IDs, publishes TF + PointStamped.

        Input Arguments:
        ---
        `frame` : [numpy.ndarray]
            Latest RGB frame

        Returns:
        ---
        None
        '''
        roi = frame[self.box_y1:self.box_y2, self.box_x1:self.box_x2]
        if roi is None or roi.size == 0:
            return

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Strict HSV mask for green
        lower = np.array([35, 60, 60])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 5)

        # Find contours
        cnt_data = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnt_data[-2]
        if not cnts:
            return

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        fruit_positions = []

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # ---------------- LOOP OVER CONTOURS ----------------
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 320:
                continue

            x, y, w, h = cv2.boundingRect(c)

            # Convert to full-frame coordinates
            xF = x + self.box_x1
            yF = y + self.box_y1

            # Purple/Violet rejection test (good fruits)
            body_region = roi[int(y + h * 0.45): y + h, x: x + w]

            purple_reject = False
            if body_region.size > 0:
                hsv_body = cv2.cvtColor(body_region, cv2.COLOR_BGR2HSV)
                H = hsv_body[:, :, 0]
                if ((H > 130) & (H < 170)).mean() > 0.15:
                    purple_reject = True

            if purple_reject:
                continue

            # Depth extraction
            EXP = 12
            x0 = max(0, xF - EXP)
            y0 = max(0, yF - EXP)
            x1 = min(frame.shape[1] - 1, xF + w + EXP)
            y1 = min(frame.shape[0] - 1, yF + h + EXP)

            depth_patch = self.depth[y0:y1, x0:x1]
            ok = np.isfinite(depth_patch) & (depth_patch > 0)
            if ok.sum() == 0:
                continue

            d = self._depth_to_m(np.median(depth_patch[ok]))
            if d is None:
                continue

            # Compute centroid (image)
            M = cv2.moments(c)
            cx_r = int(M['m10'] / M['m00']) + self.box_x1
            cy_r = int(M['m01'] / M['m00']) + self.box_y1

            # 3D coordinate in camera frame
            X = d
            Y = -(cx_r - cx) * d / fx
            Z = -(cy_r - cy) * d / fy

            fruit_positions.append((c, np.array([X, Y, Z]), (x0, y0, x1, y1), (cx_r, cy_r)))

        # ---------------- STABLE ID ASSIGNMENT ----------------
        assigned = {i: None for i in range(1, MAX_BAD_FRUITS + 1)}

        for fruit in fruit_positions:
            xyz = fruit[1]
            best_id, best_dist = None, float('inf')

            for fid in range(1, MAX_BAD_FRUITS + 1):
                prev = self.prev_fruit_positions[fid]
                if prev is None:
                    best_id = fid
                    break
                dist = np.linalg.norm(xyz - prev)
                if dist < best_dist:
                    best_id = fid
                    best_dist = dist

            assigned[best_id] = fruit
            self.prev_fruit_positions[best_id] = xyz

        # ---------------- PUBLISH TF + POINTS ----------------
        for fid in range(1, MAX_BAD_FRUITS + 1):
            item = assigned[fid]
            if item is None:
                continue

            cnt, xyz_cam, bbox, centroid = item
            cxp, cyp = centroid

            # Draw fruit on image
            cv2.rectangle(frame, bbox[:2], bbox[2:], (0, 0, 255), 2)
            cv2.putText(frame, f"bad_fruit_{fid}", (bbox[0], bbox[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(frame, (cxp, cyp), 5, (255, 0, 0), -1)

            # Depth at centroid
            d_c = self._depth_patch(cxp, cyp)
            if d_c is None:
                continue

            X = d_c
            Y = -(cxp - cx) * d_c / fx
            Z = -(cyp - cy) * d_c / fy
            P_cam = np.array([X, Y, Z])

            # Transform to base_link
            try:
                tf = self.tf_buffer.lookup_transform("base_link", self.camera_frame, rclpy.time.Time())
                R = tf_transformations.quaternion_matrix([
                    tf.transform.rotation.x,
                    tf.transform.rotation.y,
                    tf.transform.rotation.z,
                    tf.transform.rotation.w])[:3, :3]
                T = np.array([
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z])
                P_base = R @ P_cam + T
            except Exception:
                P_base = self.cam_rot @ P_cam + self.cam_translation

            # Publish fruit TF
            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = "base_link"
            tf_msg.child_frame_id = f"{TEAM_PREFIX}_bad_fruit_{fid}"
            tf_msg.transform.translation.x = float(P_base[0])
            tf_msg.transform.translation.y = float(P_base[1])
            tf_msg.transform.translation.z = float(P_base[2])
            tf_msg.transform.rotation.w = 1.0
            self.tf_pub.sendTransform(tf_msg)

            # Publish centroid point
            pmsg = PointStamped()
            pmsg.header.stamp = self.get_clock().now().to_msg()
            pmsg.header.frame_id = "base_link"
            pmsg.point.x = float(P_base[0])
            pmsg.point.y = float(P_base[1])
            pmsg.point.z = float(P_base[2])
            self.point_pub.publish(pmsg)

    # ============================================================================
    #                                 DESTROY
    # ============================================================================

    def destroy_node(self):
        '''
        Purpose:
        ---
        Destroys OpenCV windows safely during node shutdown.
        '''
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


# ============================================================================
#                                   MAIN
# ============================================================================

def main():
    '''
    Purpose:
    ---
    Initializes ROS and starts the Task3ADetection node.

    Example call:
    ---
    Called automatically when the script is executed.
    '''
    rclpy.init()
    node = Task3ADetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
