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

CAN_MARKER_ID = 3
TEAM_PREFIX = "4784"
MAX_BAD_FRUITS = 2

class ArucoCenterFixedOri(Node):
    def __init__(self):
        super().__init__("aruco_center_fixed_ori_task3")
        self.bridge = CvBridge()
        self.color = None
        self.depth = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # Parameters
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

        # Subscribers
        self.create_subscription(RosImage, color_topic, self.color_cb, 5)
        self.create_subscription(RosImage, depth_topic, self.depth_cb, 5)
        self.create_subscription(CameraInfo, caminfo_topic, self.caminfo_cb, 5)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_pub = TransformBroadcaster(self)

        # Publishers
        self.point_pub = self.create_publisher(PointStamped, '/bad_fruit_points', 10)
        self.img_pub = self.create_publisher(RosImage, '/task3/annotated_image', 6)

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters_create()
        self.marker_length = 0.03  # 3 cm marker size

        # ROI for fruit detection
        self.box_x1, self.box_y1 = 5, 210
        self.box_x2, self.box_y2 = 380, 400

        # Fallback camera transform
        self.cam_translation = np.array([-1.08, 0.007, 1.09])
        roll, pitch, yaw = 0.0, 0.7330383, 0.0
        self.cam_rot = (
            self._rot_z(yaw) @ self._rot_y(pitch) @ self._rot_x(roll)
        )

        self.get_logger().info("Task3 node ready — REAL ORIENTATION ENABLED")

    # ----------------------------------------------------
    # Rotation helpers
    # ----------------------------------------------------
    def _rot_x(self, a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])

    def _rot_y(self, a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])

    def _rot_z(self, a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    # ----------------------------------------------------
    # Callbacks
    # ----------------------------------------------------
    def caminfo_cb(self, msg):
        self.camera_matrix = np.array(msg.k).reshape((3,3))
        self.dist_coeffs = np.array(msg.d)

    def color_cb(self, msg):
        try:
            self.color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            self.color = None
        self.process()

    def depth_cb(self, msg):
        try:
            self.depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except:
            self.depth = None

    # ----------------------------------------------------
    # Depth utility
    # ----------------------------------------------------
    def _depth_patch(self, u, v, half=4):
        if self.depth is None:
            return None
        h, w = self.depth.shape
        u0, u1 = max(0,u-half), min(w-1,u+half)
        v0, v1 = max(0,v-half), min(h-1,v+half)
        patch = self.depth[v0:v1+1, u0:u1+1]
        valid = (patch > 0) & np.isfinite(patch)
        return float(np.median(patch[valid])) if valid.any() else None

    # ----------------------------------------------------
    # MAIN PROCESS
    # ----------------------------------------------------
    def process(self):
        if self.color is None or self.depth is None or self.camera_matrix is None:
            return

        img = self.color.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

        if ids is not None:
            # Estimate pose of each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )

            for i, marker_id_arr in enumerate(ids):
                marker_id = int(marker_id_arr[0])

                # Pixel center
                pts = corners[i][0]
                u = int(np.mean(pts[:,0]))
                v = int(np.mean(pts[:,1]))

                depth = self._depth_patch(u, v)
                if depth is None:
                    continue

                # Convert rvec/tvec → rotation matrix
                R_cam_marker, _ = cv2.Rodrigues(rvecs[i])
                q_cam_marker = tf_transformations.quaternion_from_matrix(
                    np.vstack((np.hstack((R_cam_marker, [[0],[0],[0]])), [0,0,0,1]))
                )

                # tvec = marker position in camera frame
                P_cam = tvecs[i][0]

                # Transform to base_link
                try:
                    tf = self.tf_buffer.lookup_transform("base_link", self.camera_frame, rclpy.time.Time())
                    T = tf.transform.translation
                    Rq = [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
                    R = tf_transformations.quaternion_matrix(Rq)[:3,:3]
                    P_base = R @ P_cam + np.array([T.x, T.y, T.z])
                    q_base = tf_transformations.quaternion_multiply(Rq, q_cam_marker)
                except:
                    # fallback
                    P_base = self.cam_rot @ P_cam + self.cam_translation
                    q_base = q_cam_marker

                # ---------- CAN MARKER ONLY ----------
                if marker_id == CAN_MARKER_ID:
                    msg = TransformStamped()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = "base_link"
                    msg.child_frame_id = f"{TEAM_PREFIX}_fertilizer_1"

                    msg.transform.translation.x = float(P_base[0])
                    msg.transform.translation.y = float(P_base[1])
                    msg.transform.translation.z = float(P_base[2])

                    # REAL ORIENTATION
                    msg.transform.rotation.x = q_base[0]
                    msg.transform.rotation.y = q_base[1]
                    msg.transform.rotation.z = q_base[2]
                    msg.transform.rotation.w = q_base[3]

                    self.tf_pub.sendTransform(msg)

                # Draw ID
                cv2.circle(img, (u,v), 6, (0,255,0), -1)
                cv2.putText(img, f"ID {marker_id}", (u+5,v-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Bad fruits
        self._process_bad_fruits(img)

        # ROI box
        cv2.rectangle(img, (self.box_x1,self.box_y1),
                      (self.box_x2,self.box_y2), (255,0,0), 2)

        # Publish annotated
        try:
            ros_img = self.bridge.cv2_to_imgmsg(img, "bgr8")
            self.img_pub.publish(ros_img)
        except:
            pass

        if self.show_gui:
            cv2.imshow("Task3 annotated", img)
            cv2.waitKey(1)

    # ----------------------------------------------------
    # BAD FRUIT DETECTION
    # ----------------------------------------------------
    def _process_bad_fruits(self, frame):
        roi = frame[self.box_y1:self.box_y2, self.box_x1:self.box_x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, np.array([0,0,70]), np.array([180,40,200]))
        mask = cv2.medianBlur(mask, 5)
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bad_count = 1
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            if bad_count > MAX_BAD_FRUITS:
                break
            if cv2.contourArea(c) < 300:
                continue

            x,y,w,h = cv2.boundingRect(c)
            x_global = x + self.box_x1
            y_global = y + self.box_y1

            # Draw
            cv2.rectangle(frame, (x_global,y_global), (x_global+w,y_global+h), (0,0,255), 2)

            cx = x_global + w//2
            cy = y_global + h//2
            cv2.circle(frame, (cx,cy), 5, (0,255,0), -1)

            depth = self._depth_patch(cx,cy)
            if depth is None:
                continue

            fx = self.camera_matrix[0,0]
            fy = self.camera_matrix[1,1]
            cx0 = self.camera_matrix[0,2]
            cy0 = self.camera_matrix[1,2]

            X_cam = depth
            Y_cam = -(cx - cx0) * depth / fx
            Z_cam = -(cy - cy0) * depth / fy
            P_cam = np.array([X_cam,Y_cam,Z_cam])

            try:
                tf = self.tf_buffer.lookup_transform("base_link", self.camera_frame, rclpy.time.Time())
                T = tf.transform.translation
                Rq = [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
                R = tf_transformations.quaternion_matrix(Rq)[:3,:3]
                P_base = R @ P_cam + np.array([T.x,T.y,T.z])
            except:
                P_base = self.cam_rot @ P_cam + self.cam_translation

            # Publish TF
            msg = TransformStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.child_frame_id = f"{TEAM_PREFIX}_bad_fruit_{bad_count}"
            msg.transform.translation.x = float(P_base[0])
            msg.transform.translation.y = float(P_base[1])
            msg.transform.translation.z = float(P_base[2])
            msg.transform.rotation.w = 1.0
            self.tf_pub.sendTransform(msg)

            bad_count += 1


def main():
    rclpy.init()
    node = ArucoCenterFixedOri()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

