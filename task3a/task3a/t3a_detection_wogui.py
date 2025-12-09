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

# ---------- CONFIG ----------
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

        # parameters
        # show_gui default changed to False so OpenCV window won't pop up
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("caminfo_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("show_gui", False)

        color_topic = self.get_parameter("color_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        caminfo_topic = self.get_parameter("caminfo_topic").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.show_gui = self.get_parameter("show_gui").get_parameter_value().bool_value

        # subscribers
        self.create_subscription(RosImage, color_topic, self.color_cb, 5)
        self.create_subscription(RosImage, depth_topic, self.depth_cb, 5)
        self.create_subscription(CameraInfo, caminfo_topic, self.caminfo_cb, 5)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_pub = TransformBroadcaster(self)

        # publishers
        self.point_pub = self.create_publisher(PointStamped, '/bad_fruit_points', 10)
        self.img_pub = self.create_publisher(RosImage, '/task3/annotated_image', 6)

        # ---------------- ArUco Setup (works for all OpenCV versions) ----------------
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        except Exception:
            self.get_logger().warn("Failed to get predefined dictionary for ArUco.")

        try:
            # Newer OpenCV
            self.params = cv2.aruco.DetectorParameters()
        except AttributeError:
            # Older OpenCV
            self.params = cv2.aruco.DetectorParameters_create()

        # marker orientation - normalized quaternion
        q = np.array([0.707, 0.028, 0.034, 0.707])
        self.q_marker3 = (q / np.linalg.norm(q)).tolist()

        # ROI (tweak to your setup)
        self.box_x1, self.box_y1 = 60, 430
        self.box_x2, self.box_y2 = 375, 600

        # fallback camera-to-base transform (if TF lookup fails)
        self.cam_translation = np.array([-1.08, 0.007, 1.09])
        roll = 0.0
        pitch = 0.7330383
        yaw = 0.0
        self.cam_rot = self._rot_z(yaw) @ self._rot_y(pitch) @ self._rot_x(roll)

        # Fruit ID stability (remember previous 3D centroid)
        self.prev_fruit_positions = {i: None for i in range(1, MAX_BAD_FRUITS+1)}

        self.get_logger().info("Task3 node started — centroid TF only (GUI disabled)")

    # ----------------- Rotation Helpers -----------------
    def _rot_x(self, a):
        ca, sa = math.cos(a), math.sin(a)
        return np.array([
            [1, 0, 0],
            [0, ca, -sa],
            [0, sa, ca]
        ])

    def _rot_y(self, a):
        ca, sa = math.cos(a), math.sin(a)
        return np.array([
            [ca, 0, sa],
            [0, 1, 0],
            [-sa, 0, ca]
        ])

    def _rot_z(self, a):
        ca, sa = math.cos(a), math.sin(a)
        return np.array([
            [ca, -sa, 0],
            [sa, ca, 0],
            [0, 0, 1]
        ])

    # ---------------- Callbacks ----------------
    def caminfo_cb(self, msg):
        try:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
        except Exception as e:
            self.get_logger().error(f"caminfo parse error: {e}")

    def color_cb(self, msg):
        try:
            self.color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"color cv bridge error: {e}")
            return
        self.process()

    def depth_cb(self, msg):
        try:
            # prefer 32FC1 if available, fallback to whatever
            self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        except Exception:
            try:
                self.depth = self.bridge.imgmsg_to_cv2(msg)
            except Exception as e:
                self.get_logger().error(f"depth cv bridge error: {e}")
                self.depth = None

    # ---------------- Depth Fix (mm → m) ----------------
    def _depth_to_m(self, d):
        if d is None:
            return None
        try:
            if not np.isfinite(d) or d <= 0:
                return None
        except Exception:
            pass
        d = float(d)
        # Some depth streams are in mm (>10), others in meters (<=10)
        if d > 10.0:
            return d / 1000.0
        return d

    def _depth_patch(self, u, v):
        if self.depth is None:
            return None
        h, w = self.depth.shape
        if u < 0 or v < 0 or u >= w or v >= h:
            return None
        u0, u1 = max(0, u-4), min(w-1, u+4)
        v0, v1 = max(0, v-4), min(h-1, v+4)
        patch = self.depth[v0:v1+1, u0:u1+1]
        ok = np.isfinite(patch) & (patch > 0)
        if ok.sum() == 0:
            return None
        return self._depth_to_m(np.median(patch[ok]))

    # ---------------- Main Loop ----------------
    def process(self):
        if self.color is None or self.depth is None or self.camera_matrix is None:
            return

        img = self.color.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ---------------- ArUco detection ----------------
        try:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
        except Exception:
            # fallback (older OpenCV signature)
            res = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
            if isinstance(res, tuple) and len(res) >= 2:
                corners, ids = res[0], res[1]
            else:
                corners, ids = None, None

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # ---------------- ArUco Handling ----------------
        if ids is not None:
            for i, crn in enumerate(corners):
                ID = int(ids[i][0])
                pts = crn.reshape(4, 2).astype(int)
                u = int(np.mean(pts[:, 0]))
                v = int(np.mean(pts[:, 1]))

                d = self._depth_patch(u, v)
                if d is None:
                    continue

                X = d
                Y = -(u - cx) * d / fx
                Z = -(v - cy) * d / fy
                P_cam = np.array([X, Y, Z])

                # base transform
                try:
                    t = self.tf_buffer.lookup_transform("base_link", self.camera_frame, rclpy.time.Time())
                    R = tf_transformations.quaternion_matrix([
                        t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w])[:3, :3]
                    T = np.array([
                        t.transform.translation.x,
                        t.transform.translation.y,
                        t.transform.translation.z])
                    P_base = R @ P_cam + T
                except Exception:
                    P_base = self.cam_rot @ P_cam + self.cam_translation

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
                    self.get_logger().info(f"Published fertilizer TF at {P_base}")

        # ---------------- Fruit detection (improved) ----------------
        self._process_fruits(img)

        # ROI box (visual)
        cv2.rectangle(img, (self.box_x1, self.box_y1), (self.box_x2, self.box_y2), (255, 0, 0), 2)

        # publish annotated image
        try:
            out = self.bridge.cv2_to_imgmsg(img, "bgr8")
            out.header.frame_id = self.camera_frame
            self.img_pub.publish(out)
        except Exception as e:
            self.get_logger().error(f"failed to publish annotated image: {e}")

        # GUI disabled by default; safe guarded
        if self.show_gui:
            try:
                cv2.imshow("Task3 Fruits", img)
                cv2.waitKey(1)
            except Exception:
                pass

    # ---------------- Improved Fruit Processing ----------------
    def _process_fruits(self, frame):

        roi = frame[self.box_y1:self.box_y2, self.box_x1:self.box_x2]
        if roi is None or roi.size == 0:
            return
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # ---------------- STRICT GREEN MASK ----------------
        lower = np.array([35, 60, 60], dtype=np.uint8)
        upper = np.array([85, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 5)

        # find contours
        cnts_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts_info) == 3:
            _, cnts, _ = cnts_info
        else:
            cnts, _ = cnts_info

        if not cnts:
            return

        cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        fruit_positions = []

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 320:   # filter out small noisy blobs; tune if needed
                continue

            x, y, w, h = cv2.boundingRect(c)            # coords relative to ROI
            xF = x + self.box_x1                          # full-image coords
            yF = y + self.box_y1

            # BODY COLOR CHECK → reject purple/violet fruit (good fruit)
            body_y0 = int(y + h * 0.45)
            body_y1 = y + h
            body_x0 = x
            body_x1 = x + w
            body_region = roi[body_y0:body_y1, body_x0:body_x1]

            purple_reject = False
            if body_region is not None and body_region.size > 0:
                try:
                    hsv_body = cv2.cvtColor(body_region, cv2.COLOR_BGR2HSV)
                    H = hsv_body[:, :, 0].astype(np.int32)
                    purple_mask = (H > 130) & (H < 170)
                    if purple_mask.mean() > 0.15:
                        purple_reject = True
                except Exception:
                    purple_reject = False

            if purple_reject:
                continue

            # DEPTH CALCULATION (use expanded bounding area)
            EXP = 12
            x0 = max(0, xF - EXP)
            y0 = max(0, yF - EXP)
            x1 = min(self.depth.shape[1]-1, xF + w + EXP)
            y1 = min(self.depth.shape[0]-1, yF + h + EXP)

            dpatch = self.depth[y0:y1, x0:x1]
            if dpatch is None or dpatch.size == 0:
                continue
            ok = np.isfinite(dpatch) & (dpatch > 0)
            if ok.sum() == 0:
                continue

            d = self._depth_to_m(np.median(dpatch[ok]))
            if d is None:
                continue

            cxp = xF + w // 2
            cyp = yF + h // 2

            X = d
            Y = -(cxp - cx) * d / fx
            Z = -(cyp - cy) * d / fy

            # compute contour centroid (moment) relative to ROI, convert to full-image pixel coords
            M = cv2.moments(c)
            if M.get('m00', 0) != 0:
                cx_contour = int(M['m10'] / M['m00'])
                cy_contour = int(M['m01'] / M['m00'])
                centroid_x = cx_contour + self.box_x1
                centroid_y = cy_contour + self.box_y1
            else:
                centroid_x = cxp
                centroid_y = cyp

            fruit_positions.append((c, np.array([X, Y, Z]), (x0, y0, x1, y1), (centroid_x, centroid_y)))

        # ---------- Stable ID assignment (nearest previous) ----------
        assigned = {i: None for i in range(1, MAX_BAD_FRUITS+1)}

        for fruit in fruit_positions:
            xyz = fruit[1]
            best_id = None
            best_dist = float('inf')

            for fid in range(1, MAX_BAD_FRUITS+1):
                prev = self.prev_fruit_positions.get(fid)
                if prev is None:
                    best_id = fid
                    break
                dist = np.linalg.norm(xyz - prev)
                if dist < best_dist:
                    best_dist = dist
                    best_id = fid

            assigned[best_id] = fruit
            self.prev_fruit_positions[best_id] = xyz

        # ---------- Publish & Draw (centroid TF & PointStamped) ----------
        for fid in range(1, MAX_BAD_FRUITS+1):
            item = assigned.get(fid)
            if item is None:
                continue

            cnt, xyz, bbox, centroid = item
            x0, y0, x1, y1 = bbox
            centroid_x, centroid_y = centroid

            # draw on full frame (visual only; image still published)
            try:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.putText(frame, f"bad_fruit_{fid}", (x0, y0 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
            except Exception:
                pass

            # depth at centroid
            depth_centroid = self._depth_patch(centroid_x, centroid_y)
            if depth_centroid is None:
                self.get_logger().debug(f"bad_fruit_{fid}: centroid depth invalid at ({centroid_x},{centroid_y})")
                continue

            Xc = depth_centroid
            Yc = -(centroid_x - self.camera_matrix[0, 2]) * depth_centroid / self.camera_matrix[0, 0]
            Zc = -(centroid_y - self.camera_matrix[1, 2]) * depth_centroid / self.camera_matrix[1, 1]
            P_cam_centroid = np.array([Xc, Yc, Zc])

            # transform to base_link (try TF lookup, fallback)
            try:
                t = self.tf_buffer.lookup_transform("base_link", self.camera_frame, rclpy.time.Time())
                R = tf_transformations.quaternion_matrix([
                    t.transform.rotation.x, t.transform.rotation.y,
                    t.transform.rotation.z, t.transform.rotation.w])[:3, :3]
                T = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
                P_base_centroid = R @ P_cam_centroid + T
            except Exception:
                P_base_centroid = self.cam_rot @ P_cam_centroid + self.cam_translation

            # publish centroid TF
            center_tf = TransformStamped()
            center_tf.header.stamp = self.get_clock().now().to_msg()
            center_tf.header.frame_id = "base_link"
            center_tf.child_frame_id = f"{TEAM_PREFIX}_bad_fruit_{fid}"
            center_tf.transform.translation.x = float(P_base_centroid[0])
            center_tf.transform.translation.y = float(P_base_centroid[1])
            center_tf.transform.translation.z = float(P_base_centroid[2])
            center_tf.transform.rotation.w = 1.0
            self.tf_pub.sendTransform(center_tf)

            # publish PointStamped for centroid as well
            pmsg = PointStamped()
            pmsg.header.stamp = self.get_clock().now().to_msg()
            pmsg.header.frame_id = "base_link"
            pmsg.point.x = float(P_base_centroid[0])
            pmsg.point.y = float(P_base_centroid[1])
            pmsg.point.z = float(P_base_centroid[2])
            self.point_pub.publish(pmsg)

            self.get_logger().info(f"Published bad_fruit_{fid}_center TF at {P_base_centroid}")

    # ----------------------------------------------------
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

