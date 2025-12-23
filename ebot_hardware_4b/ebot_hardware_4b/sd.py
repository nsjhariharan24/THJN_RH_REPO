#!/usr/bin/env python3
"""

# Team ID:          4784
# Theme:            Krishi coBot
# Author List:      Hariharan S, Thilakraj S, Jeevan Uday Alexander, Niranjan R
# Filename:         shape_detector_task3b.py

"""

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
import numpy as np
import math
import cv2
from typing import List, Tuple, Optional


class PlantIDManager:
    def __init__(self):
        self.left_x = -2.395
        self.right_x = -0.522
        self.y_levels = [-4.0852, -2.7458, -1.3886, -0.041]
    def assign(self, robot_x: float, robot_y: float,yaw, fl) -> int:
        # -------- lane selection (FIXED) --------
        if fl == 1:
            if robot_x < (self.left_x + self.right_x) / 2.0:
                lane_offset = 4     # LEFT lane → plants 1–4
            else:
                lane_offset = 0     # RIGHT lane → plants 5–8
            idx = min(
                range(len(self.y_levels)),
                key=lambda i: abs(robot_y - self.y_levels[i])
            )
            return lane_offset + idx + 1
        # -------- lane selection (DYNAMIC) --------  
        elif yaw > 0 and fl == 0:
            if robot_x < (self.left_x + self.right_x) / 2.0:
                lane_offset = 0     # LEFT lane → plants 1–4
            else:
                lane_offset = 4     # RIGHT lane → plants 5–8
            idx = min(
                range(len(self.y_levels)),
                key=lambda i: abs(robot_y - self.y_levels[i])
            )
            return lane_offset + idx + 1
        elif yaw < 0 and fl == 0 :
            if robot_x < (self.left_x + self.right_x) / 2.0:
                lane_offset = 4     # LEFT lane → plants 1–4
            else:
                lane_offset = 0     # RIGHT lane → plants 5–8
            idx = min(
                range(len(self.y_levels)),
                key=lambda i: abs(robot_y - self.y_levels[i])
            )
            return lane_offset + idx + 1

            
# ---------------- Shared math helpers ----------------
def polar_to_xy(ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return np.vstack((x, y)).T

def fit_line_two_points(p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float, float]:
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    a = dy; b = -dx
    n = math.hypot(a, b)
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    a /= n; b /= n
    c = -(a*p1[0] + b*p1[1])
    return a, b, c

def intersection(l1, l2) -> Optional[np.ndarray]:
    a1,b1,c1 = l1; a2,b2,c2 = l2
    det = a1*b2 - a2*b1
    if abs(det) < 1e-9:
        return None
    x = (b1*c2 - b2*c1)/det
    y = (a2*c1 - a1*c2)/det
    return np.array([x, y])

def seg_contains_point(A: np.ndarray, B: np.ndarray, pt: np.ndarray, tol=0.06) -> bool:
    AB = B - A; AP = pt - A
    denom = np.dot(AB, AB)
    if denom < 1e-9: return False
    t = np.dot(AP, AB) / denom
    return (-tol <= t) and (t <= 1.0 + tol)

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = v1 / (np.linalg.norm(v1) + 1e-9)
    n2 = v2 / (np.linalg.norm(v2) + 1e-9)
    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return math.degrees(math.acos(dot))

def extend_segment(A: np.ndarray, B: np.ndarray, L: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    v = B - A
    length = np.linalg.norm(v)
    if length < 1e-6:
        return A.copy(), B.copy()
    u = v / length
    mid = (A + B) / 2.0
    return mid - u*(L/2.0), mid + u*(L/2.0)

# split-and-merge + merge_colinear
def split_and_merge(points: np.ndarray, max_dist: float = 0.04, min_pts: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    def recurse(i0: int, i1: int):
        if i1 - i0 + 1 < min_pts:
            segments.append((points[i0].copy(), points[i1].copy()))
            return
        pA = points[i0]; pB = points[i1]
        a,b,c = fit_line_two_points(pA, pB)
        subset = points[i0:i1+1]
        dists = np.abs(a*subset[:,0] + b*subset[:,1] + c)
        idx = int(np.argmax(dists))
        if dists[idx] > max_dist:
            mid = i0 + idx
            if mid == i0 or mid == i1:
                segments.append((pA.copy(), pB.copy()))
                return
            recurse(i0, mid)
            recurse(mid, i1)
        else:
            segments.append((pA.copy(), pB.copy()))
            return
    if points.shape[0] >= 2:
        recurse(0, points.shape[0]-1)
    return segments

def merge_colinear(segments: List[Tuple[np.ndarray, np.ndarray]], angle_thresh=12.0, endpoint_merge=0.20):
    segs = [ (A.copy(), B.copy()) for (A,B) in segments ]
    used = [False]*len(segs)
    merged = []
    for i in range(len(segs)):
        if used[i]: continue
        A1,B1 = segs[i]
        mergedA = A1.copy(); mergedB = B1.copy()
        for j in range(i+1, len(segs)):
            if used[j]: continue
            A2,B2 = segs[j]
            ang = angle_between(B1 - A1, B2 - A2)
            if abs(ang) > angle_thresh: continue
            dlist = [ np.linalg.norm(mergedA - A2), np.linalg.norm(mergedA - B2),
                      np.linalg.norm(mergedB - A2), np.linalg.norm(mergedB - B2) ]
            if min(dlist) > endpoint_merge: continue
            pts = np.vstack([mergedA, mergedB, A2, B2])
            dirv = mergedB - mergedA
            if np.linalg.norm(dirv) < 1e-6: continue
            u = dirv / np.linalg.norm(dirv)
            projs = pts.dot(u)
            mergedA = pts[np.argmin(projs)].copy()
            mergedB = pts[np.argmax(projs)].copy()
            used[j] = True
        used[i] = True
        merged.append((mergedA, mergedB))
    return merged

# ---------------- SquareDetectorForceStop Node ----------------
class SquareDetectorForceStop(Node):
    def __init__(self,plant_mgr: PlantIDManager):
        super().__init__("square_detector_force_stop")

        # subs/pubs
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.shape_pub = self.create_publisher(Bool, "/shape_stop_signal", 10)
        self.status_pub = self.create_publisher(String, "/detection_status", 10)
        
        # robot pose (for global conversion)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # detection params
        self.cluster_gap = 0.20
        self.split_merge_dist = 0.06
        self.min_segment_pts = 9
        self.min_line_length = 0.09
        self.extend_len = 0.6
        self.endpoint_tol = 0.06
        self.angle_tol_deg = 3.0
        self.side_sim_ratio = 0.30
        self.perp_dot_tol = math.sin(math.radians(self.angle_tol_deg)) + 1e-6
        self.triplet_centroid_maxdist = 0.45

        # control params
        self.max_detect_distance = 0.5
        self.delay_seconds = 10.0
        self.force_stop_duration = 3.0
        self.force_stop_rate = 10.0
        self.post_publish_cooldown = 2.5

        # dedup / plant tracking
        self.max_unique_squares = 2
        #self.seen_plants: List[Tuple[float,float,int]] = []  # (gx, gy, plant_id)
        self.dedup_radius = 0.6
        #self.accepted_stop_count = 0
        #self.next_plant_id = 1
        self.plant_mgr = plant_mgr
        self.accepted_stop_count = 0


        # pending detection stored at timer-start (gx,gy,rx,ry)
        self.pending_detection: Optional[Tuple[float,float,float,float]] = None

        # last status to avoid NO_SHAPE spam
        self.last_status: Optional[str] = None

        # visualization params
        self.vis_size = 700
        self.scale = 150.0
        self.offset = np.array([self.vis_size//2, self.vis_size//2], dtype=int)

        # timers/state
        self.square_detect_time = None
        self.last_stop_publish_time = -9999
        self.forced_stop_timer = None
        self.forced_stop_end_time = None

        self.last_print_t = -10.0
        self.print_cooldown = 1.0

        self.get_logger().info("🟩 SquareDetectorForceStop started (store pending detection at timer-start).")

        # create GUI window in main thread and allow resizing
        cv2.namedWindow("SquareDetector (force-stop)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SquareDetector (force-stop)", self.vis_size, self.vis_size)

    def odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        self.robot_yaw = math.atan2(siny, cosy)

    def scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        angles = np.linspace(msg.angle_min, msg.angle_max, ranges.shape[0], endpoint=True)
        ranges = np.nan_to_num(ranges, nan=msg.range_max + 1.0, posinf=msg.range_max + 1.0)
        valid = (ranges > msg.range_min) & (ranges < (msg.range_max + 1e-6))
        if np.sum(valid) < 12:
            self._publish_no_shape()
            self._visualize(np.zeros((0,2)), [], [])
            return

        idxs = np.where(valid)[0]
        r = ranges[idxs]; a = angles[idxs]
        points = polar_to_xy(r, a)

        # cluster contiguous
        clusters = []
        if points.shape[0] > 0:
            cur = [points[0]]
            for i in range(1, points.shape[0]):
                if np.linalg.norm(points[i] - points[i-1]) > self.cluster_gap:
                    clusters.append(np.array(cur)); cur = [points[i]]
                else:
                    cur.append(points[i])
            if len(cur) > 0:
                clusters.append(np.array(cur))

        # split-and-merge -> raw_segments
        raw_segments = []
        for cl in clusters:
            if cl.shape[0] < self.min_segment_pts:
                continue
            segs = split_and_merge(cl, max_dist=self.split_merge_dist, min_pts=self.min_segment_pts)
            for (A,B) in segs:
                L = np.linalg.norm(B - A)
                if L < self.min_line_length:
                    continue
                mean_r = (np.linalg.norm(A) + np.linalg.norm(B)) / 2.0
                if not (0.05 < mean_r < 3.0):
                    continue
                raw_segments.append((A.copy(), B.copy()))

        if len(raw_segments) < 2:
            self._publish_no_shape()
            self._visualize(points, [], [])
            return

        # merge colinear
        merged = merge_colinear(raw_segments, angle_thresh=12.0, endpoint_merge=0.20)

        # extend segments
        ext_segments = []
        for (A,B) in merged:
            Ae, Be = extend_segment(A, B, self.extend_len)
            ext_segments.append((Ae, Be, A, B))

        # candidate pairs
        candidate_pairs = []
        for i in range(len(ext_segments)):
            for j in range(i+1, len(ext_segments)):
                Ae, Be, Ao, Bo = ext_segments[i]
                Ce, De, Co, Do = ext_segments[j]
                mid1 = (Ae + Be)/2.0; mid2 = (Ce + De)/2.0
                if np.linalg.norm(mid1 - mid2) > 1.2:
                    continue
                candidate_pairs.append((i,j))

        detected = []
        for (i,j) in candidate_pairs:
            Ae, Be, Ao, Bo = ext_segments[i]
            Ce, De, Co, Do = ext_segments[j]

            l1 = fit_line_two_points(Ao, Bo); l2 = fit_line_two_points(Co, Do)
            inter = intersection(l1, l2)

            corner_point = None; intersection_valid = False

            # Rule A: intersection on both original segments
            if inter is not None and seg_contains_point(Ao, Bo, inter, tol=0.06) and seg_contains_point(Co, Do, inter, tol=0.06):
                intersection_valid = True; corner_point = inter

            # Rule B: endpoint adjacency
            if not intersection_valid:
                endpoints1 = [Ao, Bo]; endpoints2 = [Co, Do]
                min_d = 1e9; best_pair = None
                for e1 in endpoints1:
                    for e2 in endpoints2:
                        d = np.linalg.norm(e1 - e2)
                        if d < min_d:
                            min_d = d; best_pair = (e1, e2)
                if min_d <= self.endpoint_tol:
                    e1,e2 = best_pair
                    corner_point = (e1 + e2)/2.0
                    intersection_valid = True

            if not intersection_valid:
                continue

            v1 = Bo - Ao; v2 = Do - Co
            raw_ang = angle_between(v1, v2)
            interior = min(raw_ang, 180.0 - raw_ang)

            dA1 = np.linalg.norm(Ao - corner_point); dB1 = np.linalg.norm(Bo - corner_point)
            len1 = min(dA1, dB1)
            dA2 = np.linalg.norm(Co - corner_point); dB2 = np.linalg.norm(Do - corner_point)
            len2 = min(dA2, dB2)

            if len1 < self.min_line_length or len2 < self.min_line_length:
                continue

            if abs(interior - 90.0) > self.angle_tol_deg:
                continue
            mean_len = (len1 + len2)/2.0
            if abs(len1 - len2) / (mean_len + 1e-9) > self.side_sim_ratio:
                continue
            u1 = v1 / (np.linalg.norm(v1) + 1e-9); u2 = v2 / (np.linalg.norm(v2) + 1e-9)
            if abs(np.dot(u1, u2)) > self.perp_dot_tol:
                continue

            detected.append({
                'corner': corner_point,
                'angle': interior,
                'len1': len1,
                'len2': len2,
                'i': i, 'j': j
            })

        # triplet rule unchanged
        seg_info = []
        for (A,B) in merged:
            centroid = (A + B)/2.0
            bearing = math.atan2(centroid[1], centroid[0])
            v = B - A
            seg_info.append({'A':A,'B':B,'centroid':centroid,'bearing':bearing,'v':v})
        seg_info.sort(key=lambda s: s['bearing'])

        triplet_detected = []
        for k in range(len(seg_info)-2):
            s1 = seg_info[k]; s2 = seg_info[k+1]; s3 = seg_info[k+2]
            v1 = s1['v']; v2 = s2['v']; v3 = s3['v']
            ang12 = angle_between(v1, v2); interior12 = min(ang12, 180.0 - ang12)
            ang23 = angle_between(v2, v3); interior23 = min(ang23, 180.0 - ang23)
            if (abs(interior12 - 90.0) <= self.angle_tol_deg) and (abs(interior23 - 90.0) <= self.angle_tol_deg):
                c1 = s1['centroid']; c2 = s2['centroid']; c3 = s3['centroid']
                if (np.linalg.norm(c1 - c2) <= self.triplet_centroid_maxdist and np.linalg.norm(c2 - c3) <= self.triplet_centroid_maxdist):
                    corner1 = (c1 + c2)/2.0
                    corner2 = (c2 + c3)/2.0
                    len1 = min(np.linalg.norm(s1['A'] - corner1), np.linalg.norm(s1['B'] - corner1))
                    len2 = min(np.linalg.norm(s2['A'] - corner1), np.linalg.norm(s2['B'] - corner1))
                    len3 = min(np.linalg.norm(s3['A'] - corner2), np.linalg.norm(s3['B'] - corner2))
                    if len1 >= self.min_line_length and len2 >= self.min_line_length and len3 >= self.min_line_length:
                        triplet_detected.append({
                            'corner1': corner1,
                            'corner2': corner2,
                            'angles': (interior12, interior23),
                            'indices': (k, k+1, k+2)
                        })

        # pick final detection (prefer pair)
        final_squares = []
        if len(detected) > 0:
            used = [False]*len(detected)
            for i in range(len(detected)):
                if used[i]: continue
                cluster = [detected[i]]; used[i]=True
                for j in range(i+1, len(detected)):
                    if used[j]: continue
                    if np.linalg.norm(detected[j]['corner'] - detected[i]['corner']) < 0.06:
                        cluster.append(detected[j]); used[j]=True
                best = None; best_score = 1e9
                for c in cluster:
                    score = abs(c['angle'] - 90.0) + abs(c['len1'] - c['len2'])
                    if score < best_score:
                        best_score = score; best = c
                if best is not None:
                    final_squares.append(best)
        elif len(triplet_detected) > 0:
            best = None; best_score = 1e9
            for t in triplet_detected:
                a1,a2 = t['angles']; score = abs(a1 - 90.0) + abs(a2 - 90.0)
                if score < best_score:
                    best_score = score; best = t
            if best is not None:
                final_squares.append({'corner': best['corner1'], 'angle': (best['angles'][0] + best['angles'][1])/2.0, 'triplet': True})

        now = self.get_clock().now().seconds_nanoseconds()[0]

        accepted_for_timer = False
        new_detection_global = None
        new_detection_corner = None

        if len(final_squares) > 0:
            first_corner = final_squares[0]['corner']
            dist_to_corner = first_corner[1]

            # robot-frame -> global
            rx = 0.0; ry = float(first_corner[1])
            gx = float(self.robot_x)
            gy = self.robot_y + ry
            new_detection_global = (gx, gy)
            new_detection_corner = (rx, ry)

            if now - self.last_print_t > self.print_cooldown:
                self.last_print_t = now
                self.get_logger().info(f"SQUARE observed @ robot x={rx:.3f}, y={ry:.3f} | dist={dist_to_corner:.2f} -> global ({gx:.3f},{gy:.3f})")

            if dist_to_corner <= self.max_detect_distance:
                #self.plant_mgr.assign(gx, gy)


                if self.accepted_stop_count >= self.max_unique_squares:
                    self.get_logger().info(f"Ignoring additional square at ({gx:.3f},{gy:.3f}) — already triggered {self.accepted_stop_count} stops.")
                    accepted_for_timer = False
                else:
                    accepted_for_timer = True

        # When we accept a detection and start timer, store pending_detection (gx,gy,rx,ry)
        if accepted_for_timer and (now - self.last_stop_publish_time >= self.post_publish_cooldown):
            if self.square_detect_time is None:
                # capture pending detection coords now
                if new_detection_global is not None and new_detection_corner is not None:
                    gx, gy = new_detection_global
                    rx, ry = new_detection_corner
                    self.pending_detection = (gx, gy, rx, ry)
                else:
                    self.pending_detection = None
                self.square_detect_time = now
                self.get_logger().info(f"⏳ Square accepted for timer. Will force-stop after {self.delay_seconds:.1f}s. Stored pending_detection={self.pending_detection}")

        # When timer expires -> publish using pending_detection (if available)
        if self.square_detect_time is not None and (now - self.square_detect_time >= self.delay_seconds):
            if self.forced_stop_timer is None:
                if self.pending_detection is not None:
                    _, _, rx_report, ry_report = self.pending_detection
                else:
                    rx_report, ry_report = (0.0, 0.0)

                gx_report = float(self.robot_x)
                gy_report = float(self.robot_y)

                plant_id = self.plant_mgr.assign(gx_report, gy_report,self.robot_yaw,1)


                msg_bool = Bool(); msg_bool.data = True; self.shape_pub.publish(msg_bool)
                st = String(); st.data = f"BAD_HEALTH,{gx_report:.3f},{gy_report:.3f},{plant_id}"
                self.status_pub.publish(st)
                self.last_status = st.data

                self.accepted_stop_count += 1

                self.get_logger().warn(f"🟥 Forced STOP starting: publishing True and holding zero cmd_vel for {self.force_stop_duration:.1f}s -> {st.data}")

                interval = 1.0 / float(self.force_stop_rate)
                self.forced_stop_end_time = now + self.force_stop_duration
                self.forced_stop_timer = self.create_timer(interval, slf._stop_hold_cb)
                self.last_stop_publish_time = now

                self.pending_detection = None
                self.square_detect_time = None

        if not accepted_for_timer and self.forced_stop_timer is None:
            self._publish_no_shape()

        detections_for_vis = []
        for s in final_squares:
            detections_for_vis.append({'corner': s['corner'], 'angle': s.get('angle', 90.0)})
        self._visualize(points, merged, detections_for_vis)
    
    

    def _stop_hold_cb(self):
        now = self.get_clock().now().seconds_nanoseconds()[0]
        if now >= self.forced_stop_end_time:
            msgf = Bool()
            msgf.data = False
            self.shape_pub.publish(msgf)

            if self.last_status != "NO_SHAPE":
                st = String()
                st.data = "NO_SHAPE"
                self.status_pub.publish(st)
                self.last_status = "NO_SHAPE"

            self.get_logger().info("🟢 Square stop released")

            self.forced_stop_timer.cancel()
            self.forced_stop_timer = None


    def _publish_no_shape(self):
        if self.last_status == "NO_SHAPE":
            return
        msg = Bool(); msg.data = False; self.shape_pub.publish(msg)
        st = String(); st.data = "NO_SHAPE"; self.status_pub.publish(st)
        self.last_status = "NO_SHAPE"

    def _visualize(self, points: np.ndarray, segments: List[Tuple[np.ndarray, np.ndarray]], detections: List[dict]):
        img = np.zeros((self.vis_size, self.vis_size, 3), dtype=np.uint8)
        for gx in range(0, self.vis_size, 50):
            cv2.line(img, (gx,0), (gx,self.vis_size), (30,30,30), 1)
        for gy in range(0, self.vis_size, 50):
            cv2.line(img, (0,gy), (0+self.vis_size-0, gy), (30,30,30), 1)

        def w2i(pt):
            p = (pt * self.scale).astype(int) + self.offset
            return int(p[0]), int(p[1])

        for p in points:
            cv2.circle(img, w2i(p), 1, (0,0,255), -1)

        for (A,B) in segments:
            cv2.line(img, w2i(A), w2i(B), (0,255,0), 2)
            cv2.circle(img, w2i(A), 3, (0,200,0), -1); cv2.circle(img, w2i(B), 3, (0,200,0), -1)

        for d in detections:
            c = d['corner']; px,py = w2i(c)
            cv2.circle(img, (px,py), 6, (0,255,255), -1)
            cv2.putText(img, f"SQ {d['angle']:.0f}deg", (px+8,py-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        cv2.imshow("SquareDetector (force-stop)", img)

# ---------------- SimpleTriangleDetector Node ----------------
# ---------------- SimpleTriangleDetector Node ----------------
class SimpleTriangleDetector(Node):
    def __init__(self, plant_mgr: PlantIDManager):
        super().__init__("simple_triangle_detector")
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.shape_pub = self.create_publisher(Bool, "/shape_stop_signal", 10)
        self.status_pub = self.create_publisher(String, "/detection_status", 10)
       
        # odom
        self.robot_x = 0.0; self.robot_y = 0.0; self.robot_yaw = 0.0

        # params
        self.cluster_gap = 0.20
        self.split_merge_dist = 0.06
        self.min_segment_pts = 9
        self.min_line_length = 0.07
        self.extend_len = 0.6
        self.endpoint_tol = 0.08

        # triangle rule
        self.angle_target = 135.0
        self.angle_tol = 5.0

        # detection distance
        self.max_detect_distance = 0.9

        # timing
        self.force_stop_duration = 3.0
        self.force_stop_rate = 10.0
        self.post_publish_cooldown = 3.0

        # dedup / counters
        self.max_unique = 3
        #self.seen_plants: List[Tuple[float,float,int]] = []
        self.dedup_radius = 1
        #self.accepted_count = 0
        #self.next_plant_id = 1
        self.plant_mgr = plant_mgr
        self.accepted_count = 0


        # flags
        self.flags: List[Tuple[float,float]] = []
        self.flag_dedup = 0.3
        self.flag_close_thresh = 0.48
        self.immediate_confirm_delay = 3.2
        self.confirm_time: Optional[float] = None
        self.confirm_report: Optional[Tuple[float,float]] = None

        # visualization
        self.vis_size = 700; self.scale = 150.0
        self.offset = np.array([self.vis_size//2, self.vis_size//2], dtype=int)

        self.last_stop_publish_time = -9999
        self.forced_stop_timer = None
        self.forced_stop_end_time = None
        self.last_status = None

        self.get_logger().info("🔺 SimpleTriangleDetector started (IMMEDIATE-CONFIRM delay mode).")

        cv2.namedWindow("SimpleTriangleDetector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SimpleTriangleDetector", self.vis_size, self.vis_size)

    def odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        self.robot_yaw = math.atan2(siny, cosy)

    def scan_cb(self, msg: LaserScan):

        # ----------- preprocessing -----------
        ranges_all = np.array(msg.ranges, dtype=np.float32)
        angles_all = np.linspace(msg.angle_min, msg.angle_max, ranges_all.shape[0], endpoint=True)
        ranges = np.nan_to_num(ranges_all, nan=msg.range_max + 1.0, posinf=msg.range_max + 1.0)

        valid = (ranges > msg.range_min) & (ranges < (msg.range_max + 1e-6))
        if np.sum(valid) < 12:
            self._publish_no_shape()
            self._visualize(np.zeros((0,2)), [], None)
            return

        idxs = np.where(valid)[0]
        r = ranges[idxs]; a = angles_all[idxs]
        points = polar_to_xy(r, a)

        # ----------- cluster -----------
        clusters = []
        cur = [points[0]]
        for i in range(1, points.shape[0]):
            if np.linalg.norm(points[i] - points[i-1]) > self.cluster_gap:
                clusters.append(np.array(cur)); cur = [points[i]]
            else:
                cur.append(points[i])
        clusters.append(np.array(cur))

        # ----------- split-merge -> raw segments -----------
        raw_segments = []
        for cl in clusters:
            if cl.shape[0] < self.min_segment_pts: continue
            segs = split_and_merge(cl, max_dist=self.split_merge_dist, min_pts=self.min_segment_pts)
            for (A,B) in segs:
                if np.linalg.norm(B - A) < self.min_line_length: continue
                raw_segments.append((A.copy(), B.copy()))

        if len(raw_segments) < 2:
            self._publish_no_shape()
            self._visualize(points, raw_segments, None)
            return

        # ----------- merge colinear -----------
        merged = merge_colinear(raw_segments, angle_thresh=12.0, endpoint_merge=0.20)

        # ----------- extend segments -----------
        ext = []
        for (A,B) in merged:
            Ae, Be = extend_segment(A, B, self.extend_len)
            ext.append((Ae, Be, A, B))

        # ----------- find triangle candidate -----------
        candidate = None
        for i in range(len(ext)):
            for j in range(i+1, len(ext)):
                Ae, Be, Ao, Bo = ext[i]
                Ce, De, Co, Do = ext[j]

                l1 = fit_line_two_points(Ao, Bo)
                l2 = fit_line_two_points(Co, Do)
                inter = intersection(l1, l2)

                # find corner
                corner = None; valid_corner = False
                if inter is not None and seg_contains_point(Ao, Bo, inter, tol=self.endpoint_tol) \
                                     and seg_contains_point(Co, Do, inter, tol=self.endpoint_tol):
                    corner = inter; valid_corner = True
                else:
                    # endpoint adjacency
                    endpoints1 = [Ao, Bo]; endpoints2 = [Co, Do]
                    min_d = 1e9; best = None
                    for e1 in endpoints1:
                        for e2 in endpoints2:
                            d = np.linalg.norm(e1 - e2)
                            if d < min_d:
                                min_d = d; best = (e1, e2)
                    if min_d <= self.endpoint_tol:
                        corner = (best[0] + best[1]) / 2.0
                        valid_corner = True

                if not valid_corner:
                    continue

                # compute angle
                def side_vec(corner_pt, A, B):
                    dA = np.linalg.norm(A - corner_pt)
                    dB = np.linalg.norm(B - corner_pt)
                    far = A if dA > dB else B
                    return far - corner_pt, np.linalg.norm(far - corner_pt)

                v1, L1 = side_vec(corner, Ao, Bo)
                v2, L2 = side_vec(corner, Co, Do)

                if L1 < self.min_line_length or L2 < self.min_line_length:
                    continue

                ang = angle_between(v1, v2)
                dist = np.linalg.norm(corner)

                if dist > self.max_detect_distance:
                    continue

                if abs(ang - self.angle_target) <= self.angle_tol:
                    candidate = {
                        'corner': corner, 'v1': v1, 'v2': v2,
                        'L1': L1, 'L2': L2, 'ang': ang,
                        'dist': dist
                    }
                    break
            if candidate is not None:
                break

        # ----------- ignore triangle in first-lane zone -----------
        if candidate is not None:
            if (0.2 <= self.robot_x <= 0.35) and (self.robot_y < 0):
                #self.get_logger().info("(robot in first lane).")
                candidate = None

        # ============================================================
        #     *** FROM HERE ON: NO MORE gx, gy (triangle coords) ***
        # ============================================================

        # ----------- accept detection -----------
        now = self.get_clock().now().seconds_nanoseconds()[0]

        if candidate is not None:
            corner = candidate['corner']
            rx = 0.0; ry = float(corner[1])  # LOCAL points only

            # (gx, gy) removed completely for publishing
            # But still needed for flag geometry
            gx = float(self.robot_x)
            gy = self.robot_y + ry

            # ------- FLAG DEDUP LOGIC (unchanged) -------
            too_close = False
            for (fx, fy) in self.flags:
                if math.hypot(fx - gx, fy - gy) <= self.flag_dedup:
                    too_close = True
                    break

            if not too_close:
                self.flags.append((gx, gy))

                if len(self.flags) >= 2:
                    f1 = self.flags[-2]; f2 = self.flags[-1]
                    d = math.hypot(f1[0] - f2[0], f1[1] - f2[1])

                    if d <= self.flag_close_thresh:
                        self.confirm_time = now
                        self.confirm_report = ((f1[0]+f2[0])/2, (f1[1]+f2[1])/2)

        # ----------- confirm stop -----------
        if (self.confirm_time is not None) and (now - self.confirm_time >= self.immediate_confirm_delay):
            if self.forced_stop_timer is None:
                self._trigger_forced_stop_immediate()
            self.confirm_time = None
            self.confirm_report = None

        self._visualize(points, merged, candidate)

    # ====================================================================
    #                STOP HANDLER (ALWAYS PUBLISH ODOM)
    # ====================================================================
    def _trigger_forced_stop_immediate(self):

        gx_report = float(self.robot_x)
        gy_report = float(self.robot_y)

        plant_id = self.plant_mgr.assign(gx_report, gy_report,self.robot_yaw,0)


        msg_bool = Bool(); msg_bool.data = True; self.shape_pub.publish(msg_bool)

        st = String()
        st.data = f"FERTILIZER_REQUIRED,{gx_report:.3f},{gy_report:.3f},{plant_id}"
        self.status_pub.publish(st)
        self.last_status = st.data

        self.get_logger().warn(f"🔴 Forced STOP: {st.data}")

        interval = 1.0 / float(self.force_stop_rate)
        now = self.get_clock().now().seconds_nanoseconds()[0]
        self.forced_stop_end_time = now + self.force_stop_duration
        self.forced_stop_timer = self.create_timer(interval, self._stop_hold_cb)

        self.accepted_count += 1

    # --- remaining helper functions unchanged ---
    

    def _stop_hold_cb(self):
        now = self.get_clock().now().seconds_nanoseconds()[0]
        if now >= self.forced_stop_end_time:
            msgf = Bool()
            msgf.data = False
            self.shape_pub.publish(msgf)

            s = String()
            s.data = "NO_SHAPE"
            self.status_pub.publish(s)
            self.last_status = "NO_SHAPE"

            self.get_logger().info("🟢 Triangle stop released")

            self.forced_stop_timer.cancel()
            self.forced_stop_timer = None



    def _publish_no_shape(self):
        if self.last_status == "NO_SHAPE": return
        m = Bool(); m.data = False; self.shape_pub.publish(m)
        s = String(); s.data = "NO_SHAPE"; self.status_pub.publish(s)
        self.last_status = "NO_SHAPE"

    def _visualize(self, points, segments, det):
        img = np.zeros((self.vis_size, self.vis_size, 3), dtype=np.uint8)
        def w2i(pt):
            p = (pt * self.scale).astype(int) + self.offset
            return (p[0], p[1])

        for p in points:
            cv2.circle(img, w2i(p), 1, (0,0,255), -1)
        for (A,B) in segments:
            cv2.line(img, w2i(A), w2i(B), (0,255,0), 2)

        if det is not None:
            corner = det['corner']
            v1 = det['v1']; v2 = det['v2']
            B = corner + v1 / (np.linalg.norm(v1)+1e-9) * det['L1']
            C = corner + v2 / (np.linalg.norm(v2)+1e-9) * det['L2']
            pts = [corner, B, C]
            pts = np.array([w2i(p) for p in pts])
            cv2.polylines(img, [pts], False, (0,255,255), 2)

        cv2.imshow("SimpleTriangleDetector", img)


# ---------------- Main ----------------
def main():
    rclpy.init()
    try:
        plant_mgr = PlantIDManager()
        square_node = SquareDetectorForceStop(plant_mgr)
        tri_node = SimpleTriangleDetector(plant_mgr)

        executor = SingleThreadedExecutor()
        executor.add_node(square_node)
        executor.add_node(tri_node)

        last_yaw = None
        YAW_EPS = 1e-4  # radians (~0.0057 deg)

        try:
            while rclpy.ok():
                # yaw comes from odom subscriber inside nodes
                current_yaw = square_node.robot_yaw
                #print(f"Yaw (rad): {current_yaw:.4f} | Yaw (deg): {math.degrees(current_yaw):.2f}")


                # spin ONLY if yaw is stable
                if (math.degrees(current_yaw)<-89 and math.degrees(current_yaw)>-91) or (math.degrees(current_yaw)<91 or math.degrees(current_yaw)>89):
                    executor.spin_once(timeout_sec=0.05)

                last_yaw = current_yaw

                # keep OpenCV GUI responsive
                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # ESC
                    break

        except KeyboardInterrupt:
            pass

        finally:
            # cancel any active timers
            for node in (square_node, tri_node):
                try:
                    if node.forced_stop_timer is not None:
                        node.forced_stop_timer.cancel()
                except Exception:
                    pass

            cv2.destroyAllWindows()

            try:
                square_node.destroy_node()
            except Exception:
                pass

            try:
                tri_node.destroy_node()
            except Exception:
                pass

            executor.shutdown()

    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
