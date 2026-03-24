import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header, Float32MultiArray

import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO
import math

# YOLO Pose Keypoints:
# 0: Nose, 1: LEye, 2: REye, 3: LEar, 4: REar, 5: LShoulder, 6: RShoulder
# 7: LElbow, 8: RElbow, 9: LWrist, 10: RWrist, 11: LHip, 12: RHip
# 13: LKnee, 14: RKnee, 15: LAnkle, 16: RAnkle
# 17: Custom Neck (Midpoint 5 & 6)

HUMAN_CAPSULES = [
    (17, 0, 0.18, (255, 0, 0)),   # Head (Neck to Nose)
    (5, 7, 0.12, (0, 255, 0)),    # L Upper Arm
    (7, 9, 0.08, (0, 255, 0)),    # L Forearm
    (6, 8, 0.12, (0, 255, 255)),  # R Upper Arm
    (8, 10, 0.08, (0, 255, 255)), # R Forearm
    (17, 11, 0.22, (255, 255, 0)),# Torso L
    (17, 12, 0.22, (255, 255, 0)),# Torso R
    (9, 9, 0.15, (0, 255, 0)),    # L Hand
    (10, 10, 0.15, (0, 255, 0))   # R Hand
]

SKELETON_LINES = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Face
    (17, 5), (17, 6), (17, 0),           # Neck connections
    (5, 7), (7, 9), (6, 8), (8, 10),     # Arms
    (5, 11), (6, 12), (11, 12),          # Torso
]

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=0.1, beta=0.5, d_cutoff=1.0):
        self.t_prev = t0
        self.x_prev = x0
        self.dx_prev = np.full(x0.shape, dx0) if isinstance(dx0, float) else dx0
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

    def alpha(self, t, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / t)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0:
            return x

        a_d = self.alpha(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        speed = np.linalg.norm(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        a = self.alpha(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.t_prev = t
        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat

class RobotMask:
    def __init__(self, T_C_to_W):
        self.T_W_to_C = np.linalg.inv(T_C_to_W)
        self.robot_spheres = np.zeros((0, 4))

    def update_robot_spheres(self, spheres):
        self.robot_spheres = np.zeros((0, 4)) 
        for sphere in spheres:
            sx, sy, sz, r = sphere
            sphere_pos_cam = self.T_W_to_C @ np.array([sx, sy, sz, 1.0])
            sphere_cam = np.array([[sphere_pos_cam[0], sphere_pos_cam[1], sphere_pos_cam[2], r]])
            self.robot_spheres = np.append(self.robot_spheres, sphere_cam, axis=0)            

    def is_joint_occluded(self, joint_pos):
        for sphere in self.robot_spheres:
            if self.check_sphere_occlusion(sphere, joint_pos):
                    return True
        return False
    
    def check_sphere_occlusion(self, sphere, joint_pos):
        sx, sy, sz, r = sphere
        sphere_center = np.array([sx, sy, sz])

        if np.dot(joint_pos, sphere_center) > 0:
            joint_pos_norm = joint_pos / np.linalg.norm(joint_pos)
            sphere_center_proj_scalar = np.dot(sphere_center, joint_pos_norm)
            sphere_center_proj_vec = sphere_center_proj_scalar * joint_pos_norm
            proj_vec = sphere_center_proj_vec - sphere_center
            proj_length = np.linalg.norm(proj_vec)

            return proj_length < (r + 0.03) and np.linalg.norm(joint_pos) > np.linalg.norm(sphere_center)

        return False

class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")

        self.use_auto_calibration = True 
        self.is_calibrated = not self.use_auto_calibration

        self.model = YOLO("yolo26l-pose.pt")

        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER
        init.camera_fps = 60
        if self.zed.open(init) != sl.ERROR_CODE.SUCCESS:
            print("ZED failed to open.")
            exit()

        self.cam_param = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        
        # Replace simple history with 1Euro Filters
        self.filters_3d = {}
        self.filters_2d = {}

        self.last_good_centroids = ([], [], [], [])

        # Avoid flickering issues
        self.missed_frames = 0
        self.MAX_COAST_FRAMES = 60

        self.image_sl = sl.Mat()
        self.point_cloud = sl.Mat()

        self.freq = 60
        self.timer = self.create_timer(1/self.freq, self.publish_tracker_state)

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        self.tracker_pub = self.create_publisher(MarkerArray, "tracker_centroids", qos_profile)
        
        self.robot_spheres = np.zeros((0, 4))
        self.sphere_sub = self.create_subscription(
            Float32MultiArray, 
            "franka/robot_spheres", 
            self.sphere_callback, 
            qos_profile
        )

    def run_calibration(self):
        """Looks for checkerboard and calculates the World-to-Camera Transform."""
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_sl, sl.VIEW.LEFT)
            bgra_frame = self.image_sl.get_data()
            frame_rgb = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2RGB)
            gray = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2GRAY)

            pattern_size = (8, 11) 
            square_size = 0.03

            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            calibration_successful = False
            
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(frame_rgb, pattern_size, corners2, ret)
                
                objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
                objp *= square_size
                
                K = np.array([
                    [self.cam_param.fx, 0, self.cam_param.cx],
                    [0, self.cam_param.fy, self.cam_param.cy],
                    [0, 0, 1]
                ], dtype=np.float64)
                
                dist_coeffs = np.zeros((4, 1))
                success, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist_coeffs)
             
                if success:
                    R, _ = cv2.Rodrigues(rvec)

                    offset_table = np.array([[0.34], [0.0], [0.0]])
                    tvec = tvec + (R @ offset_table)

                    flip_xz = np.array([
                        [-1.0,  0.0,  0.0],
                        [0.0, 1.0,  0.0],
                        [0.0,  0.0, -1.0]
                    ])
                    R = R @ flip_xz
                    
                    R_inv = R.T
                    t_inv = -R_inv @ tvec
                    
                    self.T_C_to_W = np.eye(4)
                    self.T_C_to_W[:3, :3] = R_inv
                    self.T_C_to_W[:3, 3] = t_inv.flatten()
                    self.W_to_T_C = np.linalg.inv(self.T_C_to_W)

                    self.is_calibrated = True
                    calibration_successful = True
                    self.get_logger().info("Camera calibrated successfully!")
            else:
                cv2.putText(frame_rgb, "Looking for Checkerboard...", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Calibration", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            if calibration_successful:
                cv2.waitKey(2000)
                cv2.destroyWindow("Calibration")
            else:
                cv2.waitKey(1)

    def publish_tracker_state(self):
        if not self.is_calibrated:
            self.run_calibration()
            return

        res = self.get_centroids()
        if res is None:
            return
            
        centroids_x, centroids_y, centroids_z, connection_map = res
        marker_array = MarkerArray()

        for i in range(len(centroids_x)):
            marker_id = connection_map[i]
            
            pt_cam = np.array([centroids_x[i], centroids_y[i], centroids_z[i], 1.0])
            if self.use_auto_calibration:
                pt_publish = self.T_C_to_W @ pt_cam
            else:
                pt_publish = pt_cam         

            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = self.get_skeleton_map(marker_id)
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(pt_publish[0])
            marker.pose.position.y = float(pt_publish[1])
            marker.pose.position.z = float(pt_publish[2])
            
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0 
            marker.color.g = 1.0 
            
            marker_array.markers.append(marker)

        self.tracker_pub.publish(marker_array)

    def sphere_callback(self, msg: Float32MultiArray):
        flat_data = np.array(msg.data)
        if len(flat_data) > 0:
            self.robot_spheres = flat_data.reshape(-1, 4)

        if self.is_calibrated and hasattr(self, 'T_C_to_W'):
            if not hasattr(self, 'robot_masker'):
                self.robot_masker = RobotMask(self.T_C_to_W)
            self.robot_masker.update_robot_spheres(self.robot_spheres)

    def get_pixel_radius(self, z_depth, radius_m, focal_length):
        if z_depth <= 0: return 1
        return int((radius_m * focal_length) / z_depth)

    def get_centroids(self):
        centroids_x, centroids_y, centroids_z = [], [], [] 

        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_sl, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)

            bgra_frame = self.image_sl.get_data()
            frame_rgb = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2RGB)
            
            capsule_overlay = np.zeros_like(bgra_frame)
            display_frame = bgra_frame.copy()
            
            results = self.model.track(frame_rgb, verbose=False, conf=0.6, persist=True, tracker="bytetrack.yaml")
            current_3d = {}
            current_2d = {}
            connection_map = []

            if len(results[0].boxes.data) == 0:
                # If we have recent data and haven't exceeded our patience, COAST!
                if hasattr(self, 'last_good_centroids') and len(self.last_good_centroids[0]) > 0 and self.missed_frames < self.MAX_COAST_FRAMES:
                    self.missed_frames += 1
                    # self.get_logger().info(f"YOLO blink! Coasting for frame {self.missed_frames}/{self.MAX_COAST_FRAMES}")
                    
                    # Optional: Draw text on screen so you know it's coasting
                    cv2.putText(display_frame, "COASTING (NO YOLO DET)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    cv2.imshow("ZED YOLO26 OSCBF Model", display_frame)
                    cv2.waitKey(1)
                    
                    return self.last_good_centroids
                else:
                    # We have genuinely lost the human. Let the arrays be empty.
                    cv2.putText(display_frame, "HUMAN LOST", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("ZED YOLO26 OSCBF Model", display_frame)
                    cv2.waitKey(1)
                    return centroids_x, centroids_y, centroids_z, connection_map
            
            # If we made it here, YOLO found something! Reset the missed frames counter.
            self.missed_frames = 0

            tight_bbox = results[0].boxes.data[0].cpu().numpy()
            kpts = results[0].keypoints.data[0].cpu().numpy()
            current_time = self.get_clock().now().nanoseconds / 1e9

            # Reject glitches
            bbox_cx = int((tight_bbox[0] + tight_bbox[2]) / 2)
            bbox_cy = int((tight_bbox[1] + tight_bbox[3]) / 2)
            
            # Get the physical 3D depth of the human's center
            err, pt_center = self.point_cloud.get_value(bbox_cx, bbox_cy)
            
            if err == sl.ERROR_CODE.SUCCESS and np.isfinite(pt_center[2]):
                current_center_3d = pt_center[:3]
                
                # If we have a previous center, check the distance!
                if hasattr(self, 'last_center_3d'):
                    # Calculate Euclidean distance the human moved since last frame
                    distance_moved = np.linalg.norm(current_center_3d - self.last_center_3d)
                    
                    MAX_JUMP_METERS = 0.20 
                    
                    if distance_moved > MAX_JUMP_METERS:
                        self.get_logger().warn(f"GLITCH BLOCKED: Skeleton tried to jump {distance_moved:.2f}m!")
                        # Skip frame and return last known good state
                        return self.last_good_centroids 
                
                # If the jump was safe, save this center for the next frame's check
                self.last_center_3d = current_center_3d
              
            for i in range(17):
                kx, ky, conf = kpts[i]

                if conf > 0.6:
                    err, pt3d = self.point_cloud.get_value(int(kx), int(ky))
                    
                    if np.isfinite(pt3d[2]) and pt3d[2] > 0:
                        raw_3d = pt3d[:3]
                        raw_2d = np.array([kx, ky], dtype=float)

                        # Init filter on first appearance
                        if i not in self.filters_3d:
                            self.filters_3d[i] = OneEuroFilter(current_time, raw_3d, min_cutoff=0.005, beta=3)
                            self.filters_2d[i] = OneEuroFilter(current_time, raw_2d, min_cutoff=0.005, beta=0.1)

                        is_occluded = False
                        # OCCLUSION CHECK: Is the robot blocking the LAST KNOWN position?
                        if hasattr(self, 'robot_masker'):
                            if self.robot_masker.is_joint_occluded(self.filters_3d[i].x_prev):
                                is_occluded = True

                        if is_occluded:
                            # Dead Reckon: Freeze joint at last known position
                            #current_3d[i] = self.filters_3d[i].x_prev
                            #current_2d[i] = (int(self.filters_2d[i].x_prev[0]), int(self.filters_2d[i].x_prev[1]))
                            smooth_3d = self.filters_3d[i](current_time, raw_3d)
                            smooth_2d = self.filters_2d[i](current_time, raw_2d)
                            current_3d[i] = smooth_3d
                            current_2d[i] = (int(smooth_2d[0]), int(smooth_2d[1]))
                            cv2.circle(display_frame, current_2d[i], 8, (0, 0, 255), -1) # Red if occluded
                        else:
                            # Normal Filter Update
                            smooth_3d = self.filters_3d[i](current_time, raw_3d)
                            smooth_2d = self.filters_2d[i](current_time, raw_2d)
                            
                            current_3d[i] = smooth_3d
                            current_2d[i] = (int(smooth_2d[0]), int(smooth_2d[1]))
                            cv2.circle(display_frame, current_2d[i], 4, (0, 255, 0), -1) # Green if active

                        centroids_x.append(current_3d[i][0])
                        centroids_y.append(current_3d[i][1])
                        centroids_z.append(current_3d[i][2])
                        connection_map.append(i)

            # Print coordinates beside left wrist for debugging
            if 9 in current_3d and 9 in current_2d:
                w_3d = current_3d[9]

                # Transform to base frame if auto-calibration is enabled
                if self.use_auto_calibration:
                    w_3d = self.T_C_to_W @ np.array([w_3d[0], w_3d[1], w_3d[2], 1.0])
                    w_3d = w_3d[:3]  # Drop the homogeneous coordinate

                w_2d = current_2d[9]
                coord_txt = f"W: {w_3d[0]:.2f}, {w_3d[1]:.2f}, {w_3d[2]:.2f}"
                coord_txt_2D = f"2D: {w_2d[0]}, {w_2d[1]}"
                
                text_pos_2d = (w_2d[0] + 15, w_2d[1]-20)                
                text_pos = (int(w_2d[0]) + 15, int(w_2d[1]))

                cv2.putText(display_frame, coord_txt, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)  
                # cv2.putText(display_frame, coord_txt_2D, text_pos_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2) 
            
            # Virtual Neck Processing
            if 5 in current_3d and 6 in current_3d:
                neck3d = (current_3d[5] + current_3d[6]) / 2.0
                nx = int((current_2d[5][0] + current_2d[6][0]) / 2)
                ny = int((current_2d[5][1] + current_2d[6][1]) / 2)
                current_2d[17] = (nx, ny)
                current_3d[17] = neck3d

            # Draw Volumetric Capsules
            for start, end, r_m, color in HUMAN_CAPSULES:
                if start in current_2d and end in current_2d:
                    p1, p2 = current_2d[start], current_2d[end]
                    z_avg = (current_3d[start][2] + current_3d[end][2]) / 2
                    thickness = self.get_pixel_radius(z_avg, r_m, self.cam_param.fx)
                    
                    cv2.line(capsule_overlay, p1, p2, color, thickness)
                    cv2.circle(capsule_overlay, p1, thickness // 2, color, -1)
                    cv2.circle(capsule_overlay, p2, thickness // 2, color, -1)

            # Draw Skeletal Connections
            for partA, partB in SKELETON_LINES:
                if partA in current_2d and partB in current_2d:
                    cv2.line(display_frame, current_2d[partA], current_2d[partB], (255, 255, 255), 2)

            # Draw Robot Spheres
            if hasattr(self, 'robot_masker'):            
                for sphere in self.robot_masker.robot_spheres:
                    sx, sy, sz, r = sphere
                    if sz > 0.05: 
                        u = int((sx * self.cam_param.fx / sz) + self.cam_param.cx)
                        v = int((sy * self.cam_param.fy / sz) + self.cam_param.cy)
                        pix_radius = self.get_pixel_radius(sz, r, self.cam_param.fx)
                        cv2.circle(capsule_overlay, (u, v), pix_radius, (255, 120, 0), -1)
                        cv2.circle(display_frame, (u, v), pix_radius, (255, 120, 0), 1)

            if tight_bbox is not None:
                cv2.rectangle(display_frame, (int(tight_bbox[0]), int(tight_bbox[1])), (int(tight_bbox[2]), int(tight_bbox[3])), (0, 255, 0), 2)
               
            cv2.addWeighted(capsule_overlay, 0.4, display_frame, 1.0, 0, display_frame)
            cv2.imshow("ZED YOLO26 OSCBF Model", display_frame)
            cv2.waitKey(1)

            return centroids_x, centroids_y, centroids_z, connection_map      

    def get_skeleton_map(self, id):
        match id:
            case 0: return "nose"
            case 1: return "left eye"
            case 2: return "right eye"
            case 3: return "left ear"
            case 4: return "right ear"
            case 5: return "left shoulder"
            case 6: return "right shoulder"
            case 7: return "left elbow"
            case 8: return "right elbow"
            case 9: return "left wrist"
            case 10: return "right wrist"
            case 11: return "left hip"
            case 12: return "right hip"
            case 13: return "left knee"
            case 14: return "right knee"
            case 15: return "left ankle"
            case 16: return "right ankle"
            case 17: return "neck"
            case _: return "unknown"

def main(args=None):
    rclpy.init(args=None)
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()  

if __name__ == '__main__':
    main()