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
from std_msgs.msg import Header

import pyzed.sl as sl
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

import math
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

HUMAN_CAPSULES = [
    (34, 34, 0.20, (255, 0, 0)),   # Head
    (11, 13, 0.13, (0, 255, 0)),   # Upper Arm left
    (12, 14, 0.13, (0, 255, 255)),   # Upper Arm right
    (13, 15, 0.10, (0, 255, 0)),   # Forearm left
    (14, 16, 0.10, (0, 255, 255)),   # Forearm right
    (15, 19, 0.10, (0, 0, 255)),   # Left Hand
    (16, 20, 0.10, (0, 0, 255)),   # Right Hand
    (33, 23, 0.22, (255, 255, 0)),   # Torso L
    (33, 24, 0.22, (255, 255, 0))    # Torso R
]
SKELETON_LINES = [
    (7, 8),                                     # Face
    (11, 12), (33, 34),                         # Neck connections
    (11, 13), (13, 15), (12, 14), (14, 16),     # Arms
    (15, 19), (16, 20), (15, 17), (16, 18), (15, 21), (16, 22), # Hands
    (11, 23), (12, 24), (23, 24),               # Torso
]
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=0.1, beta=0.5, d_cutoff=1.0):
        """
        min_cutoff: Decreasing this kills slow-speed jitter (but increases resting lag).
        beta: Increasing this reduces high-speed lag (but lets more high-speed jitter through).
        """
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

        # Filter the derivative (velocity)
        a_d = self.alpha(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Calculate the dynamic cutoff based on speed
        speed = np.linalg.norm(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        # Filter the actual point
        a = self.alpha(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        # Save for next frame
        self.t_prev = t
        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat


class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")

        self.use_auto_calibration = False  # Toggle checkerboard calibration here
        self.is_calibrated = not self.use_auto_calibration

        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER
        init.camera_fps = 30
        if self.zed.open(init) != sl.ERROR_CODE.SUCCESS:
            print("ZED failed to open.")
            exit()

        self.cam_param = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self.focal_length = self.cam_param.focal_length_metric
        
        self.filters_3d = {}
        self.filters_2d = {}
        
        self.image_sl = sl.Mat()
        self.point_cloud = sl.Mat()

        model_path = '/home/pdzw120w/dev/ros2_ws/src/SP_franka_oscbf/human_tracker/human_tracker/pose_landmarker_heavy.task'
 
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.pose_callback
        )

        # Init AI model
        self.landmarker = PoseLandmarker.create_from_options(self.options)
        self.latest_pose_result = None

        self.freq = 30
        self.timer = self.create_timer(1/self.freq, self.publish_tracker_state)

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        self.tracker_pub = self.create_publisher(MarkerArray, "tracker_centroids", qos_profile)

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
            
            if ret:
                # Refine corner locations for precision
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Draw for visual feedback
                cv2.drawChessboardCorners(frame_rgb, pattern_size, corners2, ret)
                
                # Generate standard 3D points of the checkerboard (Z=0 plane)
                objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
                objp *= square_size
                
                # Setup camera intrinsics matrix
                K = np.array([
                    [self.cam_param.fx, 0, self.cam_param.cx],
                    [0, self.cam_param.fy, self.cam_param.cy],
                    [0, 0, 1]
                ], dtype=np.float64)
                
                # ZED retrieve_image is usually rectified, so distortion is 0
                dist_coeffs = np.zeros((4, 1))

                # Solve for rotation and translation from World to Camera
                success, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist_coeffs)

                # Add x-offset
                tvec[0] += 0.17
                
                if success:
                    R, _ = cv2.Rodrigues(rvec)

                    flip_xz = np.array([
                        [-1.0,  0.0,  0.0],
                        [0.0, 1.0,  0.0],
                        [0.0,  0.0, -1.0]
                    ])
                    
                    R = R @ flip_xz
                    
                    # We want Camera to World transform, so we invert the matrix:
                    # P_w = R^T * P_c - R^T * t
                    R_inv = R.T
                    t_inv = -R_inv @ tvec
                    
                    self.T_C_to_W = np.eye(4)
                    self.T_C_to_W[:3, :3] = R_inv
                    self.T_C_to_W[:3, 3] = t_inv.flatten()
                    
                    self.is_calibrated = True
                    self.get_logger().info("Camera calibrated successfully!")
                    self.get_logger().info(f"Transformation Matrix:\n{self.T_C_to_W}")
                    
                    cv2.destroyWindow("Calibration")
                    return
            else:
                cv2.putText(frame_rgb, "Looking for Checkerboard...", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show the live feed to help aim the checkerboard
            cv2.imshow("Calibration", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def publish_tracker_state(self):
        # Halt execution until calibrated
        if not self.is_calibrated:
            self.run_calibration()
            return

        res = self.get_centroids()
        if res is None:
            return
        res = self.get_centroids()
        if res is None:
            return
            
        centroids_x, centroids_y, centroids_z, actual_ids = res

        marker_array = MarkerArray()

        for i in range(len(centroids_x)):
            marker_id = actual_ids[i]

            # Transform to Base Frame
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

    def get_pixel_radius(self, z_depth, radius_m, focal_length):
        if z_depth <= 0: return 1
        return int((radius_m * focal_length) / z_depth)

    def get_centroids(self):
        centroids_x, centroids_y, centroids_z = [], [], [] 

        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_sl, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)

            bgra_frame = self.image_sl.get_data()
            display_frame = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2BGR)
            frame_rgb = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2RGB)
            image_height, image_width, _ = frame_rgb.shape

            current_3d, current_2d = {}, {}
            capsule_overlay = np.zeros_like(display_frame)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            self.landmarker.detect_async(mp_image, int(self.get_clock().now().nanoseconds / 1e6))  
            
            mp_mapping = [0, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            
            if self.latest_pose_result is not None and len(self.latest_pose_result.pose_world_landmarks) > 0:
                world_landmarks = self.latest_pose_result.pose_world_landmarks[0]
                pixel_landmarks = self.latest_pose_result.pose_landmarks[0]

                # Camera Intrinsics
                fx = self.cam_param.fx
                fy = self.cam_param.fy
                cx = self.cam_param.cx
                cy = self.cam_param.cy

                # Create Right Shoulder Anchor
                u_shoulder = int(pixel_landmarks[12].x * image_width)
                v_shoulder = int(pixel_landmarks[12].y * image_height)
                u_shoulder, v_shoulder = np.clip(u_shoulder, 0, image_width-1), np.clip(v_shoulder, 0, image_height-1)

                err, point_cloud_value = self.point_cloud.get_value(u_shoulder, v_shoulder)

                if err == sl.ERROR_CODE.SUCCESS and np.isfinite(point_cloud_value[2]):
                    
                    # Get true real-world Z depth of anchor
                    anchor_z_real = point_cloud_value[2]
                    
                    # Get MediaPipe's Z depth for the same anchor
                    anchor_z_mp = world_landmarks[12].z

                    for mp_idx in mp_mapping:
                        p_landmark = pixel_landmarks[mp_idx]
                        mp_joint = world_landmarks[mp_idx]
                        
                        # Get continuous 2D pixel coordinates for the math
                        u = p_landmark.x * image_width
                        v = p_landmark.y * image_height
                        
                        # Get integer 2D pixel coordinates for OpenCV drawing
                        px = int(np.clip(u, 0, image_width - 1))
                        py = int(np.clip(v, 0, image_height - 1))

                        # Get true z value relative to anchor (note: MediaPipes origin is MidHip)
                        true_z = anchor_z_real + (mp_joint.z - anchor_z_mp)

                        # Calculate true X and Y using focal length
                        true_x = ((u - cx) * true_z) / fx
                        true_y = ((v - cy) * true_z) / fy

                        raw_3d = np.array([true_x, true_y, true_z])
                        raw_2d = np.array([px, py], dtype=float)
                        
                        current_time = self.get_clock().now().nanoseconds / 1e9

                        if mp_idx not in self.filters_3d:
                            # Tweak min_cutoff (lower = less rest jitter) and beta (higher = less motion lag)
                            self.filters_3d[mp_idx] = OneEuroFilter(current_time, raw_3d, min_cutoff=0.005, beta=3)
                            self.filters_2d[mp_idx] = OneEuroFilter(current_time, raw_2d, min_cutoff=0.005, beta=0.1)

                        # Apply the 1 Euro Low Pass Filter
                        smooth_3d = self.filters_3d[mp_idx](current_time, raw_3d)
                        smooth_2d = self.filters_2d[mp_idx](current_time, raw_2d)

                        current_2d[mp_idx] = (int(smooth_2d[0]), int(smooth_2d[1]))
                        current_3d[mp_idx] = smooth_3d
                          
                        centroids_x.append(smooth_3d[0])
                        centroids_y.append(smooth_3d[1])
                        centroids_z.append(smooth_3d[2])
                        
                        cv2.circle(display_frame, current_2d[mp_idx], 3, (0, 0, 255), -1)

            # Print coordinates beside left wrist for debugging
            if 15 in current_3d and 15 in current_2d:
                w_3d = current_3d[15]
                w_2d = current_2d[15]
                coord_txt = f"W: {w_3d[0]:.2f}, {w_3d[1]:.2f}, {w_3d[2]:.2f}"
                coord_txt_2D = f"2D: {w_2d[0]}, {w_2d[1]}"
                
                text_pos_2d = (w_2d[0] + 15, w_2d[1]-20)                
                text_pos = (w_2d[0] + 15, w_2d[1])

                cv2.putText(display_frame, coord_txt, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)  
                # cv2.putText(display_frame, coord_txt_2D, text_pos_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)  
              
                
            # Process Virtual Neck 
            if 11 in current_3d and 12 in current_3d:
                neck3d = (current_3d[11] + current_3d[12]) / 2.0
                nx = int((current_2d[11][0] + current_2d[12][0]) / 2)
                ny = int((current_2d[11][1] + current_2d[12][1]) / 2)
                current_2d[33] = (nx, ny)
                current_3d[33] = neck3d

            # Process Center of Head
            if 7 in current_3d and 8 in current_3d:
                head3d = (current_3d[7] + current_3d[8]) / 2.0
                nx = int((current_2d[7][0] + current_2d[8][0]) / 2)
                ny = int((current_2d[7][1] + current_2d[8][1]) / 2)
                current_2d[34] = (nx, ny)
                current_3d[34] = head3d
                        
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
            for start, end in SKELETON_LINES:
                if start in current_2d and end in current_2d:
                    cv2.line(display_frame, current_2d[start], current_2d[end], (255, 255, 255), 2)
            
            # Blend the volumes
            cv2.addWeighted(capsule_overlay, 0.4, display_frame, 1.0, 0, display_frame)
            cv2.imshow("ZED Live Stream", display_frame)
            cv2.waitKey(1)

            return centroids_x, centroids_y, centroids_z, mp_mapping

        return None

    def pose_callback(self, result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.latest_pose_result = result 

    def get_skeleton_map(self, id):
        match id:
            case 0: return "nose"
            case 1: return "left eye (inner)"
            case 2: return "left eye"
            case 3: return "left eye (outer)"
            case 4: return "right eye (inner)"
            case 5: return "right eye"
            case 6: return "right eye (outer)"
            case 7: return "left ear"
            case 8: return "right ear"
            case 9: return "mouth (left)"
            case 10: return "mouth (right)"
            case 11: return "left shoulder"
            case 12: return "right shoulder"
            case 13: return "left elbow"
            case 14: return "right elbow"
            case 15: return "left wrist"
            case 16: return "right wrist"
            case 17: return "left pinky"
            case 18: return "right pinky"
            case 19: return "left index"
            case 20: return "right index"
            case 21: return "left thumb"
            case 22: return "right thumb"
            case 23: return "left hip"
            case 24: return "right hip"
            case 25: return "left knee"
            case 26: return "right knee"
            case 27: return "left ankle"
            case 28: return "right ankle"
            case 29: return "left heel"
            case 30: return "right heel"
            case 31: return "left foot index"
            case 32: return "right foot index"
            case 33: return "neck"
            case 34: return "head"
            case _: return "unknown"

def main(args=None):
    rclpy.init(args=None)
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()  

if __name__ == '__main__':
    main()