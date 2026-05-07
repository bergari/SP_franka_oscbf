import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header, Float32MultiArray
from .geometry_utils import RobotMask, get_pixel_radius
from .filters import OneEuroFilter
from .constants import YOLO_TO_HMR2, YOLO_EDGES, HUMAN_CAPSULES, get_skeleton_map, surface_to_joint_proj


import pyzed.sl as sl
import cv2
import numpy as np
import threading
import time
import os
from datetime import datetime
import signal
import sys
import torch
import math
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT

class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")

        self.use_auto_calibration = True # Set to True if calibration with checkerboard desired
        self.record_stream = False # Set to True if recording is desired
        self.live_stream = True # Set to True if livestream is desired

        self.is_calibrated = not self.use_auto_calibration
        self.beta_is_calibrated = False
        self.joints_3d_prev = {}
        self.target_lengths = {}
        self.occlusion_reason = {}
        self.prev_sources = {}
        self.source_transitions = {}
        self.source_transition_duration = 0.12 # Short ramp for hard YOLO/HMR2 switches without adding too much CBF lag
        self.prev_yolo_3d = {}
        self.prev_yolo_time = {}
        self.prev_yolo_trust = {}
        self.yolo_accept_trust = 0.75
        self.yolo_reject_trust = 0.25
        self.depth_sample_radius = 2
        self.filter_min_cutoff = 0.6
        self.filter_beta = 2.0
        self.self_occlusion_radius_m = 0.12

        # Initialize YOLO
        self.model = YOLO("/usr/local/zed/resources/yolo26m-pose.pt")

        # Initialize 4D Humans (HMR2)
        self.get_logger().info("Loading 4D Humans HMR2 Model...")
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        _original_load = torch.load
        def _trusted_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)
        try:
            torch.load = _trusted_load
            self.hmr2_model, self.model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
            self.hmr2_model = torch.compile(self.hmr2_model)
        finally:
            torch.load = _original_load  

        self.hmr2_model = self.hmr2_model.to(self.device)
        self.hmr2_model.eval()
        self.hmr2_transform = T.Compose([
            T.Resize((256, 256)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.calibrated_betas = torch.zeros(1, 10, device=self.device)
        self.smpl_model = self.hmr2_model.smpl

        # Initialize ZED Camera
        self.declare_parameter('serial_number', 0)
        serial_num = self.get_parameter('serial_number').get_parameter_value().integer_value

        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER
        init.camera_fps = 30
        self.video_writer = None

        if serial_num != 0:
            init.set_from_serial_number(serial_num)
            self.get_logger().info(f"Opening ZED camera with Serial Number: {serial_num}")
        else:
            self.get_logger().info("No serial number provided, opening default ZED camera.")

        if self.zed.open(init) != sl.ERROR_CODE.SUCCESS:
            print("ZED failed to open.")
            exit()

        self.cam_param = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self.image_sl = sl.Mat()
        self.point_cloud = sl.Mat()
        
        self.filters_3d = {}

        qos_profile_rviz = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE, 
            history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE,
        )

        qos_profile = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, 
            history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.VOLATILE,
        )

        self.tracker_pub = self.create_publisher(MarkerArray, "tracker_centroids", qos_profile_rviz)
        # self.yolo_pub = self.create_publisher(MarkerArray, "tracker_yolo", qos_profile_rviz)
        # self.hmr_pub = self.create_publisher(MarkerArray, "tracker_hmr", qos_profile_rviz)

        self.sphere_sub = self.create_subscription(
            Float32MultiArray, 
            "/franka/robot_spheres", 
            self.sphere_callback, 
            qos_profile
        )

        self.ee_target_sub = self.create_subscription(
            Float32MultiArray, 
            "/ee_target_pos_array", 
            self.ee_target_callback, 
            qos_profile
        )
        self.latest_ee_target_3d = None

        self.freq = 30 
        self.timer = self.create_timer(1/self.freq, self.publish_tracker_state)

    def smooth_source_transition(self, yolo_id, source, raw_3d, current_time):
        if source == "BLEND" or self.prev_sources.get(yolo_id) == "BLEND":
            return raw_3d

        prev_source = self.prev_sources.get(yolo_id)

        if prev_source is not None and prev_source != source:
            start_pos = raw_3d
            if yolo_id in self.filters_3d:
                start_pos = self.filters_3d[yolo_id].x_prev.copy()

            self.source_transitions[yolo_id] = {
                "source": source,
                "start_time": current_time,
                "start_pos": start_pos,
            }

        transition = self.source_transitions.get(yolo_id)
        if transition is None or transition["source"] != source:
            return raw_3d

        elapsed = current_time - transition["start_time"]
        progress = elapsed / self.source_transition_duration

        if progress >= 1.0:
            del self.source_transitions[yolo_id]
            return raw_3d

        progress = max(0.0, progress)
        blend = progress * progress * (3.0 - 2.0 * progress)
        return (1.0 - blend) * transition["start_pos"] + blend * raw_3d

    def clamp01(self, value):
        return max(0.0, min(1.0, float(value)))

    def smoothstep(self, edge0, edge1, value):
        if edge0 == edge1:
            return 1.0 if value >= edge1 else 0.0

        x = self.clamp01((value - edge0) / (edge1 - edge0))
        return x * x * (3.0 - 2.0 * x)

    def sample_depth_near_keypoint(self, kx, ky, image_shape):
        h, w = image_shape[:2]
        cx = int(round(kx))
        cy = int(round(ky))
        radius = self.depth_sample_radius
        samples = []

        for py in range(max(0, cy - radius), min(h, cy + radius + 1)):
            for px in range(max(0, cx - radius), min(w, cx + radius + 1)):
                err, pt3d = self.point_cloud.get_value(px, py)
                if err == sl.ERROR_CODE.SUCCESS and np.isfinite(pt3d[2]) and pt3d[2] > 0:
                    samples.append([pt3d[0], pt3d[1], pt3d[2]])

        total_pixels = (2 * radius + 1) ** 2
        if not samples:
            return None, 0.0, 1.0

        samples = np.array(samples)
        median_xyz = np.median(samples, axis=0)
        valid_ratio = len(samples) / total_pixels
        depth_std = np.std(samples[:, 2])
        depth_trust = self.smoothstep(0.2, 0.65, valid_ratio)
        depth_trust *= 1.0 - self.smoothstep(0.03, 0.12, depth_std)
        return median_xyz, self.clamp01(depth_trust), depth_std

    def set_trust_factor(self, yolo_trust, trust_factors, yolo_id, reason, factor):
        if yolo_id not in yolo_trust:
            return

        factor = self.clamp01(factor)
        trust_factors.setdefault(yolo_id, {})[reason] = min(
            trust_factors.setdefault(yolo_id, {}).get(reason, 1.0),
            factor,
        )
        yolo_trust[yolo_id] *= factor

    def apply_trust_penalty(self, yolo_trust, trust_factors, yolo_id, factor, reason):
        if yolo_id not in yolo_trust:
            return

        self.set_trust_factor(yolo_trust, trust_factors, yolo_id, reason, factor)

    def penalize_less_plausible_joint(self, candidate_3d, yolo_trust, trust_factors, first_id, second_id, reason, factor=0.15):
        if first_id not in candidate_3d or second_id not in candidate_3d:
            return

        if first_id in self.filters_3d and second_id in self.filters_3d:
            first_err = np.linalg.norm(candidate_3d[first_id] - self.filters_3d[first_id].x_prev)
            second_err = np.linalg.norm(candidate_3d[second_id] - self.filters_3d[second_id].x_prev)
        else:
            first_err = candidate_3d[first_id][2]
            second_err = candidate_3d[second_id][2]

        penalty_id = first_id if first_err > second_err else second_id
        self.apply_trust_penalty(yolo_trust, trust_factors, penalty_id, factor, reason)

    def update_occlusion_reasons(self, yolo_trust, trust_factors):
        self.occlusion_reason = {}

        for yolo_id, trust in yolo_trust.items():
            if trust >= self.yolo_accept_trust:
                continue

            joint_factors = trust_factors.get(yolo_id, {})
            if not joint_factors:
                self.occlusion_reason[yolo_id] = "uncertain"
                continue

            self.occlusion_reason[yolo_id] = min(joint_factors, key=joint_factors.get)

    def yolo_blend_weight(self, trust):
        if trust <= self.yolo_reject_trust:
            return 0.0
        if trust >= self.yolo_accept_trust:
            return 1.0
        return self.smoothstep(self.yolo_reject_trust, self.yolo_accept_trust, trust)

    def distance_to_segment_2d(self, point, seg_start, seg_end):
        point = np.array(point, dtype=float)
        seg_start = np.array(seg_start, dtype=float)
        seg_end = np.array(seg_end, dtype=float)
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)

        if seg_len_sq <= 1e-6:
            return np.linalg.norm(point - seg_start), 0.0

        t = np.dot(point - seg_start, seg_vec) / seg_len_sq
        t = self.clamp01(t)
        closest = seg_start + t * seg_vec
        return np.linalg.norm(point - closest), t

    def apply_self_occlusion_penalties(self, candidate_3d, candidate_2d, yolo_trust, trust_factors):
        arm_segments = [(5, 7), (7, 9), (6, 8), (8, 10)]
        core_targets = [11, 12]

        for target_id in core_targets:
            if target_id not in candidate_3d or target_id not in candidate_2d:
                continue

            target_depth = candidate_3d[target_id][2]
            if target_depth <= 0.05:
                continue

            pixel_radius = (self.self_occlusion_radius_m * self.cam_param.fx) / target_depth
            pixel_radius = max(25.0, min(95.0, pixel_radius))

            best_visibility = 1.0
            for start_id, end_id in arm_segments:
                if start_id not in candidate_2d or end_id not in candidate_2d:
                    continue
                if start_id not in candidate_3d or end_id not in candidate_3d:
                    continue

                dist_px, t = self.distance_to_segment_2d(
                    candidate_2d[target_id],
                    candidate_2d[start_id],
                    candidate_2d[end_id],
                )
                segment_depth = ((1.0 - t) * candidate_3d[start_id][2]) + (t * candidate_3d[end_id][2])

                # If the arm is clearly behind the hip, it is not occluding it.
                if segment_depth > target_depth + 0.08:
                    continue

                visibility = self.smoothstep(0.35 * pixel_radius, pixel_radius, dist_px)
                best_visibility = min(best_visibility, visibility)

            if best_visibility < 1.0:
                self.set_trust_factor(
                    yolo_trust,
                    trust_factors,
                    target_id,
                    "self_occlusion",
                    best_visibility,
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

            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)   
            
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

    def run_beta_calibration(self):
        self.get_logger().info("Starting Beta Calibration. Please stand fully visible...")
        
        while rclpy.ok() and not self.beta_is_calibrated:
            if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue
            
            self.zed.retrieve_image(self.image_sl, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            frame_rgb = cv2.cvtColor(self.image_sl.get_data(), cv2.COLOR_BGRA2RGB)
            
            # Get YOLO/ZED Ground Truth
            results = self.model.track(frame_rgb, verbose=False, conf=0.7, persist=True)
            if len(results[0].boxes.data) == 0:
                continue
                
            kpts = results[0].keypoints.data[0].cpu().numpy()
            zed_3d = {}
            for yolo_id in YOLO_TO_HMR2.keys():
                kx, ky, conf = kpts[yolo_id]

                if conf > 0.98: # Only use very high confidence points for calibration
                    err, pt3d = self.point_cloud.get_value(int(kx), int(ky))
                    if err == sl.ERROR_CODE.SUCCESS and np.isfinite(pt3d[2]):
                        zed_3d[yolo_id] = torch.tensor(pt3d[:3], device=self.device)

            calibration_bones = [(5, 6),(5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)] # Arms & Torso
            if not all(u in zed_3d and v in zed_3d for u, v in calibration_bones):
                continue # Wait for a clear frame

            # Calculate the metric target lengths from ZED
            for u, v in calibration_bones:
                if u in zed_3d and v in zed_3d:
                    self.target_lengths[(u,v)] = torch.norm(zed_3d[u] - zed_3d[v]).item()
            
            # Get Initial HMR2 Pose
            if len(results[0].boxes.data) == 0:
                # No human detected! Skip the 4D Humans processing for this frame
                return
            tight_bbox = results[0].boxes.data[0].cpu().numpy()[:4]
            x1, y1, x2, y2 = tight_bbox
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            half_size = int((max(x2-x1, y2-y1) * 1.2) / 2)
            
            # Simplified crop for calibration
            crop = frame_rgb[max(0, cy-half_size):min(frame_rgb.shape[0], cy+half_size), 
                             max(0, cx-half_size):min(frame_rgb.shape[1], cx+half_size)]
            img_tensor = self.hmr2_transform(Image.fromarray(crop)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                hmr2_out = self.hmr2_model({'img': img_tensor})
                # Lock the pose (angles) for the optimization
                fixed_body_pose = hmr2_out['pred_smpl_params']['body_pose'].detach()
                fixed_global_orient = hmr2_out['pred_smpl_params']['global_orient'].detach()

            # Optimize the Betas
            self.get_logger().info("Clear frame acquired. Optimizing body shape parameters...")
            betas_opt = torch.zeros((1, 10), requires_grad=True, device=self.device)
            optimizer = torch.optim.Adam([betas_opt], lr=0.1)
            
            for step in range(300): 
                optimizer.zero_grad()
                
                # Pass angles + our learning betas into the raw SMPL model
                smpl_out = self.smpl_model(body_pose=fixed_body_pose, 
                                           global_orient=fixed_global_orient, 
                                           betas=betas_opt)
                pred_joints = smpl_out.joints[0] # Shape [45, 3]

                loss = 0.0
                for (u, v) in calibration_bones:
                    hmr_u, hmr_v = YOLO_TO_HMR2[u], YOLO_TO_HMR2[v]
                    pred_length = torch.norm(pred_joints[hmr_u] - pred_joints[hmr_v])
                    weight = 1.0
                    if (u, v) in [(7, 9), (8, 10)]:  
                        weight = 2.0  # Make forearms more important to get the shape right
                    if (u, v) == (11, 12):  
                        weight = 0.5  # Hips width defined differently in yolo vs smpl, so give it less weight
                        
                    loss += weight * (pred_length - self.target_lengths[(u,v)])**2

                loss.backward()
                optimizer.step()

            # Save the optimized betas
            self.calibrated_betas = betas_opt.detach()
            self.beta_is_calibrated = True
            self.get_logger().info("Calibration complete! Operator proportions locked.")
            return
        
    def build_marker_array(self, pose_dict, ns_prefix, offset_y, color=None, sources=None):
        marker_array = MarkerArray()
        
        # Sphere markers for joints
        for yolo_id, pt3d in pose_dict.items():
            pt_cam = np.array([pt3d[0], pt3d[1], pt3d[2], 1.0])
            pt_publish = self.T_C_to_W @ pt_cam if self.use_auto_calibration else pt_cam         

            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"{ns_prefix}_{get_skeleton_map(yolo_id)}" # use this only for debugging, in real use we want them all in the same namespace
            marker.id = yolo_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # For Rviz visualization / debugging uncomment this
            # marker.pose.position.x = float(pt_publish[2]) 
            # marker.pose.position.y = float(-pt_publish[0]) + offset_y 
            # marker.pose.position.z = float(-pt_publish[1]) 

            # For inference uncomment this
            marker.pose.position.x = float(pt_publish[0])
            marker.pose.position.y = float(pt_publish[1]) + offset_y
            marker.pose.position.z = float(pt_publish[2])

            marker.scale.x = marker.scale.y = marker.scale.z = 0.08

            if sources is not None:
                source_str = sources.get(yolo_id, "UNKNOWN")
                marker.text = source_str
                # Fused coloring: Green for YOLO, Magenta for HMR2
                if sources.get(yolo_id) == "YOLO":
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 1.0
                elif sources.get(yolo_id) == "BLEND":
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.65, 0.0, 1.0
                else:
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 1.0, 1.0
            else:
                # Pure skeleton coloring
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = color[0], color[1], color[2], 1.0
                
            marker_array.markers.append(marker)

        # Line markers for bones
        line_marker = Marker()
        line_marker.header.frame_id = "base_link"
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = f"{ns_prefix}_bones"
        line_marker.id = 1000 
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.02 
        line_marker.color.r, line_marker.color.g, line_marker.color.b, line_marker.color.a = 1.0, 1.0, 1.0, 0.8
        
        for partA, partB in YOLO_EDGES:
            if partA in pose_dict and partB in pose_dict:
                
                ptA_cam = np.array([pose_dict[partA][0], pose_dict[partA][1], pose_dict[partA][2], 1.0])
                ptA_pub = self.T_C_to_W @ ptA_cam if self.use_auto_calibration else ptA_cam
                pA = Point()

                # For Rviz / debugging uncomment this
                # pA.x = float(ptA_pub[2])
                # pA.y = float(-ptA_pub[0]) + offset_y
                # pA.z = float(-ptA_pub[1])

                # For inference uncomment this
                pA.x = float(ptA_pub[0])
                pA.y = float(ptA_pub[1]) + offset_y
                pA.z = float(ptA_pub[2])
                
                ptB_cam = np.array([pose_dict[partB][0], pose_dict[partB][1], pose_dict[partB][2], 1.0])
                ptB_pub = self.T_C_to_W @ ptB_cam if self.use_auto_calibration else ptB_cam
                pB = Point()

                # For Rviz / debugging uncomment this
                # pB.x = float(ptB_pub[2])
                # pB.y = float(-ptB_pub[0]) + offset_y
                # pB.z = float(-ptB_pub[1])

                # For inference uncomment this
                pB.x = float(ptB_pub[0])
                pB.y = float(ptB_pub[1]) + offset_y
                pB.z = float(ptB_pub[2])
                
                line_marker.points.append(pA)
                line_marker.points.append(pB)

        marker_array.markers.append(line_marker)
        return marker_array

    def publish_tracker_state(self):
        if os.path.exists("/recordings/QUIT"):
            self.get_logger().warn("QUIT file detected! Saving video and exiting...")
            os.remove("/recordings/QUIT") # Delete it so it doesn't loop
            import sys
            sys.exit(0) # This triggers the 'finally' block in main()
        if self.use_auto_calibration and not self.is_calibrated:
            self.run_calibration()
            return

        if not self.beta_is_calibrated:
            self.run_beta_calibration() 
            return

        res = self.get_centroids()
        if res is None: return

        fused_3d, pure_yolo_3d, pure_hmr_3d, fused_sources = res

        # Fused Result (Center, Multi-Colored)
        fused_msg = self.build_marker_array(fused_3d, "fused", offset_y=0.0, sources=fused_sources)
        self.tracker_pub.publish(fused_msg)

        # # Pure YOLO (Left, offset +1.0 meter, Green color)
        # yolo_msg = self.build_marker_array(pure_yolo_3d, "yolo", offset_y=1.0, color=(0.0, 1.0, 0.0))
        # self.yolo_pub.publish(yolo_msg)

        # # Pure HMR2 (Right, offset -1.0 meter, Magenta color)
        # hmr_msg = self.build_marker_array(pure_hmr_3d, "hmr2", offset_y=-1.0, color=(1.0, 0.0, 1.0))
        # self.hmr_pub.publish(hmr_msg)

    def sphere_callback(self, msg: Float32MultiArray):
        flat_data = np.array(msg.data)
        if len(flat_data) > 0:
            self.robot_spheres = flat_data.reshape(-1, 4)

        if self.is_calibrated and hasattr(self, 'T_C_to_W'):
            if not hasattr(self, 'robot_masker'):
                self.robot_masker = RobotMask(self.T_C_to_W)
            self.robot_masker.update_robot_spheres(self.robot_spheres)

    def ee_target_callback(self, msg):
        """Stores the latest 3D target position [x, y, z] in the robot's base frame."""
        sx, sy, sz = msg.data
        if self.is_calibrated and hasattr(self, 'T_C_to_W'):
            self.T_W_to_C = np.linalg.inv(self.T_C_to_W)
            self.latest_ee_target_3d = (self.T_W_to_C @ np.array([sx, sy, sz, 1.0]))[:3]

    def get_centroids(self):
        if self.zed.grab() != sl.ERROR_CODE.SUCCESS: return None
        
        # Grab current time early so the temporal filter can use it
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        self.zed.retrieve_image(self.image_sl, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)

        bgra_frame = self.image_sl.get_data()
        frame_rgb = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2RGB)
        display_frame = bgra_frame.copy()
        capsule_overlay = np.zeros_like(bgra_frame)
        
        # YOLO Inference
        results = self.model.track(frame_rgb, verbose=False, conf=0.6, persist=True)
        if len(results[0].boxes.data) == 0:
            # cv2.imshow("Fused Tracker", display_frame)
            # cv2.waitKey(1)
            return None

        if len(results[0].boxes.data) == 0:
            # No human detected! Skip the 4D Humans processing for this frame
            return
        tight_bbox = results[0].boxes.data[0].cpu().numpy()[:4]
        kpts = results[0].keypoints.data[0].cpu().numpy()
        
        candidate_3d = {}
        candidate_2d = {}
        yolo_confidence = {}
        yolo_trust = {}
        trust_factors = {}
        self.occlusion_reason = {}  # Reset occlusion reasons each frame
        
        for yolo_id in YOLO_TO_HMR2.keys():
            kx, ky, conf = kpts[yolo_id]
            yolo_confidence[yolo_id] = conf
            yolo_trust[yolo_id] = 1.0
            self.set_trust_factor(yolo_trust, trust_factors, yolo_id, "low_conf", self.smoothstep(0.70, 0.98, conf))

            pt3d_raw, depth_trust, depth_std = self.sample_depth_near_keypoint(kx, ky, frame_rgb.shape)
            depth_reason = "noisy_depth" if depth_std > 0.08 else "no_depth"
            self.set_trust_factor(yolo_trust, trust_factors, yolo_id, depth_reason, depth_trust)

            if pt3d_raw is None:
                continue

            xyz = np.array([pt3d_raw[0], pt3d_raw[1], pt3d_raw[2]])
            norm = np.linalg.norm(xyz)
            
            if norm > 0:
                offset = surface_to_joint_proj(yolo_id)
                xyz = xyz + (offset * (xyz / norm))
            
            pt3d = xyz
            
            # Uncomment to tend to use hmr2 for joints below desk level
            # if self.is_calibrated and hasattr(self, 'T_C_to_W'):
            #     pt_cam = np.array([pt3d[0], pt3d[1], pt3d[2], 1.0])       
            #     pt_world = self.T_C_to_W @ pt_cam
            #     desk_visibility = self.smoothstep(0.10, 0.20, pt_world[2])
            #     self.set_trust_factor(yolo_trust, trust_factors, yolo_id, "inside_desk", desk_visibility)

            candidate_3d[yolo_id] = pt3d
            candidate_2d[yolo_id] = (int(kx), int(ky))

        pure_yolo_3d = {k: v for k, v in candidate_3d.items()} # save for debugging/visualization

        for yolo_id in list(candidate_3d.keys()):
            pt3d = candidate_3d[yolo_id]
            # Robot Sphere Check
            if hasattr(self, 'robot_masker'):
                    robot_visibility = self.robot_masker.visibility_score(pt3d)
                    self.set_trust_factor(yolo_trust, trust_factors, yolo_id, "robot", robot_visibility)

        # OVERLAP CHECK
        # Shoulders (IDs 5 and 6)
        if 5 in candidate_3d and 6 in candidate_3d:
            shoulder_width = np.linalg.norm(candidate_3d[5] - candidate_3d[6])
            if shoulder_width < 0.25: 
                self.penalize_less_plausible_joint(candidate_3d, yolo_trust, trust_factors, 5, 6, "overlap")

        # Hips (IDs 11 and 12)
        if 11 in candidate_3d and 12 in candidate_3d:
            hip_width = np.linalg.norm(candidate_3d[11] - candidate_3d[12])
            if hip_width < 0.15: 
                self.penalize_less_plausible_joint(candidate_3d, yolo_trust, trust_factors, 11, 12, "overlap")

        # Wrists (IDs 9 and 10)
        if 9 in candidate_3d and 10 in candidate_3d:
            wrist_dist = np.linalg.norm(candidate_3d[9] - candidate_3d[10])
            if wrist_dist < 0.15: # Set threshold for wrist overlap in meters
                self.penalize_less_plausible_joint(candidate_3d, yolo_trust, trust_factors, 9, 10, "overlap")

        self.apply_self_occlusion_penalties(candidate_3d, candidate_2d, yolo_trust, trust_factors)

        # TEMPORAL Check
        current_3d = {}
        current_2d = {}

        # Loop through all joints and reduce trust for implausible temporal jumps.
        for yolo_id, pt3d in candidate_3d.items():
            if yolo_id in self.prev_yolo_3d:
                prev_pt3d = self.prev_yolo_3d[yolo_id]
                t_prev = self.prev_yolo_time[yolo_id]
                dt = current_time - t_prev
                
                if 0.0 < dt < 0.5 and self.prev_yolo_trust.get(yolo_id, 0.0) > self.yolo_reject_trust:
                    dist = np.linalg.norm(pt3d - prev_pt3d)
                    max_expected_jump = (3.0 * dt) + 0.15
                    if self.prev_sources.get(yolo_id) == "HMR2":
                        max_expected_jump = max(max_expected_jump, 0.9)

                    temporal_factor = 1.0 - self.smoothstep(
                        max_expected_jump,
                        max_expected_jump + 0.45,
                        dist,
                    )
                    self.set_trust_factor(yolo_trust, trust_factors, yolo_id, "temporal", temporal_factor)

            current_3d[yolo_id] = pt3d
            current_2d[yolo_id] = candidate_2d[yolo_id]

        # Bone Length Check
        limb_chains = [
            (7, 9),   # L Elbow -> L Wrist
            (8, 10),  # R Elbow -> R Wrist
            (5, 7),   # L Shoulder -> L Elbow
            (6, 8),   # R Shoulder -> R Elbow
            (11, 12), # L Hip -> R Hip
            (5, 6)    # L Shoulder -> R Shoulder (for head/neck length check)
        ]

        for parent, child in limb_chains:
            if parent in current_3d and child in current_3d:
                # Measure what the ZED camera sees
                zed_bone_length = np.linalg.norm(current_3d[parent] - current_3d[child])
                true_bone_length = self.target_lengths.get((parent, child))
                if true_bone_length is None or true_bone_length <= 0:
                    continue

                length_error = abs(zed_bone_length - true_bone_length)
                length_factor = 1.0 - self.smoothstep(0.2 * true_bone_length, 0.5 * true_bone_length, length_error)
                if length_factor < 1.0:

                    # Look at history to find which one is hallucinating
                    err_parent = 0.0
                    err_child = 0.0
                    
                    if parent in self.filters_3d:
                        err_parent = np.linalg.norm(current_3d[parent] - self.filters_3d[parent].x_prev)
                        
                    if child in self.filters_3d:
                        err_child = np.linalg.norm(current_3d[child] - self.filters_3d[child].x_prev)
                    
                    penalty_id = parent if err_parent > err_child else child
                    self.apply_trust_penalty(yolo_trust, trust_factors, penalty_id, length_factor, "length")

        self.update_occlusion_reasons(yolo_trust, trust_factors)

        for yolo_id, pt3d in candidate_3d.items():
            if yolo_trust.get(yolo_id, 0.0) > self.yolo_reject_trust:
                self.prev_yolo_3d[yolo_id] = pt3d.copy()
                self.prev_yolo_time[yolo_id] = current_time
                self.prev_yolo_trust[yolo_id] = yolo_trust[yolo_id]

        # HMR2 Inference
        x1, y1, x2, y2 = tight_bbox
        # Need to define center of the crop for HMR2 to later undistort image
        core_x, core_y = [], []
        # Look for Shoulders (5, 6) and Hips (11, 12)
        for target_id in [5, 6, 11, 12]:
            # Can use candidate 2d because even if the 3D is filtered out, the 2D keypoint can still be visible and useful for centering the crop
            if target_id in candidate_2d and yolo_trust.get(target_id, 0.0) > self.yolo_reject_trust: 
                core_x.append(candidate_2d[target_id][0])
                core_y.append(candidate_2d[target_id][1])
        
        # If we can see the core, use its center. Otherwise, fall back to bbox center.
        if len(core_x) > 0:
            cx = int(np.mean(core_x))
            cy = int(np.mean(core_y))
        else:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
        # Keep the size based on the bounding box so the whole body fits in the crop
        size = max(x2-x1, y2-y1) * 1.8 # Give some extra space around person so person is not cramped in
        half_size = int(size / 2)
        img_h, img_w = frame_rgb.shape[:2]
      
        src_x1, src_y1 = max(0, cx - half_size), max(0, cy - half_size)
        src_x2, src_y2 = min(img_w, cx + half_size), min(img_h, cy + half_size)
        square_crop = np.zeros((half_size*2, half_size*2, 3), dtype=np.uint8)
        dst_x1, dst_y1 = max(0, -(cx - half_size)), max(0, -(cy - half_size))
        dst_x2, dst_y2 = dst_x1 + (src_x2 - src_x1), dst_y1 + (src_y2 - src_y1)

        # Fusion of YOLO and HMR2
        fused_3d_dict = {}
        fused_sources = {}
        raw_fused_positions = {} # Temporarily store pre-filtered positions for chain math
        
        # Use black padding if the crop goes outside the image boundaries to maintain the aspect ratio of the human body
        if src_x2 > src_x1 and src_y2 > src_y1:
            square_crop[dst_y1:dst_y2, dst_x1:dst_x2] = frame_rgb[src_y1:src_y2, src_x1:src_x2]
            img_tensor = self.hmr2_transform(Image.fromarray(square_crop)).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Run the feed-forward model to get the joint angles
                hmr2_out = self.hmr2_model({'img': img_tensor})
                
                pred_pose = hmr2_out['pred_smpl_params']['body_pose']
                pred_orient = hmr2_out['pred_smpl_params']['global_orient']
                
                # Swap Betas: Generate the custom skeleton using the cached ZED proportions
                custom_smpl_out = self.smpl_model(body_pose=pred_pose, 
                                                  global_orient=pred_orient, 
                                                  betas=self.calibrated_betas)
                
                # custom_smpl_out.joints now holds 3D coordinates that respect the real bone lengths
                joints_3d = custom_smpl_out.joints[0].cpu().numpy()

                # Undistort perspective crop tilt
                fx, fy = self.cam_param.fx, self.cam_param.fy
                cx_cam, cy_cam = self.cam_param.cx, self.cam_param.cy
                
                ray_x = (cx - cx_cam) / fx
                ray_y = (cy - cy_cam) / fy
                ray_true = np.array([ray_x, ray_y, 1.0])
                ray_true = ray_true / np.linalg.norm(ray_true) 
                
                ray_hmr = np.array([0.0, 0.0, 1.0])
                v = np.cross(ray_hmr, ray_true) # axis of rotation
                c = np.dot(ray_hmr, ray_true) # cosine of angle between the two rays
                s = np.linalg.norm(v) # sine of angle between the two rays
                
                if s > 1e-6: 
                    v_skew = np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]
                    ])
                    R_perspective = np.eye(3) + v_skew + np.dot(v_skew, v_skew) * ((1 - c) / (s ** 2))
                    joints_3d = joints_3d @ R_perspective.T
                self.joints_3d_prev = joints_3d
                
                # Alignment: Snap HMR pose to real world
                core_yolo_ids = [5, 6, 11, 12] 
                yolo_anchors = []
                hmr2_anchors = []
                
                for y_id in core_yolo_ids:
                    if y_id in current_3d and self.yolo_blend_weight(yolo_trust.get(y_id, 0.0)) > 0.5:
                        h_id = YOLO_TO_HMR2[y_id]
                        yolo_anchors.append(current_3d[y_id])
                        hmr2_anchors.append(joints_3d[h_id])
                
                if len(yolo_anchors) > 0:
                    yolo_centroid = np.mean(yolo_anchors, axis=0)
                    hmr2_centroid = np.mean(hmr2_anchors, axis=0)
                    translation_offset = yolo_centroid - hmr2_centroid
                    hmr2_absolute = joints_3d + translation_offset
                elif any(y_id in self.filters_3d for y_id in core_yolo_ids):
                    prev_anchors = []
                    hmr2_anchors = []
                    for y_id in core_yolo_ids:
                        if y_id in self.filters_3d:
                            h_id = YOLO_TO_HMR2[y_id]
                            prev_anchors.append(self.filters_3d[y_id].x_prev)
                            hmr2_anchors.append(joints_3d[h_id])

                    prev_centroid = np.mean(prev_anchors, axis=0)
                    hmr2_centroid = np.mean(hmr2_anchors, axis=0)
                    translation_offset = prev_centroid - hmr2_centroid
                    hmr2_absolute = joints_3d + translation_offset
                else:
                    # Fallback just in case the camera is completely blocked
                    hmr2_absolute = joints_3d

                pure_hmr_3d = {} # for debugging/visualization
                for y_id, h_id in YOLO_TO_HMR2.items():
                    pure_hmr_3d[y_id] = hmr2_absolute[h_id]


                # Define the Parent-Child relationships for the limbs
                parent_map = {
                    7: 5,   # L Elbow relies on L Shoulder
                    9: 7,   # L Wrist relies on L Elbow
                    8: 6,   # R Elbow relies on R Shoulder
                    10: 8,  # R Wrist relies on R Elbow
                }

                # Define the strict resolution order (Core -> Limbs)
                resolution_order = [
                    0, 5, 6, 11, 12, # Resolve Head, Shoulders, and Hips first
                    7, 8,            # Then Elbows
                    9, 10            # Then Wrists
                ]

                for yolo_id in resolution_order:
                    # Skip if the joint isn't in our active tracking map
                    if yolo_id not in YOLO_TO_HMR2:
                        continue
                        
                    hmr2_id = YOLO_TO_HMR2[yolo_id]
                    yolo_weight = self.yolo_blend_weight(yolo_trust.get(yolo_id, 0.0))

                    if yolo_id in parent_map:
                        parent_id = parent_map[yolo_id]
                        parent_hmr2_id = YOLO_TO_HMR2[parent_id]
                        hmr2_bone_vector = hmr2_absolute[hmr2_id] - hmr2_absolute[parent_hmr2_id]
                        parent_fused_pos = raw_fused_positions.get(parent_id, hmr2_absolute[parent_hmr2_id])
                        hmr2_raw_3d = parent_fused_pos + hmr2_bone_vector
                    else:
                        hmr2_raw_3d = hmr2_absolute[hmr2_id]

                    if yolo_id in current_3d and yolo_weight >= 1.0:
                        raw_3d = current_3d[yolo_id]
                        fused_sources[yolo_id] = "YOLO"
                    elif yolo_id in current_3d and yolo_weight > 0.0:
                        raw_3d = (yolo_weight * current_3d[yolo_id]) + ((1.0 - yolo_weight) * hmr2_raw_3d)
                        fused_sources[yolo_id] = "BLEND"
                    else:
                        raw_3d = hmr2_raw_3d
                        fused_sources[yolo_id] = "HMR2"

                    filter_input_3d = self.smooth_source_transition(
                        yolo_id,
                        fused_sources[yolo_id],
                        raw_3d,
                        current_time,
                    )

                    # Save for the next joint in the chain to reference
                    raw_fused_positions[yolo_id] = filter_input_3d   

                    # Low pass filtering
                    if yolo_id not in self.filters_3d:
                        self.filters_3d[yolo_id] = OneEuroFilter(
                            current_time,
                            filter_input_3d,
                            min_cutoff=self.filter_min_cutoff,
                            beta=self.filter_beta,
                        )
                        
                    smooth_3d = self.filters_3d[yolo_id](current_time, filter_input_3d)
                    fused_3d_dict[yolo_id] = smooth_3d
                    self.prev_sources[yolo_id] = fused_sources[yolo_id]

                # 2D Visualization
                draw_points = {}

                if tight_bbox is not None:
                    cv2.rectangle(display_frame, (int(tight_bbox[0]), int(tight_bbox[1])), (int(tight_bbox[2]), int(tight_bbox[3])), (0, 255, 0), 2)
                
                for yolo_id, pt3d in fused_3d_dict.items():
                    if pt3d[2] > 0.1:
                        # Pinhole projection back to 2D
                        u = int((pt3d[0] * fx / pt3d[2]) + cx_cam)
                        v = int((pt3d[1] * fy / pt3d[2]) + cy_cam)
                        draw_points[yolo_id] = (u, v)
                        
                        if fused_sources[yolo_id] == "YOLO":
                            color = (0, 255, 0)
                        elif fused_sources[yolo_id] == "BLEND":
                            color = (0, 165, 255)
                        else:
                            color = (255, 0, 255)
                        cv2.circle(display_frame, (u, v), 4, color, -1)
                        kx, ky, conf = kpts[yolo_id]
                        # cv2.putText(display_frame, f"conf: {conf:.2f}", (u, v-20), 
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if self.occlusion_reason.get(yolo_id):
                            cv2.putText(display_frame, f"{self.occlusion_reason[yolo_id]}", (u, v-20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Virtual Neck Processing
                # if 5 in fused_3d_dict and 6 in fused_3d_dict and 5 in draw_points and 6 in draw_points:
                #     neck3d = (fused_3d_dict[5] + fused_3d_dict[6]) / 2.0
                #     nx = int((draw_points[5][0] + draw_points[6][0]) / 2)
                #     ny = int((draw_points[5][1] + draw_points[6][1]) / 2)
                #     draw_points[17] = (nx, ny)
                #     fused_3d_dict[17] = neck3d

                for partA, partB in YOLO_EDGES:
                    if partA in draw_points and partB in draw_points:
                        cv2.line(display_frame, draw_points[partA], draw_points[partB], (255, 255, 255), 2)

                # # Draw Volumetric Capsules
                # for start, end, r_m, color in HUMAN_CAPSULES:
                #     if start in draw_points and end in draw_points:
                #         p1, p2 = draw_points[start], draw_points[end]
                #         z_avg = (fused_3d_dict[start][2] + fused_3d_dict[end][2]) / 2
                #         thickness = get_pixel_radius(z_avg, r_m, self.cam_param.fx)
                        
                #         cv2.line(capsule_overlay, p1, p2, color, thickness)
                #         cv2.circle(capsule_overlay, p1, thickness // 2, color, -1)
                #         cv2.circle(capsule_overlay, p2, thickness // 2, color, -1)

                # Draw target pos
                if self.latest_ee_target_3d is not None:
                    sx, sy, sz = self.latest_ee_target_3d
                    if sz > 0.05:                        
                        u = int((sx * self.cam_param.fx / sz) + self.cam_param.cx)
                        v = int((sy * self.cam_param.fy / sz) + self.cam_param.cy)
                        cv2.circle(display_frame, (u, v), radius=10, color=(0, 255, 0), thickness=-1)
                
                # Draw Robot Spheres
                # if hasattr(self, 'robot_masker'):            
                #     for sphere in self.robot_masker.robot_spheres:
                #         sx, sy, sz, r = sphere
                #         if sz > 0.05: 
                #             u = int((sx * self.cam_param.fx / sz) + self.cam_param.cx)
                #             v = int((sy * self.cam_param.fy / sz) + self.cam_param.cy)
                #             pix_radius = get_pixel_radius(sz, r, self.cam_param.fx)
                #             cv2.circle(capsule_overlay, (u, v), pix_radius, (255, 120, 0), -1)
                #             cv2.circle(display_frame, (u, v), pix_radius, (255, 120, 0), 1)

        cv2.addWeighted(capsule_overlay, 0.4, display_frame, 1.0, 0, display_frame)

        if self.record_stream:
            if self.video_writer is None:
                h, w = display_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID') 
                path = f"/recordings/tracker_{datetime.now().strftime('%H%M%S')}.avi"
                self.video_writer = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
                self.get_logger().info(f"Started recording: {path}")

            if display_frame.shape[2] == 4:
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGRA2BGR)

            self.video_writer.write(display_frame)

        if self.live_stream:
            cv2.imshow("Fused Tracker", display_frame)
            cv2.waitKey(1)

        return fused_3d_dict, pure_yolo_3d, pure_hmr_3d, fused_sources

    def destroy_node(self):
        """Native ROS 2 override to clean up hardware and files."""
        self.get_logger().info("Closing VideoWriter and ZED Camera...")
        
        if hasattr(self, 'video_writer') and self.video_writer:
            self.video_writer.release()
            
        if hasattr(self, 'zed') and self.zed.is_opened():
            self.zed.close()
            
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass 
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
