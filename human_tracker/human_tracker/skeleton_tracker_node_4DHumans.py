import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header, Float32MultiArray

import pyzed.sl as sl
import cv2
import numpy as np
import torch
import math
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as T

from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT

# Mapping yolo pose to hmr2 format (we only use the joints that are common to both)
YOLO_TO_HMR2 = {
    0: 0,   # Nose
    5: 5,   # L Shoulder
    6: 2,   # R Shoulder
    7: 6,   # L Elbow
    8: 3,   # R Elbow
    9: 7,   # L Wrist
    10: 4,  # R Wrist
    11: 12, # L Hip
    12: 9,  # R Hip
}

YOLO_EDGES = [
    (0, 5), (0, 6), (5, 6),     # Head to shoulders
    (5, 7), (7, 9),             # L Arm
    (6, 8), (8, 10),            # R Arm
    (5, 11), (6, 12), (11, 12), # Torso
]

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

class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")

        self.use_auto_calibration = True 
        self.is_calibrated = not self.use_auto_calibration
        self.beta_is_calibrated = False
        self.joints_3d_prev = {}
        self.target_lengths = {}
        self.occlusion_reason = {}
        self.prev_sources = {}

        # Initialize YOLO
        self.model = YOLO("yolo26m-pose.pt")

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
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        # init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        init.coordinate_units = sl.UNIT.METER
        init.camera_fps = 30
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

        self.tracker_pub = self.create_publisher(MarkerArray, "tracker_centroids", qos_profile)
        self.yolo_pub = self.create_publisher(MarkerArray, "tracker_yolo", qos_profile_rviz)
        self.hmr_pub = self.create_publisher(MarkerArray, "tracker_hmr", qos_profile_rviz)

        self.sphere_sub = self.create_subscription(
            Float32MultiArray, 
            "franka/robot_spheres", 
            self.sphere_callback, 
            qos_profile
        )

        self.freq = 30 
        self.timer = self.create_timer(1/self.freq, self.publish_tracker_state)

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
            marker.ns = f"{ns_prefix}_{self.get_skeleton_map(yolo_id)}" # use this only for debugging, in real use we want them all in the same namespace
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
                # Fused coloring: Green for YOLO, Magenta for HMR2
                if sources.get(yolo_id) == "YOLO":
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0, 0.0, 1.0
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

    def get_pixel_radius(self, z_depth, radius_m, focal_length):
        if z_depth <= 0: return 1
        return int((radius_m * focal_length) / z_depth)

    def publish_tracker_state(self):
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

        # Pure YOLO (Left, offset +1.0 meter, Green color)
        yolo_msg = self.build_marker_array(pure_yolo_3d, "yolo", offset_y=1.0, color=(0.0, 1.0, 0.0))
        self.yolo_pub.publish(yolo_msg)

        # Pure HMR2 (Right, offset -1.0 meter, Magenta color)
        hmr_msg = self.build_marker_array(pure_hmr_3d, "hmr2", offset_y=-1.0, color=(1.0, 0.0, 1.0))
        self.hmr_pub.publish(hmr_msg)

    def sphere_callback(self, msg: Float32MultiArray):
        flat_data = np.array(msg.data)
        if len(flat_data) > 0:
            self.robot_spheres = flat_data.reshape(-1, 4)

        if self.is_calibrated and hasattr(self, 'T_C_to_W'):
            if not hasattr(self, 'robot_masker'):
                self.robot_masker = RobotMask(self.T_C_to_W)
            self.robot_masker.update_robot_spheres(self.robot_spheres)

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
        # if len(results[0].boxes.data) == 0:
        #     cv2.imshow("Fused Tracker", display_frame)
        #     cv2.waitKey(1)
        #     return None
        
        if len(results[0].boxes.data) == 0:
            # No human detected! Skip the 4D Humans processing for this frame
            return
        tight_bbox = results[0].boxes.data[0].cpu().numpy()[:4]
        kpts = results[0].keypoints.data[0].cpu().numpy()
        
        candidate_3d = {}
        candidate_2d = {}
        yolo_confidence = {}
        self.occlusion_reason = {}  # Reset occlusion reasons each frame
        
        for yolo_id in YOLO_TO_HMR2.keys():
            kx, ky, conf = kpts[yolo_id]
            yolo_confidence[yolo_id] = conf

            if conf > 0.95:
                err, pt3d = self.point_cloud.get_value(int(kx), int(ky))
                if err == sl.ERROR_CODE.SUCCESS and np.isfinite(pt3d[2]) and pt3d[2] > 0:
                    candidate_3d[yolo_id] = pt3d[:3]
                    candidate_2d[yolo_id] = (int(kx), int(ky))
                else:
                    self.occlusion_reason[yolo_id] = "no_depth"
            else:
                self.occlusion_reason[yolo_id] = "low_conf"

        pure_yolo_3d = {k: v for k, v in candidate_3d.items()} # save for debugging/visualization

        for yolo_id in list(candidate_3d.keys()):
            pt3d = candidate_3d[yolo_id]
            # Robot Sphere Check
            if hasattr(self, 'robot_masker'):
                    if self.robot_masker.is_joint_occluded(pt3d, yolo_id):
                        del candidate_3d[yolo_id]
                        self.occlusion_reason[yolo_id] = "robot"

        # OVERLAP CHECK
        # Shoulders (IDs 5 and 6)
        if 5 in candidate_3d and 6 in candidate_3d:
            shoulder_width = np.linalg.norm(candidate_3d[5] - candidate_3d[6])
            if shoulder_width < 0.25: 
                err_5 = np.linalg.norm(candidate_3d[5] - self.filters_3d[5].x_prev) if 5 in self.filters_3d else candidate_3d[5][2]
                err_6 = np.linalg.norm(candidate_3d[6] - self.filters_3d[6].x_prev) if 6 in self.filters_3d else candidate_3d[6][2]
                
                # The joint that deviates more from its history (or is farther if no history) is likely the hallucination
                if err_5 > err_6:
                    del candidate_3d[5]
                    self.occlusion_reason[5] = "overlap"
                else:
                    del candidate_3d[6]
                    self.occlusion_reason[6] = "overlap"

        # Hips (IDs 11 and 12)
        if 11 in candidate_3d and 12 in candidate_3d:
            hip_width = np.linalg.norm(candidate_3d[11] - candidate_3d[12])
            if hip_width < 0.15: 
                err_11 = np.linalg.norm(candidate_3d[11] - self.filters_3d[11].x_prev) if 11 in self.filters_3d else candidate_3d[11][2]
                err_12 = np.linalg.norm(candidate_3d[12] - self.filters_3d[12].x_prev) if 12 in self.filters_3d else candidate_3d[12][2]

                # The joint that deviates more from its history (or is farther if no history) is likely the hallucination
                if err_11 > err_12:
                    del candidate_3d[11]
                    self.occlusion_reason[11] = "overlap"
                else:
                    del candidate_3d[12]
                    self.occlusion_reason[12] = "overlap"

        # Wrists (IDs 9 and 10)
        if 9 in candidate_3d and 10 in candidate_3d:
            wrist_dist = np.linalg.norm(candidate_3d[9] - candidate_3d[10])
            if wrist_dist < 0.15: # Set threshold for wrist overlap in meters
                if 9 in self.filters_3d and 10 in self.filters_3d:
                    err_9 = np.linalg.norm(candidate_3d[9] - self.filters_3d[9].x_prev)
                    err_10 = np.linalg.norm(candidate_3d[10] - self.filters_3d[10].x_prev)
                    
                    # The joint that deviates more from its history is likely the hallucination
                    if err_9 > err_10:
                        del candidate_3d[9]
                        self.occlusion_reason[9] = "overlap"
                    else:
                        del candidate_3d[10]
                        self.occlusion_reason[10] = "overlap"

                    # If no history, use current depth (the farther one is likely the hallucination)
                else: 
                    if candidate_3d[9][2] > candidate_3d[10][2]:
                        del candidate_3d[9]
                        self.occlusion_reason[9] = "overlap"
                    else:
                        del candidate_3d[10]
                        self.occlusion_reason[10] = "overlap"


        # TEMPORAL Check
        current_3d = {}
        current_2d = {}

        # Loop through all joints that have survived the overlap check
        for yolo_id, pt3d in candidate_3d.items():
            is_valid = True

            # if yolo_id in self.filters_3d:
            #     prev_pt3d = self.filters_3d[yolo_id].x_prev
            #     t_prev = self.filters_3d[yolo_id].t_prev
            #     dt = current_time - t_prev
                
            #     if dt > 0:
            #         dist = np.linalg.norm(pt3d - prev_pt3d)
                    
            #         max_allowed_jump = (1.0 * dt) + 0.10
            #         if dt > 1.0: 
            #             max_allowed_jump = 0.5
                        
            #         # --- NEW: OCCLUSION RECOVERY ---
            #         # If it was hallucinated by HMR2 last frame, allow a massive 
            #         # jump so it can snap back to the true YOLO measurement.
            #         if self.prev_sources.get(yolo_id) == "HMR2":
            #             max_allowed_jump = 1.0  # Allow a 1-meter teleport
            #         # -------------------------------
                        
            #         if dist > max_allowed_jump:
            #             is_valid = False
            #             self.occlusion_reason[yolo_id] = "temporal"
            
            # If it survived the velocity check, it is a trusted ZED measurement
            if is_valid:
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
                true_bone_length = self.target_lengths[(parent, child)] 

                # If the ZED measurement is off by more than 20%, one joint is hallucinating
                if abs(zed_bone_length - true_bone_length) > 0.2*true_bone_length:

                    # Look at history to find which one is hallucinating
                    err_parent = 0.0
                    err_child = 0.0
                    
                    if parent in self.filters_3d:
                        err_parent = np.linalg.norm(current_3d[parent] - self.filters_3d[parent].x_prev)
                        
                    if child in self.filters_3d:
                        err_child = np.linalg.norm(current_3d[child] - self.filters_3d[child].x_prev)
                    
                      # The joint with the highest error gets removed
                    if err_parent > err_child:
                        del current_3d[parent]
                        self.occlusion_reason[parent] = "length"
                        if parent in current_2d:
                            del current_2d[parent]
                    else:
                        del current_3d[child]
                        self.occlusion_reason[child] = "length"
                        if child in current_2d:
                            del current_2d[child]

        # HMR2 Inference
        x1, y1, x2, y2 = tight_bbox
        # Need to define center of the crop for HMR2 to later undistort image
        core_x, core_y = [], []
        # Look for Shoulders (5, 6) and Hips (11, 12)
        for target_id in [5, 6, 11, 12]:
            # Can use candidate 2d because even if the 3D is filtered out, the 2D keypoint can still be visible and useful for centering the crop
            if target_id in candidate_2d: 
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
                    if y_id in current_3d: # At this point, current_3d consists of non occupied Yolo/ZED joint measurements
                        h_id = YOLO_TO_HMR2[y_id]
                        yolo_anchors.append(current_3d[y_id])
                        hmr2_anchors.append(joints_3d[h_id])
                
                if len(yolo_anchors) > 0:
                    yolo_centroid = np.mean(yolo_anchors, axis=0)
                    hmr2_centroid = np.mean(hmr2_anchors, axis=0)
                    translation_offset = yolo_centroid - hmr2_centroid
                    hmr2_absolute = joints_3d + translation_offset
                else:
                    # Fallback just in case the camera is completely blocked
                    hmr2_absolute = joints_3d

                pure_hmr_3d = {} # for debugging/visualization
                for y_id, h_id in YOLO_TO_HMR2.items():
                    pure_hmr_3d[y_id] = hmr2_absolute[h_id]

                # Fusion of YOLO and HMR2
                # ---------------------------------------------------------
                # Fusion of YOLO and HMR2 (Forward Kinematics Approach)
                # ---------------------------------------------------------
                fused_3d_dict = {}
                fused_sources = {}
                raw_fused_positions = {} # Temporarily store pre-filtered positions for chain math

                # 1. Define the Parent-Child relationships for the limbs
                parent_map = {
                    7: 5,   # L Elbow relies on L Shoulder
                    9: 7,   # L Wrist relies on L Elbow
                    8: 6,   # R Elbow relies on R Shoulder
                    10: 8,  # R Wrist relies on R Elbow
                }

                # 2. Define the strict resolution order (Core -> Limbs)
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
                    raw_3d = None
                    
                    # CASE A: Joint is visible to ZED/YOLO
                    if yolo_id in current_3d:
                        raw_3d = current_3d[yolo_id]
                        fused_sources[yolo_id] = "YOLO"
                        
                    # CASE B: Joint is occluded. Rely on HMR2.
                    else:
                        fused_sources[yolo_id] = "HMR2"
                        
                        # If it is a limb joint, use Forward Kinematics (relative to parent)
                        if yolo_id in parent_map:
                            parent_id = parent_map[yolo_id]
                            parent_hmr2_id = YOLO_TO_HMR2[parent_id]
                            
                            # Calculate the 3D bone vector from HMR2's perspective
                            hmr2_bone_vector = hmr2_absolute[hmr2_id] - hmr2_absolute[parent_hmr2_id]
                            
                            # Attach that vector to wherever the parent ended up in our fused reality
                            parent_fused_pos = raw_fused_positions.get(parent_id, hmr2_absolute[parent_hmr2_id])
                            raw_3d = parent_fused_pos + hmr2_bone_vector
                            
                        # If it is a core joint, just use the absolute aligned HMR2 position
                        else:
                            raw_3d = hmr2_absolute[hmr2_id]

                    # Save for the next joint in the chain to reference
                    raw_fused_positions[yolo_id] = raw_3d   

                    # Low pass filtering
                    if yolo_id not in self.filters_3d:
                        # min_cutoff: smoothness when human stationary, beta determines how much filter opens up as velocity increases
                        self.filters_3d[yolo_id] = OneEuroFilter(current_time, raw_3d, min_cutoff=0.01, beta=3)
                        
                    smooth_3d = self.filters_3d[yolo_id](current_time, raw_3d)
                    fused_3d_dict[yolo_id] = smooth_3d

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
                        
                        color = (0, 255, 0) if fused_sources[yolo_id] == "YOLO" else (255, 0, 255)
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
                #         thickness = self.get_pixel_radius(z_avg, r_m, self.cam_param.fx)
                        
                #         cv2.line(capsule_overlay, p1, p2, color, thickness)
                #         cv2.circle(capsule_overlay, p1, thickness // 2, color, -1)
                #         cv2.circle(capsule_overlay, p2, thickness // 2, color, -1)

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

        cv2.addWeighted(capsule_overlay, 0.4, display_frame, 1.0, 0, display_frame)
        # cv2.imshow("Fused Tracker", display_frame)
        # cv2.waitKey(1)

        self.prev_sources = fused_sources

        return fused_3d_dict, pure_yolo_3d, pure_hmr_3d, fused_sources

    def get_skeleton_map(self, id):
        mapping = {0:"nose", 5:"left shoulder", 6:"right shoulder", 7:"left elbow", 8:"right elbow", 
                   9:"left wrist", 10:"right wrist", 11:"left hip", 12:"right hip", 
                   13:"left knee", 14:"right knee", 15:"left ankle", 16:"right ankle"}
        return mapping.get(id, "unknown")

class OneEuroFilter:
    """ Implements the One Euro Filter for smoothing 3D joint positions over time."""
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
        if t_e <= 0.0: return x
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

    def is_joint_occluded(self, joint_pos, yolo_id):
        for i, sphere in enumerate(self.robot_spheres):
            self.yolo_id = yolo_id # for debugging
            self.i = i # for debugging
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

            return proj_length < (r + 0.05) and np.linalg.norm(joint_pos) > np.linalg.norm(sphere_center) - 0.1 # Add small buffer

        return False

def main(args=None):
    rclpy.init(args=None)
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()  

if __name__ == '__main__':
    main()