import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header


import pyzed.sl as sl
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

 # Skeleton Connection Map (Pairs of indices to connect)

# 0. Nose
# 1. Left Eye
# 2. Right Eye
# 3. Left Ear
# 4. Right Ear
# 5. Left Shoulder
# 6. Right Shoulder
# 7. Left Elbow
# 8. Right Elbow
# 9. Left Wrist
# 10. Right Wrist
# 11. Left Hip
# 12. Right Hip
# 13. Left Knee
# 14. Right Knee
# 15. Left Ankle
# 16. Right Ankle 

# Capsule Map for CBF: (Start_Joint, End_Joint, Radius_in_Meters, Color_BGR)
HUMAN_CAPSULES = [
    (17, 0, 0.18, (255, 0, 0)),   # Head (Neck to Nose)
    (5, 7, 0.12, (0, 255, 0)),    # L Upper Arm
    (7, 9, 0.08, (0, 255, 0)),    # L Forearm
    (6, 8, 0.12, (0, 255, 255)),  # R Upper Arm
    (8, 10, 0.08, (0, 255, 255)), # R Forearm
    (17, 11, 0.22, (255, 255, 0)),# Torso L
    (17, 12, 0.22, (255, 255, 0)),# Torso R
#    (11, 13, 0.15, (0, 0, 255)),  # L Thigh
#    (13, 15, 0.12, (0, 0, 255)),  # L Shin
#    (12, 14, 0.15, (255, 0, 255)),# R Thigh
#    (14, 16, 0.12, (255, 0, 255)), # R Shin
    (9, 9, 0.1, (0, 255, 0)),      # L Hand
    (10, 10, 0.1, (0, 255, 0))    # R Hand
]

# Standard Skeletal Connections (The thin lines)
SKELETON_LINES = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Face
    (17, 5), (17, 6), (17, 0),          # Neck connections
    (5, 7), (7, 9), (6, 8), (8, 10),    # Arms
    (5, 11), (6, 12), (11, 12),         # Torso
#    (11, 13), (13, 15), (12, 14), (14, 16) # Legs
]

class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")

        self.model = YOLO("yolo26l-pose.pt")

        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD1080
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER
        init.camera_fps = 60
        if self.zed.open(init) != sl.ERROR_CODE.SUCCESS:
            print("ZED failed to open.")
            exit()

        self.cam_param = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self.history = {i: deque(maxlen=5) for i in range(18)}

        self.image_sl = sl.Mat()
        self.point_cloud = sl.Mat()

        self.freq = 10
        self.timer = self.create_timer(1/self.freq, self.publish_tracker_state)
        
        # TODO: Define the Message Type and Topic Name for ellipsoid centroids
        self.tracker_pub = self.create_publisher(MarkerArray, "tracker_centroids", 10)

    def publish_tracker_state(self):
        time = self.get_clock().now()
        secs, nanosecs = time.seconds_nanoseconds()
        t = secs + nanosecs / 1e9

        res = self.get_centroids()
        if res is None:
            return
            
        centroids_x, centroids_y, centroids_z = res

        # Need to create Message Object! Cannot publish centroids as arrays directly
        marker_array = MarkerArray()

        # Turn coordinates into Points/Markers
        for i in range(len(centroids_x)):
            marker = Marker()
            marker.header.frame_id = "camera_link" # Should match TF tree
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Set the position
            marker.pose.position.x = float(centroids_x[i])
            marker.pose.position.y = float(centroids_y[i])
            marker.pose.position.z = float(centroids_z[i])
            
            # Scale and Color (REQUIRED for visibility in RViZ)
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0 
            marker.color.g = 1.0 # Green spheres
            
            marker_array.markers.append(marker)

        self.tracker_pub.publish(marker_array)
        

    def get_pixel_radius(self,z_depth, radius_m, focal_length):
        if z_depth <= 0: return 1
        return int((radius_m * focal_length) / z_depth)

    def get_centroids(self):
        centroids_x = []
        centroids_y = []
        centroids_z = []   

        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_sl, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
        
            bgra_frame = self.image_sl.get_data()
            frame_rgb = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2RGB)
            
            # Layer for the volumetric ellipsoids
            capsule_overlay = np.zeros_like(bgra_frame)
            # Main frame for the skeleton and distance text
            display_frame = bgra_frame.copy()
            
            results = self.model(frame_rgb, verbose=False, conf=0.5)
            current_3d = {}
            current_2d = {}

            if len(results[0].keypoints.data) > 0:
                kpts = results[0].keypoints.data[0].cpu().numpy()
                
                # 1. Process 17 Keypoints
                for i in range(17):
                    kx, ky, conf = kpts[i]
                    if conf > 0.5:
                        err, pt3d = self.point_cloud.get_value(int(kx), int(ky))
                        if np.isfinite(pt3d[2]):
                            # Update 3D History
                            self.history[i].append(pt3d[:3])
                            
                            # SMOOTHED 3D POINT
                            avg_3d = np.mean(self.history[i], axis=0)
                            current_3d[i] = avg_3d
                            
                            centroids_x.append(avg_3d[0])
                            centroids_y.append(avg_3d[1])
                            centroids_z.append(avg_3d[2])
                            
                            # --- NEW: SMOOTHED 2D PIXELS ---
                            # Instead of using raw 'kx, ky', we project the smoothed 3D back to 2D
                            # This removes pixel-jitter from the skeleton lines
                            if np.isfinite(avg_3d[2]) and avg_3d[2] > 0:
                                u = int((avg_3d[0] * self.cam_param.fx / avg_3d[2]) + self.cam_param.cx)
                                v = int((avg_3d[1] * self.cam_param.fy / avg_3d[2]) + self.cam_param.cy)
                                current_2d[i] = (u, v)
                            else:
                                continue

                # 2. Process Virtual Neck
                if 5 in current_3d and 6 in current_3d:
                    neck3d = (current_3d[5] + current_3d[6]) / 2
                    self.history[17].append(neck3d)
                    current_3d[17] = np.mean(self.history[17], axis=0)
                    nx = int((current_2d[5][0] + current_2d[6][0]) / 2)
                    ny = int((current_2d[5][1] + current_2d[6][1]) / 2)
                    current_2d[17] = (nx, ny)

                # 3. Draw Volumetric Capsules (Background Overlay)
                for start, end, r_m, color in HUMAN_CAPSULES:
                    if start in current_2d and end in current_2d:
                        p1, p2 = current_2d[start], current_2d[end]
                        z_avg = (current_3d[start][2] + current_3d[end][2]) / 2
                        thickness = self.get_pixel_radius(z_avg, r_m, self.cam_param.fx)
                        
                        cv2.line(capsule_overlay, p1, p2, color, thickness)
                        cv2.circle(capsule_overlay, p1, thickness // 2, color, -1)
                        cv2.circle(capsule_overlay, p2, thickness // 2, color, -1)

                # 4. Draw Skeletal Connections (Foreground)
                for partA, partB in SKELETON_LINES:
                    if partA in current_2d and partB in current_2d:
                        cv2.line(display_frame, current_2d[partA], current_2d[partB], (255, 255, 255), 2)
                        cv2.circle(display_frame, current_2d[partA], 3, (0, 0, 255), -1)

                # 5. Add Distance Text
                if 17 in current_3d:
                    z_val = current_3d[17][2]
                    cv2.putText(display_frame, f"Person: {z_val:.2f}m", (current_2d[17][0], current_2d[17][1]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Blend the volumes with the skeleton frame
            cv2.addWeighted(capsule_overlay, 0.4, display_frame, 1.0, 0, display_frame)
            cv2.imshow("ZED YOLO26 OSCBF Model", display_frame)
            cv2.waitKey(1)

            return centroids_x, centroids_y, centroids_z  
        
        # finally:
        #     self.zed.close()
        #     cv2.destroyAllWindows()  

def main(args=None):
    rclpy.init(args=None)
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()  

if __name__ == '__main__':
    main()