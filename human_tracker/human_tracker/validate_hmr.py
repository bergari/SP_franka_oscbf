import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
import numpy as np
import csv
import os
from datetime import datetime

from .constants import YOLO_TO_HMR2

class ValidationNode(Node):
    def __init__(self):
        super().__init__('validation_node')

        self.seen_cam1 = False
        self.seen_cam2 = False
        self.required_joint_ids = sorted(YOLO_TO_HMR2.keys())
        self.quota_buckets = (
            "CAM1_HMR2_CAM2_YOLO",
            "CAM1_YOLO_CAM2_HMR2",
            "YOLO_vs_YOLO",
        )
        self.bucket_targets = {
            "CAM1_HMR2_CAM2_YOLO": 500,
            "CAM1_YOLO_CAM2_HMR2": 500,
            "YOLO_vs_YOLO": 1000,
        }
        self.sample_counts = {
            bucket: {joint_id: 0 for joint_id in self.required_joint_ids}
            for bucket in self.quota_buckets
        }
        self.collection_complete = False
        self.last_progress_log = 0
        
        # Using a raw integer '10' for QoS forces ROS 2 to use the most permissive, 
        # reliable default profile. This bypasses any strict policy mismatches.
        self.cam1_sub = self.create_subscription(MarkerArray, '/cam1/tracker_centroids', self.cam1_cb, 10)
        self.cam2_sub = self.create_subscription(MarkerArray, '/cam2/tracker_centroids', self.cam2_cb, 10)

        self.latest_msg1 = None
        self.latest_msg2 = None
        
        # Setup CSV logging
        os.makedirs('/recordings', exist_ok=True)
        self.csv_path = f"/recordings/hmr_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp",
                "Joint_ID",
                "Cam1_Source",
                "Cam2_Source",
                "Cam1_X_m",
                "Cam1_Y_m",
                "Cam1_Z_m",
                "Cam2_X_m",
                "Cam2_Y_m",
                "Cam2_Z_m",
                "Error_X_m",
                "Error_Y_m",
                "Error_Z_m",
                "Error_m",
                "Error_Mode",
                "Error_Bucket",
            ])
            
        self.get_logger().info("Validation Node Started. Waiting for data streams...")
        self.get_logger().info(f"Saving data to: {self.csv_path}")
        self.get_logger().info(
            "Collecting per joint: 500 cam1-occluded, 500 cam2-occluded, "
            "and 1000 unoccluded samples."
        )

        # Compare at 20Hz
        self.timer = self.create_timer(0.05, self.compare_timer_cb) 

    def quota_is_full(self, bucket, joint_id):
        return self.sample_counts[bucket][joint_id] >= self.bucket_targets[bucket]

    def collection_is_complete(self):
        return all(
            self.sample_counts[bucket][joint_id] >= self.bucket_targets[bucket]
            for bucket in self.quota_buckets
            for joint_id in self.required_joint_ids
        )

    def total_samples_collected(self):
        return sum(
            self.sample_counts[bucket][joint_id]
            for bucket in self.quota_buckets
            for joint_id in self.required_joint_ids
        )

    def target_total_samples(self):
        return sum(self.bucket_targets.values()) * len(self.required_joint_ids)

    def format_unfinished_joint_progress(self):
        unfinished = []

        for joint_id in self.required_joint_ids:
            bucket_parts = []
            joint_total = 0
            joint_target = 0

            for bucket in self.quota_buckets:
                count = self.sample_counts[bucket][joint_id]
                target = self.bucket_targets[bucket]
                joint_total += count
                joint_target += target

                if count < target:
                    label = {
                        "CAM1_HMR2_CAM2_YOLO": "cam1_occ",
                        "CAM1_YOLO_CAM2_HMR2": "cam2_occ",
                        "YOLO_vs_YOLO": "unocc",
                    }[bucket]
                    bucket_parts.append(f"{label} {count}/{target}")

            if bucket_parts:
                joint_name = {
                    0: "nose",
                    5: "L shoulder",
                    6: "R shoulder",
                    7: "L elbow",
                    8: "R elbow",
                    9: "L wrist",
                    10: "R wrist",
                    11: "L hip",
                    12: "R hip",
                }.get(joint_id, f"ID {joint_id}")
                unfinished.append(
                    f"{joint_name} {joint_total}/{joint_target} ({', '.join(bucket_parts)})"
                )

        return "; ".join(unfinished)

    def log_progress(self):
        total = self.total_samples_collected()
        target_total = self.target_total_samples()

        if total - self.last_progress_log < 100:
            return

        self.last_progress_log = total
        unfinished_progress = self.format_unfinished_joint_progress()
        if not unfinished_progress:
            unfinished_progress = "all joints complete"

        self.get_logger().info(
            f"Validation progress: {total}/{target_total} rows | "
            f"unfinished: {unfinished_progress}"
        )

    def cam1_cb(self, msg):
        self.latest_msg1 = msg
        if not self.seen_cam1:
            self.get_logger().info("✅ Successfully receiving data from Cam 1!")
            self.seen_cam1 = True
        
    def cam2_cb(self, msg):
        self.latest_msg2 = msg
        if not self.seen_cam2:
            self.get_logger().info("✅ Successfully receiving data from Cam 2!")
            self.seen_cam2 = True

    def compare_timer_cb(self):
        if self.collection_complete:
            return

        if not self.latest_msg1:
            return # Still waiting for cam 1
        if not self.latest_msg2:
            return # Still waiting for cam 2
            
        msg1 = self.latest_msg1
        msg2 = self.latest_msg2
        
        # Clear them so we don't compare stale data repeatedly
        self.latest_msg1 = None
        self.latest_msg2 = None

        joints1, joints2 = {}, {}
        sources1, sources2 = {}, {}
        
        for m in msg1.markers:
            if m.type == 2:  # 2 is Marker.SPHERE
                joints1[m.id] = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])
                sources1[m.id] = m.text if m.text else "UNKNOWN"

        for m in msg2.markers:
            if m.type == 2: 
                joints2[m.id] = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])
                sources2[m.id] = m.text if m.text else "UNKNOWN"

        common_ids = set(joints1.keys()).intersection(set(joints2.keys()))
        
        # Debugging step: Log exactly how many joints it is matching
        if not common_ids:
            self.get_logger().warn(f"Warning: Messages received, but 0 matching joint IDs found!")
            return
        else:
            self.get_logger().debug(f"Comparing frames... Found {len(common_ids)} common joints")
        
        occlusion_errors = []
        rows_written = 0
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            timestamp = self.get_clock().now().nanoseconds / 1e9
            
            for j_id in sorted(common_ids):
                if j_id not in self.required_joint_ids:
                    continue

                source1 = sources1.get(j_id)
                source2 = sources2.get(j_id)
                p1 = joints1[j_id]
                p2 = joints2[j_id]
                error_mode = "OTHER"
                error_bucket = "OTHER"
                error_vec = p1 - p2
                
                if source1 == "HMR2" and source2 == "YOLO":
                    error_mode = "HMR2_vs_YOLO"
                    error_bucket = "CAM1_HMR2_CAM2_YOLO"
                    error_vec = p1 - p2
                elif source1 == "YOLO" and source2 == "HMR2":
                    error_mode = "HMR2_vs_YOLO"
                    error_bucket = "CAM1_YOLO_CAM2_HMR2"
                    error_vec = p2 - p1
                elif source1 == "YOLO" and source2 == "YOLO":
                    error_mode = "YOLO_vs_YOLO"
                    error_bucket = "YOLO_vs_YOLO"

                if error_bucket not in self.quota_buckets:
                    continue

                if self.quota_is_full(error_bucket, j_id):
                    continue

                dist = np.linalg.norm(error_vec)
                
                if error_mode == "HMR2_vs_YOLO":
                    occlusion_errors.append(dist)
                
                writer.writerow([
                    timestamp,
                    j_id,
                    source1,
                    source2,
                    p1[0],
                    p1[1],
                    p1[2],
                    p2[0],
                    p2[1],
                    p2[2],
                    error_vec[0],
                    error_vec[1],
                    error_vec[2],
                    dist,
                    error_mode,
                    error_bucket,
                ])
                self.sample_counts[error_bucket][j_id] += 1
                rows_written += 1
            
        if rows_written > 0:
            self.get_logger().info(f"Wrote {rows_written} quota-limited joint comparisons to CSV.", once=True)
            self.log_progress()
            
        if occlusion_errors:
            mean_error = np.mean(occlusion_errors)
            self.get_logger().info(f"HMR2 Guess Error (MPJPE): {mean_error*100:.2f} cm | Evaluated Joints: {len(occlusion_errors)}")

        if self.collection_is_complete():
            self.collection_complete = True
            self.timer.cancel()
            self.get_logger().info("Validation sample target reached for every joint. Shutting down validation node.")
            rclpy.shutdown()


def main():
    rclpy.init()
    node = ValidationNode()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    if rclpy.ok():
        rclpy.shutdown()
