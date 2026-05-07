import numpy as np

def get_pixel_radius(z_depth, radius_m, focal_length):
        if z_depth <= 0: return 1
        return int((radius_m * focal_length) / z_depth)

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
        return self.occlusion_score(joint_pos) > 0.5

    def visibility_score(self, joint_pos):
        return 1.0 - self.occlusion_score(joint_pos)

    def occlusion_score(self, joint_pos):
        if np.linalg.norm(joint_pos) <= 1e-6:
            return 1.0

        max_score = 0.0
        for sphere in self.robot_spheres:
            max_score = max(max_score, self.check_sphere_occlusion_score(sphere, joint_pos))
        return max_score
    
    def check_sphere_occlusion(self, sphere, joint_pos):
        return self.check_sphere_occlusion_score(sphere, joint_pos) > 0.5

    def check_sphere_occlusion_score(self, sphere, joint_pos):
        sx, sy, sz, r = sphere
        sphere_center = np.array([sx, sy, sz])
        joint_dist = np.linalg.norm(joint_pos)

        if joint_dist <= 1e-6:
            return 1.0

        joint_dir = joint_pos / joint_dist
        sphere_depth_on_ray = np.dot(sphere_center, joint_dir)

        if sphere_depth_on_ray <= 0.0 or sphere_depth_on_ray >= joint_dist:
            return 0.0

        closest_point_on_ray = sphere_depth_on_ray * joint_dir
        ray_distance = np.linalg.norm(closest_point_on_ray - sphere_center)
        hard_radius = r + 0.02
        soft_radius = r + max(0.05, 0.75 * r)

        if ray_distance <= hard_radius:
            return 1.0
        if ray_distance >= soft_radius:
            return 0.0

        return 1.0 - ((ray_distance - hard_radius) / (soft_radius - hard_radius))
