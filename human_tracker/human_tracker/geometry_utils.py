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