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

def get_skeleton_map(id):
    mapping = {0:"nose", 5:"left shoulder", 6:"right shoulder", 7:"left elbow", 8:"right elbow", 
                9:"left wrist", 10:"right wrist", 11:"left hip", 12:"right hip", 
                13:"left knee", 14:"right knee", 15:"left ankle", 16:"right ankle"}
    return mapping.get(id, "unknown")