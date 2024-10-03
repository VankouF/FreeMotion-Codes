import numpy as np
from utils.utils import *

FPS = 30

def load_motion(file_path, min_length, swap=False):

    try:
        motion = np.load(file_path).astype(np.float32)
    except:
        print("error: ", file_path)
        return None, None
    
    motion1 = motion[:, :22 * 3]  # 22*3 表示的是每个关节的position
    motion2 = motion[:, 62 * 3:62 * 3 + 21 * 6]  # 21*6 表示的6D旋转表示
    motion = np.concatenate([motion1, motion2], axis=1)

    if motion.shape[0] < min_length:
        return None, None
    if swap:
        motion_swap = swap_left_right(motion, 22)
    else:
        motion_swap = None
    return motion, motion_swap