import numpy as np
import torch
import random

from torch.utils import data
from tqdm import tqdm
from os.path import join as pjoin

from utils.utils import *
from utils.plot_script import *
from utils.preprocess import *


class InterHumanDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.max_cond_length = 1
        self.min_cond_length = 1
        self.max_gt_length = 300
        self.min_gt_length = 15

        self.max_length = self.max_cond_length + self.max_gt_length -1
        self.min_length = self.min_cond_length + self.min_gt_length -1

        self.motion_rep = opt.MOTION_REP
        self.data_list = []
        self.motion_dict = {}

        self.cache = opt.CACHE
        self.single_text_description = opt.SINGLE_TEXT_DESCRIPTION
        
        ignore_list = []
        try:
            ignore_list = open(os.path.join(opt.DATA_ROOT, "split/ignore_list.txt"), "r").readlines()
        except Exception as e:
            print(e)
        data_list = []
        if self.opt.MODE == "train":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "split/train.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "val":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "split/val.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "test":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "split/test.txt"), "r").readlines()
            except Exception as e:
                print(e)

        random.shuffle(data_list)
        # data_list = data_list[:70]
        
        index = 0
        for root, dirs, files in os.walk(pjoin(opt.DATA_ROOT,'motions_processed')):
           
            for file in tqdm(files):
               
                if file.endswith(".npy") and "person1" in root:
                    motion_name = file.split(".")[0]
                    if file.split(".")[0]+"\n" in ignore_list: # or int(motion_name)>1000
                        print("ignore: ", file)
                        continue
                    if file.split(".")[0]+"\n" not in data_list:
                        continue
                    file_path_person1 = pjoin(root, file)
                    file_path_person2 = pjoin(root.replace("person1", "person2"), file)
                    
                    text_path = file_path_person1.replace("motions_processed", "annots").replace("person1", "").replace("npy", "txt")
                    
                    text1_path = file_path_person1.replace("motions_processed", "separate_annots").replace("person1", "text1").replace("npy", "txt")
                    text2_path = file_path_person1.replace("motions_processed", "separate_annots").replace("person1", "text2").replace("npy", "txt")
                    
                    try:
                        texts = []
                        for idx, item in enumerate(open(text_path, "r").readlines()):
                            texts.append(item.replace("\n",""))
                    except:
                        print(idx, text_path)
                    
                    texts = [item.replace("\n", "") for item in open(text_path, "r").readlines()]
                    texts_swap = [item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                                  .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts]

                    texts1 = [item.replace("\n", "") for item in open(text1_path, "r").readlines()]
                    texts1_swap = [item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                                  .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts1]
                    
                    texts2 = [item.replace("\n", "") for item in open(text2_path, "r").readlines()]
                    texts2_swap = [item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                                  .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts2]
                    
                    motion1, motion1_swap = load_motion(file_path_person1, self.min_length, swap=True)  # [95,192]
                    motion2, motion2_swap = load_motion(file_path_person2, self.min_length, swap=True)  # [95,192]
                        
                    if motion1 is None:
                        continue
                    
                    if self.cache:
                        self.motion_dict[index] = [motion1, motion2]
                        self.motion_dict[index+1] = [motion1_swap, motion2_swap]
                    else:
                        self.motion_dict[index] = [file_path_person1, file_path_person2]
                        self.motion_dict[index + 1] = [file_path_person1, file_path_person2]

                    self.data_list.append({
                        # "idx": idx,
                        "name": motion_name,
                        "motion_id": index,
                        "swap":False,
                        "texts":texts,
                        "texts1":texts1,
                        "texts2":texts2
                    })
                    if opt.MODE == "train":
                        self.data_list.append({
                            # "idx": idx,
                            "name": motion_name+"_swap",
                            "motion_id": index+1,
                            "swap": True,
                            "texts": texts_swap,
                            "texts1":texts1_swap,
                            "texts2":texts2_swap
                        })

                    index += 2

        print("total dataset: ", len(self.data_list))

    def real_len(self):
        return len(self.data_list)

    def __len__(self):
        return self.real_len()*1

    def __getitem__(self, item):
       
        idx = item % self.real_len()
        data = self.data_list[idx]

        name = data["name"]
        motion_id = data["motion_id"]
        swap = data["swap"]
        
        num_text = len(data["texts"])
        num_choice = random.choice(list(range(num_text)))
        text, text1, text2 = data["texts"][num_choice].strip(), data["texts1"][num_choice].strip(), data["texts2"][num_choice].strip()
       
        # text = random.choice(data["texts"]).strip()
        
        if self.cache:
            full_motion1, full_motion2 = self.motion_dict[motion_id]
        else:
            file_path1, file_path2 = self.motion_dict[motion_id]
            # print(file_path1)
            # print(file_path2)
            motion1, motion1_swap = load_motion(file_path1, self.min_length, swap=swap)  # [95,192]
            motion2, motion2_swap = load_motion(file_path2, self.min_length, swap=swap)  # [95,192]
            # print(motion1.shape)
            # print(motion2.shape)
            # print(motion1_swap.shape)
            # print(motion2_swap.shape)
            if swap:
                full_motion1 = motion1_swap
                full_motion2 = motion2_swap
            else:
                full_motion1 = motion1
                full_motion2 = motion2
        try:
            length = full_motion1.shape[0]  # 95
        except:
            print(swap,file_path1)
        if length > self.max_length:  # max_length=300
            idx = random.choice(list(range(0, length - self.max_gt_length, 1)))
            gt_length = self.max_gt_length
            motion1 = full_motion1[idx:idx + gt_length]
            motion2 = full_motion2[idx:idx + gt_length]

        else:
            idx = 0
            gt_length = min(length - idx, self.max_gt_length )
            motion1 = full_motion1[idx:idx + gt_length]  # [95,192]
            motion2 = full_motion2[idx:idx + gt_length]  # [95,192]

        if np.random.rand() > 0.5:
            motion1, motion2, text1, text2 = motion2, motion1, text2, text1
        
        motion1, root_quat_init1, root_pos_init1 = process_motion_np(motion1, 0.001, 0, n_joints=22)
        motion2, root_quat_init2, root_pos_init2 = process_motion_np(motion2, 0.001, 0, n_joints=22)
        r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
        angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

        xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
        relative = np.concatenate([angle, xz], axis=-1)[0]
        motion2 = rigid_transform(relative, motion2)
        
        # # random sample a target theta (-pi,pi), get (cos(theta), 0, sin(theta))
        # theta = np.random.uniform(-np.pi, np.pi)
        # target = np.array([[np.cos(theta), 0, np.sin(theta)]])
        
        # motion1, root_quat_init1, root_pos_init1 = process_motion_np(motion1, 0.001, 0, n_joints=22, target=target)
        # motion2, root_quat_init2, root_pos_init2 = process_motion_np(motion2, 0.001, 0, n_joints=22, target=target)
        
        # # root_quat_init 表示motion从初始朝向转到z轴正方向需要的旋转的四元数表示 shape = [4]
        # # root_pos_init 表示motion从初始帧的根结点的(x,0,z)坐标，shape = [1,3]
        # r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
        # angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

        # xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
        # relative = np.concatenate([angle, xz], axis=-1)[0]
        # motion2 = rigid_transform(relative, motion2)
        
        # # get the translation distance
        # root_distance = np.linalg.norm(root_pos_init2 - root_pos_init1)
        # translation = root_distance * np.random.uniform(0., 1.0)
        # delta_translation = -1 * np.array([[translation*np.cos(theta), 0, translation*np.sin(theta)]])  # 沿着负方向走，让两者中心尽可能向原点靠拢
        
        # # translate
        
        # motion1[:,:66] = (motion1[:,:66].reshape(-1,22,3)+delta_translation).reshape(-1,66)
        # motion2[:,:66] = (motion2[:,:66].reshape(-1,22,3)+delta_translation).reshape(-1,66)
        
        # # get the motion1 and motion2 init root position and orientation
        # motion1_orientation = target[0,[0,2]]  # shape=[2]
        # motion2_orientation = get_orientation(motion2, 0, n_joints=22)[0,[0,2]] # shape=[2]
        # motion1_init_root = motion1[0,:3][[0,2]]
        # motion2_init_root = motion2[0,:3][[0,2]]
        
        # motion1_spatial_condition = np.concatenate([motion1_orientation, motion1_init_root])
        # motion2_spatial_condition = np.concatenate([motion2_orientation, motion2_init_root])
        
        gt_motion1 = motion1  # [95,262]
        gt_motion2 = motion2  # [95,262]

        gt_length = len(gt_motion1)
        
        if gt_length < self.max_gt_length:
            padding_len = self.max_gt_length - gt_length
            D = gt_motion1.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            gt_motion1 = np.concatenate((gt_motion1, padding_zeros), axis=0)
            gt_motion2 = np.concatenate((gt_motion2, padding_zeros), axis=0)


        assert len(gt_motion1) == self.max_gt_length
        assert len(gt_motion2) == self.max_gt_length

        if np.random.rand() > 0.5:
            gt_motion1, gt_motion2, text1, text2 = gt_motion2, gt_motion1, text2, text1
        
        # if self.single_text_description:
        #     text = text1
        return name, text, text1, gt_motion1, gt_motion2, gt_length, np.zeros(0)



class SingleHumanDataset(data.Dataset):
    '''
        for training single motion generation
        if mix, then humanml3d and interhuman are mixed used
        if not, only using the interhuman_single_annoted dataset
        
        # TODO mixed humanml3d haven't finished
    '''
    def __init__(self, opt):
        self.opt = opt
        self.max_cond_length = 1
        self.min_cond_length = 1
        self.max_gt_length = 300
        self.min_gt_length = 15

        self.max_length = self.max_cond_length + self.max_gt_length -1
        self.min_length = self.min_cond_length + self.min_gt_length -1

        self.motion_rep = opt.MOTION_REP
        self.data_list = []
        self.motion_dict = {}

        self.cache = opt.CACHE
        self.single_text_description = opt.SINGLE_TEXT_DESCRIPTION
        
        ignore_list = []
        try:
            ignore_list = open(os.path.join(opt.DATA_ROOT, "split/ignore_list.txt"), "r").readlines()
        except Exception as e:
            print(e)
        data_list = []
        if self.opt.MODE == "train":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "split/train.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "val":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "split/val.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "test":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "split/test.txt"), "r").readlines()
            except Exception as e:
                print(e)

        random.shuffle(data_list)
        # data_list = data_list[:70]
        
        index = 0
        for root, dirs, files in os.walk(pjoin(opt.DATA_ROOT,'motions_processed')):
           
            for file in tqdm(files):
               
                if file.endswith(".npy") and "person1" in root:
                    motion_name = file.split(".")[0]
                    if file.split(".")[0]+"\n" in ignore_list: # or int(motion_name)>1000
                        print("ignore: ", file)
                        continue
                    if file.split(".")[0]+"\n" not in data_list:
                        continue
                    file_path_person1 = pjoin(root, file)
                    file_path_person2 = pjoin(root.replace("person1", "person2"), file)
                    
                    text_path = file_path_person1.replace("motions_processed", "annots").replace("person1", "").replace("npy", "txt")
                    
                    text1_path = file_path_person1.replace("motions_processed", "separate_annots").replace("person1", "text1").replace("npy", "txt")
                    text2_path = file_path_person1.replace("motions_processed", "separate_annots").replace("person1", "text2").replace("npy", "txt")
                    
                    try:
                        texts = []
                        for idx, item in enumerate(open(text_path, "r").readlines()):
                            texts.append(item.replace("\n",""))
                    except:
                        print(idx, text_path)
                    
                    texts = [item.replace("\n", "") for item in open(text_path, "r").readlines()]
                    texts_swap = [item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                                  .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts]

                    texts1 = [item.replace("\n", "") for item in open(text1_path, "r").readlines()]
                    texts1_swap = [item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                                  .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts1]
                    
                    texts2 = [item.replace("\n", "") for item in open(text2_path, "r").readlines()]
                    texts2_swap = [item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                                  .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts2]
                    
                    motion1, motion1_swap = load_motion(file_path_person1, self.min_length, swap=True)  # [95,192]
                    motion2, motion2_swap = load_motion(file_path_person2, self.min_length, swap=True)  # [95,192]
                        
                    if motion1 is None:
                        continue
                    
                    if self.cache:
                        self.motion_dict[index] = [motion1, motion2]
                        self.motion_dict[index+1] = [motion1_swap, motion2_swap]
                    else:
                        self.motion_dict[index] = [file_path_person1, file_path_person2]
                        self.motion_dict[index + 1] = [file_path_person1, file_path_person2]

                    self.data_list.append({
                        # "idx": idx,
                        "name": motion_name,
                        "motion_id": index,
                        "swap":False,
                        "texts":texts,
                        "texts1":texts1,
                        "texts2":texts2
                    })
                    if opt.MODE == "train":
                        self.data_list.append({
                            # "idx": idx,
                            "name": motion_name+"_swap",
                            "motion_id": index+1,
                            "swap": True,
                            "texts": texts_swap,
                            "texts1":texts1_swap,
                            "texts2":texts2_swap
                        })

                    index += 2

        print("total dataset: ", len(self.data_list))

    def real_len(self):
        return len(self.data_list)

    def __len__(self):
        return self.real_len()*1

    def __getitem__(self, item):
       
        idx = item % self.real_len()
        data = self.data_list[idx]

        name = data["name"]
        motion_id = data["motion_id"]
        swap = data["swap"]
        
        num_text = len(data["texts"])
        num_choice = random.choice(list(range(num_text)))
        text, text1, text2 = data["texts"][num_choice].strip(), data["texts1"][num_choice].strip(), data["texts2"][num_choice].strip()
       
        # text = random.choice(data["texts"]).strip()
        
        if self.cache:
            full_motion1, full_motion2 = self.motion_dict[motion_id]
        else:
            file_path1, file_path2 = self.motion_dict[motion_id]
            # print(file_path1)
            # print(file_path2)
            motion1, motion1_swap = load_motion(file_path1, self.min_length, swap=swap)  # [95,192]
            motion2, motion2_swap = load_motion(file_path2, self.min_length, swap=swap)  # [95,192]
            # print(motion1.shape)
            # print(motion2.shape)
            # print(motion1_swap.shape)
            # print(motion2_swap.shape)
            if swap:
                full_motion1 = motion1_swap
                full_motion2 = motion2_swap
            else:
                full_motion1 = motion1
                full_motion2 = motion2
        try:
            length = full_motion1.shape[0]  # 95
        except:
            print(swap,file_path1)
        if length > self.max_length:  # max_length=300
            idx = random.choice(list(range(0, length - self.max_gt_length, 1)))
            gt_length = self.max_gt_length
            motion1 = full_motion1[idx:idx + gt_length]
            motion2 = full_motion2[idx:idx + gt_length]

        else:
            idx = 0
            gt_length = min(length - idx, self.max_gt_length )
            motion1 = full_motion1[idx:idx + gt_length]  # [95,192]
            motion2 = full_motion2[idx:idx + gt_length]  # [95,192]

        if np.random.rand() > 0.5:
            motion1, motion2, text1, text2 = motion2, motion1, text2, text1
        
        motion1, root_quat_init1, root_pos_init1 = process_motion_np(motion1, 0.001, 0, n_joints=22)
        motion2, root_quat_init2, root_pos_init2 = process_motion_np(motion2, 0.001, 0, n_joints=22)
        r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
        angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

        xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
        relative = np.concatenate([angle, xz], axis=-1)[0]
        motion2 = rigid_transform(relative, motion2)

        gt_motion1 = motion1  # [95,262]
        gt_motion2 = motion2  # [95,262]

        gt_length = len(gt_motion1)
        
        if gt_length < self.max_gt_length:
            padding_len = self.max_gt_length - gt_length
            D = gt_motion1.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            gt_motion1 = np.concatenate((gt_motion1, padding_zeros), axis=0)
            gt_motion2 = np.concatenate((gt_motion2, padding_zeros), axis=0)

        assert len(gt_motion1) == self.max_gt_length
        assert len(gt_motion2) == self.max_gt_length

        # TODO np.random.rand() > 0.5
        if np.random.rand() > 0.5:
            return name, text1, torch.zeros(0), gt_motion1, torch.zeros(0), gt_length, torch.zeros(0)
        else:
            return name, text2, torch.zeros(0), gt_motion2, torch.zeros(0), gt_length, torch.zeros(0)



class InterHumanPipelineInferDataset(data.Dataset):
    
    def __init__(self, opt):
        self.opt = opt
        self.max_cond_length = 1
        self.min_cond_length = 1
        self.max_gt_length = 300
        self.min_gt_length = 15

        self.max_length = self.max_cond_length + self.max_gt_length -1
        self.min_length = self.min_cond_length + self.min_gt_length -1

        self.motion_rep = opt.MOTION_REP
        self.data_list = []
        self.motion_dict = {}

        self.cache = opt.CACHE
        self.single_text_description = opt.SINGLE_TEXT_DESCRIPTION
        
        ignore_list = []
        try:
            ignore_list = open(os.path.join(opt.DATA_ROOT, "split/ignore_list.txt"), "r").readlines()
        except Exception as e:
            print(e)
        data_list = []
        if self.opt.MODE == "train":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "split/train.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "val":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "split/val.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "test":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "split/test.txt"), "r").readlines()
            except Exception as e:
                print(e)
        
        index = 0
        for root, dirs, files in os.walk(pjoin(opt.DATA_ROOT,'motions_processed')):
           
            for file in tqdm(files):
               
                if file.endswith(".npy") and "person1" in root:
                    motion_name = file.split(".")[0]
                    if file.split(".")[0]+"\n" in ignore_list: # or int(motion_name)>1000
                        print("ignore: ", file)
                        continue
                    if file.split(".")[0]+"\n" not in data_list:
                        continue
                    file_path_person1 = pjoin(root, file)
                    file_path_person2 = pjoin(root.replace("person1", "person2"), file)
                    text_path = file_path_person1.replace("motions_processed", "annots").replace("person1", "").replace("npy", "txt")

                    text1_path = file_path_person1.replace("motions_processed", "separate_annots").replace("person1", "text1").replace("npy", "txt")
                    text2_path = file_path_person1.replace("motions_processed", "separate_annots").replace("person1", "text2").replace("npy", "txt")
                    
                    try:
                        
                        texts = []
                        for idx, item in enumerate(open(text_path, "r").readlines()):
                            texts.append(item.replace("\n",""))
                    except:
                        print(idx, text_path)
                    
                    try:
                        texts = [item.replace("\n", "") for item in open(text_path, "r").readlines()]
                    except:
                        print(text_path)
                    texts_swap = [item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                                  .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts]

                    texts1 = [item.replace("\n", "") for item in open(text1_path, "r").readlines()]
                    texts1_swap = [item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                                  .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts1]
                    
                    texts2 = [item.replace("\n", "") for item in open(text2_path, "r").readlines()]
                    texts2_swap = [item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                                  .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts2]
                    
                    if self.cache:
                        motion1, motion1_swap = load_motion(file_path_person1, self.min_length, swap=True)  # [95,192]
                        motion2, motion2_swap = load_motion(file_path_person2, self.min_length, swap=True)  # [95,192]
                        
                        if motion1 is None:
                            continue
                    
                    if self.cache:
                        self.motion_dict[index] = [motion1, motion2]
                        self.motion_dict[index+1] = [motion1_swap, motion2_swap]
                    else:
                        self.motion_dict[index] = [file_path_person1, file_path_person2]
                        self.motion_dict[index + 1] = [file_path_person1, file_path_person2]

                    self.data_list.append({
                        # "idx": idx,
                        "name": motion_name,
                        "motion_id": index,
                        "swap":False,
                        "texts":texts,
                        "texts1":texts1,
                        "texts2":texts2
                    })
                    if opt.MODE == "train":
                        self.data_list.append({
                            # "idx": idx,
                            "name": motion_name+"_swap",
                            "motion_id": index+1,
                            "swap": True,
                            "texts": texts_swap,
                            "texts1":texts1_swap,
                            "texts2":texts2_swap
                        })

                    index += 2

        print("total dataset: ", len(self.data_list))

    def real_len(self):
        return len(self.data_list)

    def __len__(self):
        return self.real_len()*1

    def __getitem__(self, item):
       
        idx = item % self.real_len()
        data = self.data_list[idx]

        name = data["name"]
        motion_id = data["motion_id"]
        swap = data["swap"]
        
        # text = random.choice(data["texts"]).strip()

        # text = data["texts"][0].strip()
        # text1 = data["texts1"][0].strip()
        # text2 = data["texts2"][0].strip()
        
        num_text = len(data["texts"])
        num_choice = random.choice(list(range(num_text)))
        text, text1, text2 = data["texts"][num_choice].strip(), data["texts1"][num_choice].strip(), data["texts2"][num_choice].strip()
       
        
        if self.cache:
            full_motion1, full_motion2 = self.motion_dict[motion_id]
        else:
            file_path1, file_path2 = self.motion_dict[motion_id]
            motion1, motion1_swap = load_motion(file_path1, self.min_length, swap=swap)  # [95,192]
            motion2, motion2_swap = load_motion(file_path2, self.min_length, swap=swap)  # [95,192]
            if swap:
                full_motion1 = motion1_swap
                full_motion2 = motion2_swap
            else:
                full_motion1 = motion1
                full_motion2 = motion2

        length = full_motion1.shape[0]  # 95
        if length > self.max_length:  # max_length=300
            idx = random.choice(list(range(0, length - self.max_gt_length, 1)))
            gt_length = self.max_gt_length
            motion1 = full_motion1[idx:idx + gt_length]
            motion2 = full_motion2[idx:idx + gt_length]

        else:
            idx = 0
            gt_length = min(length - idx, self.max_gt_length )
            motion1 = full_motion1[idx:idx + gt_length]  # [95,192]
            motion2 = full_motion2[idx:idx + gt_length]  # [95,192]

        if np.random.rand() > 0.5:
            motion1, motion2, text1, text2 = motion2, motion1, text2, text1
        
        motion1, root_quat_init1, root_pos_init1 = process_motion_np(motion1, 0.001, 0, n_joints=22)
        motion2, root_quat_init2, root_pos_init2 = process_motion_np(motion2, 0.001, 0, n_joints=22)
        r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
        angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

        xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
        relative = np.concatenate([angle, xz], axis=-1)[0]
        motion2 = rigid_transform(relative, motion2)

        gt_motion1 = motion1  # [95,262]
        gt_motion2 = motion2  # [95,262]
        text1 = text1
        text2 = text2

        gt_length = len(gt_motion1)
        
        if gt_length < self.max_gt_length:
            padding_len = self.max_gt_length - gt_length
            D = gt_motion1.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            gt_motion1 = np.concatenate((gt_motion1, padding_zeros), axis=0)
            gt_motion2 = np.concatenate((gt_motion2, padding_zeros), axis=0)


        assert len(gt_motion1) == self.max_gt_length
        assert len(gt_motion2) == self.max_gt_length

        if np.random.rand() > 0.5:
            gt_motion1, gt_motion2, text1, text2 = gt_motion2, gt_motion1, text2, text1
        
        return name, text, text1, text2, gt_motion1, gt_motion2, gt_length, torch.zeros(0)

