from os.path import join as pjoin
from torch.utils.data import Dataset, DataLoader
from datasets import InterHumanPipelineInferDataset
from models import *
import copy
from datasets.evaluator_models import InterCLIP
from tqdm import tqdm
import torch

from utils.quaternion import *

normalizer = MotionNormalizerTorch()

def process_motion_np(motion, feet_thre, prev_frames, n_joints, target=np.array([[0, 0, 1]])):
    
    positions = motion[:, :n_joints*3].reshape(-1, n_joints, 3)  # [95,22,3]
    
    vels = motion[:, n_joints*3:n_joints*6].reshape(-1, n_joints, 3)  # [95,22,3]

    '''XZ at origin'''
    # move the root pos of the first frame to (0,0) of xz plane
    # for example
    # poistion_root_init = [2,3,5]  -> [0,3,5]
    
    root_pos_init = positions[prev_frames] # [95,22,3] -> [22,3]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])  # [3]
    positions = positions - root_pose_init_xz  

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across = root_pos_init[r_hip] - root_pos_init[l_hip] 
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis] 

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1) 
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]  # 归一化

    # target = np.array([[0, 0, 1]]) 
    root_quat_init = qbetween_np(forward_init, target)  
    root_quat_init_for_all = np.ones(positions.shape[:-1] + (4,)) * root_quat_init  

    positions = qrot_np(root_quat_init_for_all, positions) 

    vels = qrot_np(root_quat_init_for_all, vels) 
    
    '''Get Joint Rotation Invariant Position Represention'''
    joint_positions = positions.reshape(len(positions), -1)  # [95,66]
    joint_vels = vels.reshape(len(vels), -1)  # [95,66]
    
    motion[:, :n_joints*3] = joint_positions
    motion[:, n_joints*3:n_joints*6] = joint_vels
    
    return motion, root_quat_init, root_pose_init_xz[None]

def rigid_transform(relative, data):  

    global_positions = data[..., :22 * 3].reshape(data.shape[:-1] + (22, 3))
    global_vel = data[..., 22 * 3:22 * 6].reshape(data.shape[:-1] + (22, 3))

    relative_rot = relative[0]
    relative_t = relative[1:3]
    relative_r_rot_quat = np.zeros(global_positions.shape[:-1] + (4,))
    relative_r_rot_quat[..., 0] = np.cos(relative_rot)
    relative_r_rot_quat[..., 2] = np.sin(relative_rot)
    global_positions = qrot_np(qinv_np(relative_r_rot_quat), global_positions)
    global_positions[..., [0, 2]] += relative_t
    data[..., :22 * 3] = global_positions.reshape(data.shape[:-1] + (-1,))
    global_vel = qrot_np(qinv_np(relative_r_rot_quat), global_vel)
    data[..., 22 * 3:22 * 6] = global_vel.reshape(data.shape[:-1] + (-1,))

    return data


class EvaluationDataset(Dataset):

    def __init__(self, model, dataset, device, mm_num_samples, mm_num_repeats):
        
        self.normalizer = MotionNormalizer()
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
        self.max_length = dataset.max_length

        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        mm_idxs = idxs[:mm_num_samples]

        generated_motions = []
        mm_generated_motions = []
        # Pre-process all target captions
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                name, text, text1, text2, motion1, motion2, motion_lens, hint = data
                batch = {}
                
                if min(hint.shape) == 0:
                    hint = None
                    
                if i in mm_idxs:
                    text1 = list(text1) * mm_num_repeats
                    text2 = list(text2) * mm_num_repeats
                    text = list(text) * mm_num_repeats
                    
                batch["motion_lens"] = motion_lens
                
                model_name = self.model.__class__.__name__
                motions_output = self.pipeline_generate(self.model, motion_lens, [text1,text2], text, [motion1,motion2], 1, hint)  # [1,27,2,262]
                
                motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())
                
                B,T = motions_output.shape[0], motions_output.shape[1]
                if T < self.max_length:
                    padding_len = self.max_length - T
                    D = motions_output.shape[-1]
                    padding_zeros = np.zeros((B, padding_len, 2, D))
                    motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
                assert motions_output.shape[1] == self.max_length

                sub_dict = {'motion1': motions_output[0, :,0],
                            'motion2': motions_output[0, :,1],
                            'motion_lens': motion_lens[0],
                            'text': text[0]}
                
                if hint is not None:
                    sub_dict['spatial_condition'] = hint[0]
                # else:
                #     sub_dict['hint'] = None
                generated_motions.append(sub_dict)
                if i in mm_idxs:
                    mm_sub_dict = {'mm_motions': motions_output,
                                   'motion_lens': motion_lens[0],
                                    'text': text[0]}
                    mm_generated_motions.append(mm_sub_dict)


        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = data['motion1'], data['motion2'], data['motion_lens'], data['text']
        hint = data['spatial_condition'] if 'spatial_condition' in data else torch.zeros(0)
        return "generated", text, "placeholder", "placeholder", motion1, motion2, motion_lens, hint

    def pipeline_generate(self, model, motion_lens, texts, text_multi_person, motions, FLAG=0, hint=None):

        def generate(motion_lens, text, text_multi_person, person_num=1, motion_guidance=None, hint=None):
            T = motion_lens
            batch = {}
            batch["prompt"] = list(text)
            batch["text"] = list(text)
            batch["text_multi_person"] = list(text_multi_person)
            batch["person_num"] = person_num
            batch["motion_lens"] = T # torch.tensor([T]).unsqueeze(0).long().to(torch.device("cuda:0"))
            batch["motion_guidance"] = motion_guidance
            
            if hint is not None:
                hint = hint[:,:T,...]
            batch["spatial_condition"] = hint
            batch = model.forward_test(batch)
            output = batch["output"].reshape(batch["output"].shape[0], batch["output"].shape[1], 1, -1)
            return output
        
        generated_motions = []
        motion = generate(motion_lens.to(self.device), texts[0], text_multi_person)
        
        generated_motions.append(motion)
        
        if FLAG != 0:
            for person_idx, text in enumerate(texts[1:]):
                tmp_motions = [ m.detach().float().to(self.device) for m in generated_motions]
                motion_guidance = torch.cat(tmp_motions, dim=-2)
                motion = generate(motion_lens.to(self.device), text, text_multi_person, person_idx+2, motion_guidance, None) #hint.to(self.device)) 
                generated_motions.append(motion)
        return torch.cat(generated_motions, dim=-2) # [:motion_lens.item()]  # [B,T,P,D]

    def intergen_generate(self, model, motion_lens, texts, text_multi_person, motions, person_num=0):

        def generate(motion_lens, texts, text_multi_person, person_num=1):
            T = motion_lens
            batch = {}
            batch["text1"] = list(texts[0])
            batch["text2"] = list(texts[1])
            
            batch["text"] = list(text_multi_person)
            batch["person_num"] = person_num
            batch["motion_lens"] = T 
            
            batch = model.forward_test(batch)
            output = batch["output"].reshape(batch["output"].shape[0], batch["output"].shape[1], person_num, -1)
            return output
        
        motion = generate(motion_lens.to(self.device), texts, text_multi_person, person_num=person_num)
        
        return motion
 
class MMGeneratedDataset(Dataset):
    def __init__(self, motion_dataset):
        self.dataset = motion_dataset.mm_generated_motions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        motion_lens = data['motion_lens']
        mm_motions1 = mm_motions[:,:,0]
        mm_motions2 = mm_motions[:,:,1]
        text = data['text']
        motion_lens = np.array([motion_lens]*mm_motions1.shape[0])
        return "mm_generated", text, "placeholder", "placeholder", mm_motions1, mm_motions2, motion_lens, torch.zeros(0)


def get_dataset_motion_loader(opt, batch_size):
    opt = copy.deepcopy(opt)
    # Configurations of T2M dataset and KIT dataset is almost the same
    
    if opt.NAME == 'interhuman':
        print('Loading dataset %s ...' % opt.NAME)
        dataset = InterHumanPipelineInferDataset(opt)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset

def get_motion_loader(batch_size, model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats):
    
    dataset = EvaluationDataset(model, ground_truth_dataset, device, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader




def build_models(cfg):
    model = InterCLIP(cfg)
    checkpoint = torch.load(pjoin('eval_model/interclip.ckpt'),map_location="cpu")
    
    # checkpoint = torch.load(pjoin('checkpoints/interclip/model/5.ckpt'),map_location="cpu")
    for k in list(checkpoint["state_dict"].keys()):
        if "model" in k:
            checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return model


class EvaluatorModelWrapper(object):

    def __init__(self, cfg, device):

        self.model = build_models(cfg)
        self.cfg = cfg
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()


    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, text1, text2, motion1, motion2, motion_lens, _ = batch_data
            
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

            '''Text Encoding'''
            text_embedding = self.model.encode_text(batch)['text_emb'][align_idx]

        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, text1, text2, motion1, motion2, motion_lens, _ = batch_data
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

        return motion_embedding


class TrainEvaluatorModelWrapper(object):

    def __init__(self, cfg, device):

        self.model = build_models(cfg)
        self.cfg = cfg
        self.device = device

        self.model = self.model.to(device)
        # self.model.eval()


    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, batch_data):
        name, text, motion1, motion2, motion_lens = batch_data
        motion1 = motion1.detach().float()  # .to(self.device)
        motion2 = motion2.detach().float()  # .to(self.device)
        motions = torch.cat([motion1, motion2], dim=-1)
        motions = motions.detach().to(self.device).float()
        align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        motion_lens = motion_lens[align_idx]
        text = list(text)
        B, T = motions.shape[:2]
        cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
        padded_len = cur_len.max()
        batch = {}
        batch["text"] = text
        batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
        batch["motion_lens"] = motion_lens
        '''Motion Encoding'''
        motion_embedding = self.model.encode_motion(batch)['motion_emb']
        '''Text Encoding'''
        text_embedding = self.model.encode_text(batch)['text_emb'][align_idx]

        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

        return motion_embedding

    def compute_contrastive_loss(self, batch_data):
        
        loss_total, losses = self.model(batch_data)
        
        return loss_total, losses
