import sys
sys.path.append(sys.path[0] + r"/../")
import logging
import torch
import lightning.pytorch as pl
import torch.optim as optim
from collections import OrderedDict
from datasets import DataModule
from configs import get_config
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
from models import *
import os
import argparse
    
os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
from lightning.pytorch.strategies import DDPStrategy
torch.set_float32_matmul_precision('medium')

class LitTrainModel(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg
        self.mode = cfg.TRAIN.MODE

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model = model

        self.writer = SummaryWriter(self.log_dir)
        
        logging.basicConfig(filename=os.path.join(self.log_dir,'train.log'), filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    
    def _configure_optim(self):
        
        optimizer = optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True)
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()

    def forward(self, batch_data):
        
        name, text, text_multi_person, motion1, motion2, motion_lens, spatial_condition = batch_data
        
        motion1 = motion1.detach().float() 
        
        batch = OrderedDict({})
        if min(motion2.shape) == 0:  # for single human training
            motions = motion1
        elif min(spatial_condition.shape) == 0:  # for pure double huaman training
            motion2 = motion2.detach().float() 
            motions = torch.cat([motion1, motion2], dim=-1)
            
           
        B, T = motion1.shape[:2]
        
        batch["motions"] = motions.reshape(B, T, -1).type(torch.float32)
        batch["motion_lens"] = motion_lens.long()
        batch["person_num"] = motions.shape[-1] // motion1.shape[-1]

        if isinstance(text, torch.Tensor):
            batch["text"] = None
        else:
            batch["text"] = text
        
        if isinstance(text_multi_person, torch.Tensor):
            batch["text_multi_person"] = None
        else:
            batch["text_multi_person"] = text_multi_person
        
        loss, loss_logs = self.model(batch)
        return loss, loss_logs

    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.logs = OrderedDict()


    def training_step(self, batch, batch_idx):
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()

        return {"loss": loss,
            "loss_logs": loss_logs}


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        for k, v in outputs['loss_logs'].items():
            if k not in self.logs:
                self.logs[k] = v.item()
            else:
                self.logs[k] += v.item()

        self.it += 1
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0 and self.device.index == 0:
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS
                self.writer.add_scalar(tag, mean_loss[tag], self.it)
            self.logs = OrderedDict()
            print_current_loss(self.start_time, self.it, mean_loss,
                               self.trainer.current_epoch,
                               inner_iter=batch_idx,
                               lr=self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_train_epoch_end(self):
        # pass
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def save(self, file_name):
        state = {}
        try:
            state['model'] = self.model.module.state_dict()
        except:
            state['model'] = self.model.state_dict()
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return


def build_models(cfg):
    model = InterGenSpatialControlNet(cfg)
    return model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process configs.')
    parser.add_argument('--model_config', type=str, help='model config')
    parser.add_argument('--dataset_config', type=str, help='dataset config')
    parser.add_argument('--train_config', type=str, help='train config')

    args = parser.parse_args()

    print(os.getcwd())
    model_cfg = get_config(args.model_config)
    train_cfg = get_config(args.train_config)
    data_cfg = get_config(args.dataset_config).interhuman
        
    datamodule = DataModule(data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)
    model = build_models(model_cfg)
    
    if train_cfg.TRAIN.FROM_PRETRAIN:
        ckpt = torch.load(train_cfg.TRAIN.FROM_PRETRAIN, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print("checkpoint state loaded!")
        
    if train_cfg.TRAIN.RESUME:
        resume_path = train_cfg.TRAIN.RESUME
    else:
        resume_path = None
        
    litmodel = LitTrainModel(model, train_cfg)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=litmodel.model_dir,
                                                       every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH,
                                                       save_top_k = train_cfg.TRAIN.SAVE_TOPK,
                                                       )
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        devices="auto", accelerator='gpu',
        max_epochs=train_cfg.TRAIN.EPOCH,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=32,
        callbacks=[checkpoint_callback],
        detect_anomaly=True
    )

    trainer.fit(model=litmodel, datamodule=datamodule, ckpt_path=resume_path)
