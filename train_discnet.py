# -*- coding: utf-8 -*-
data_dir = 'preprocessed_mimic3/'
train_config = {
    "cuda_visible_devices": "0",
    "lr": 5e-4, # learning rate
    "batch_size": 5,
    "max_seq_len": 4500, # length limitation
    "tau": 1000,
    "output_dir": "./output_mimic_discnet_",
    "log_filename": "train_log.txt",
    "heading2idx_path": data_dir + 'heading2idx.json',
    "word2idx_path": data_dir + 'word2idx.json',
    "code2idx_path": data_dir + 'code2idx.json',
    "desc_dic_path": data_dir + 'desc_dict_v3.json',
    "all_headings_dic_path": data_dir + "all_headings_dic.json",
    "code2fq_path": data_dir + "code2fq.json",
    "train_set_path": data_dir + 'train_set.json',
    "dev_set_path": data_dir + 'dev_set.json',
    "test_set_path": data_dir + 'test_set.json',
    "cuda": True,
    "num_workers": 0, # number of worker for dataloader
}

import os
os.environ["CUDA_VISIBLE_DEVICES"] = train_config['cuda_visible_devices']
import torch
# torch.manual_seed(0)
import random
# random.seed(0)
import numpy as np
# np.random.seed(0)
from dataset.mimic_dataset_discourse_sen import MimicDataset, collate_fn
from torch.utils.data import DataLoader
from models_discnet_re.network import *
from models_discnet_re.hyperparameters import hp_config
from utils.helper_functions import *
import torch
from utils.evaluation import *
import pandas as pd


class HEXTrainer:
    def __init__(self,
                 train_config,
                 ):
        # 加载字典及训练数据0
        self.config = train_config
        if torch.cuda.is_available() and self.config['cuda']:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.word2idx = maybe_load_json(self.config['word2idx_path'])
        self.code2idx = maybe_load_json(self.config["code2idx_path"])
        self.idx2code = {v: k for k, v in self.code2idx.items()}
        self.desc_dic = maybe_load_json(self.config["desc_dic_path"])
        self.all_headings_dic = maybe_load_json(self.config["all_headings_dic_path"])
        self.code2fq = maybe_load_json(self.config["code2fq_path"])
        self.heading2idx = maybe_load_json(self.config["heading2idx_path"])

        self.code2fq = maybe_load_json(self.config["code2fq_path"])
        code_fq = [self.code2fq[self.idx2code[i]] for i in range(len(self.code2idx))]
        code_fq = -np.array(code_fq)
        _max_fq = np.max(code_fq)
        _min_fq = np.min(code_fq)
        normalized_code_fq = (code_fq - _min_fq) / (_max_fq - _min_fq)
        normalized_code_fq = normalized_code_fq ** self.config['tau']
        self.code_fq = torch.tensor(normalized_code_fq).to(self.device).float()

        self.max_seq_len = self.config["max_seq_len"]
        self.batch_size = self.config["batch_size"]
        self.lr = self.config["lr"]

        self.init_models()
        self.init_dataloader()

        self.optim_parameters = list(self.model.parameters())

    def init_optimizer(self, initial_lr=None, last_epoch=-1):
        if initial_lr is None or float(initial_lr) <= 0:
            initial_lr = self.lr
        self.optimizer = torch.optim.Adam([{'params': self.optim_parameters, 'initial_lr': initial_lr}], lr=initial_lr)

        self.scheduler2 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[9, 12, 15, 18], gamma=0.3,
                                                               last_epoch=last_epoch)

    def init_models(self):
        self.model = DiscNet_RE(hp_config, self.code_fq)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def init_training_dataset(self):
        train_dataset = MimicDataset(data=self.config['train_set_path'],
                                     word2idx=self.word2idx,
                                     code2idx=self.code2idx,
                                     heading2idx=self.heading2idx,
                                     desc_dic=self.desc_dic,
                                     all_headings_dic=self.all_headings_dic,
                                     max_seq_len=self.max_seq_len,
                                     data_augmentation=True,
                                     shuffle_sen=True,
                                     shuffle=True,
                                     num_len_cut=200,
                                     )
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size,
                                           num_workers=self.config['num_workers'],
                                           shuffle=False,
                                           collate_fn=collate_fn)

    def init_dataloader(self):
        self.init_training_dataset()

        valid_dataset = MimicDataset(data=self.config['dev_set_path'],
                                     word2idx=self.word2idx,
                                     code2idx=self.code2idx,
                                     heading2idx=self.heading2idx,
                                     desc_dic=self.desc_dic,
                                     all_headings_dic=self.all_headings_dic,
                                     max_seq_len=self.max_seq_len,
                                     data_augmentation=False,
                                     shuffle=False,
                                     )
        test_dataset = MimicDataset(data=self.config['test_set_path'],
                                    word2idx=self.word2idx,
                                    code2idx=self.code2idx,
                                    heading2idx=self.heading2idx,
                                    desc_dic=self.desc_dic,
                                    all_headings_dic=self.all_headings_dic,
                                    max_seq_len=self.max_seq_len,
                                    data_augmentation=False,
                                    shuffle=False,
                                    )
        self.code_desc_text = valid_dataset.code_desc_text.to(self.device)

        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size,
                                           num_workers=self.config['num_workers'],
                                           shuffle=False,
                                           collate_fn=collate_fn)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                          num_workers=self.config['num_workers'],
                                          shuffle=False,
                                          collate_fn=collate_fn)

    def train(self):
        self.model.train()
        self.iteration(self.train_dataloader, 'train')

    def valid(self):
        self.model.eval()
        with torch.no_grad():
            return self.iteration(self.valid_dataloader, 'valid')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            return self.iteration(self.test_dataloader, 'test')

    def iteration(self, data_loader, task='train'):
        creat_dir(self.config["output_dir"])
        creat_dir(self.config["output_dir"] + '/' + 'predictions')
        creat_log_df(self.config["output_dir"], self.config["log_filename"])
        log_path = os.path.join(self.config["output_dir"],
                                self.config["log_filename"])
        epoch, _ = read_log_and_get_start_epoch(log_path)
        if task == 'train':
            train = True
        else:
            train = False
        if task == 'train' and epoch == -1:
            epoch = 1
        elif task == "train":
            epoch += 1

        data_iter = tqdm(enumerate(data_loader),
                         desc=f"{task}_{epoch}",
                         total=len(data_loader),
                         bar_format="{l_bar}{r_bar}")
        total_loss = 0
        total_l2_loss = 0
        total_p_at_8 = 0
        total_p_at_15 = 0

        all_probs = []
        all_labels = []

        for i, data in data_iter:
            try:
                probs, loss, l2_loss = \
                    self.model(text=data['text'].to(self.device),
                               segments=data['segments'].to(self.device),
                               sen_start_pos=data['sen_start_pos'], sen_end_pos=data['sen_end_pos'],
                               node_start_pos=data['node_start_pos'], node_end_pos=data['node_end_pos'],
                               sen_spans=data['sen_spans'], node_spans=data['node_spans'],
                               code_desc_text=self.code_desc_text,
                               labels=data['labels'].to(self.device))

                p_at_8 = precision_at_k_(probs.detach().data,
                                         data['labels'].to(self.device), k=8)
                p_at_15 = precision_at_k_(probs.detach().data,
                                          data['labels'].to(self.device), k=15)
                total_p_at_8 += p_at_8
                total_p_at_15 += p_at_15

                total_loss += loss.item()
                total_l2_loss += l2_loss.item()
                loss = loss + 1e-4 * l2_loss

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    log_dic = {"epoch": epoch,
                               "train_loss": round(total_loss / (i + 1), 5),
                               "train_p@8": round(total_p_at_8 / (i + 1), 5),
                               "train_p@15": round(total_p_at_15 / (i + 1), 5),
                               "l2_loss": round(total_l2_loss / (i + 1), 5),
                               }
                else:
                    all_probs.append(probs.detach().cpu().data.numpy())
                    all_labels.append(data['labels'].detach().cpu().data.numpy())
                    log_dic = {"epoch": epoch,
                               f"{task}_loss": round(total_loss / (i + 1), 5),
                               f"{task}_p@8": round(total_p_at_8 / (i + 1), 5),
                               f"{task}_p@15": round(total_p_at_15 / (i + 1), 5),
                               }
                data_iter.write(str({k: v for k, v in log_dic.items()
                                     if k != "epoch"}))
            except Exception as e:
                raise(e)
                # torch.cuda.empty_cache()

        if train:
            write_log(log_path, log_dic)
            self.save_state_dict(epoch,
                                 dir_path=train_config["output_dir"],
                                 file_name="mimic.model")
        else:
            all_probs = np.concatenate(all_probs, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            np.save(f"{self.config['output_dir']}/predictions/{task}_probs_{epoch}.npy", all_probs)
            np.save(f"{self.config['output_dir']}/predictions/{task}_labels_{epoch}.npy", all_labels)

            _thresholds = get_threshold_by_fq(self.code2idx, self.code2fq, split_point=10, head_t=0.3, tail_t=0.08)
            _thresholds = np.expand_dims(_thresholds, axis=0)

            _metrics = all_metrics((all_probs > _thresholds).astype(np.int), all_labels,
                                   k=15, yhat_raw=all_probs, calc_auc=True)
            for k, v in _metrics.items():
                if pd.isnull(v):
                    _metrics[k] = -1
            _metrics[f"{task}_p@8"] = log_dic[f"{task}_p@8"]
            _metrics[f"{task}_p@15"] = log_dic[f"{task}_p@15"]
            log_dic.update({task: _metrics})
            log_dic['lr'] = self.optimizer.param_groups[0]['lr']
            if task == 'valid':
                write_valid_log(log_path, log_dic)
            elif task == 'test':
                log_dic['epoch'] = 'test'
                write_log(log_path, log_dic)
            return _metrics[f"{task}_p@8"] + _metrics[f"{task}_p@15"]

    def save_state_dict(self, epoch, dir_path="./output", file_name="mimic.model"):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        save_path = dir_path + "/" + file_name + ".epoch.{}".format(str(epoch))

        if torch.cuda.device_count() > 1:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        torch.save({"model_state_dict": model_state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict()}, save_path)
        print("{} has been saved!".format(save_path))

    def load_model(self, path="./output", strict=True):
        if os.path.isdir(path):
            path = find_most_recent_state_dict(path)
            if path is None: return None
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        r = self.model.load_state_dict(state_dict, strict=strict)
        print(r)
        print("{} has been loaded for training!".format(path))


if __name__ == '__main__':
    trainer = HEXTrainer(train_config)
    log_path = os.path.join(train_config["output_dir"],
                            train_config["log_filename"])
    start_epoch, initial_lr = read_log_and_get_start_epoch(log_path)
    trainer.init_optimizer(initial_lr=None, last_epoch=start_epoch)
    
    # trainer.load_model(dir_path=train_config['output_dir'], strict=True)
    # training
    max_score = 0
    patient = 0
    max_patient = 5
    while True:
        trainer.init_training_dataset()
        for param_group in trainer.optimizer.param_groups:
            print(f"training with learning rate {param_group['lr']}")
        if patient >= max_patient:
            print('early stop!')
            break
        trainer.train()
        _score = trainer.valid()
        if _score > max_score:
            max_score = _score
            patient = 0
        else:
            patient += 1
        trainer.scheduler2.step()


    # testing
    output_dir = train_config['output_dir']
    with open(f"{output_dir}/train_log.txt", 'r') as f:
        lines = [eval(i) for i in f.readlines()]
    score_lis = [i['valid']['valid_p@8'] + i['valid']['valid_p@15'] for i in lines]
    epoch = lines[np.argmax(score_lis)]['epoch']
    print(epoch)
    trainer.load_model(path=f'{output_dir}/mimic.model.epoch.{epoch}', strict=True)
    trainer.test()