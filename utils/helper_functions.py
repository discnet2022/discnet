import pickle
import json
import os
import torch
import numpy as np
def maybe_load_json(path_or_dict):
    if isinstance(path_or_dict, dict):
        return path_or_dict
    with open(path_or_dict, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_precison_at_n(probs, labels, n=15):
    p_at_n = 0
    for _probs, _labels in zip(probs, labels):
        top_n_ids = torch.argsort(_probs, descending=True)[:n].tolist()
        true_n_ids =  torch.where(_labels > 0)[0].tolist()
        p_at_n += len(set(top_n_ids).intersection(true_n_ids))/n
    p_at_n /= labels.shape[0]
    return p_at_n

def precision_at_k_(yhat_raw, y, k=15):
    #num true labels in top k predictions / k
    sortd = torch.argsort(yhat_raw, descending=True)
    topk = sortd[:,:k]

    #get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i,tk].sum()
            denom = len(tk)
            vals.append((num_true_in_top_k / float(denom)).item())

    return np.mean(vals)

def maybe_load_pickle(path_or_object):
    if isinstance(path_or_object, str):
        with open(path_or_object, "rb") as f:
            return pickle.load(f)
    return path_or_object

def creat_dir(dir=""):
    if not os.path.exists(dir):
        os.mkdir(dir)
def creat_log_df(dir, filename="df_log.pickle"):
    if not os.path.isfile(os.path.join(dir, filename)):
        with open(os.path.join(dir, filename), "w+", encoding="utf-8") as f:
            pass
        print("{} created!".format(filename))
def read_log_and_get_start_epoch(log_path):
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lis = [eval(i) for i in f.readlines()]
            if len(lis) > 0:
                lr = lis[-1].get('lr')
                if lr is not None:
                    lr = float(lr)
                return lis[-1]["epoch"], lr
            else:
                return -1, None
    except Exception as e:
        return -1, None
def write_log(log_path, train_log_dic):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(str(train_log_dic))
        f.write("\n")
def write_valid_log(log_path, valid_log_dic):
    with open(log_path, "r", encoding="utf-8") as f:
        log_lis = [eval(i) for i in f.readlines()]
    if len(log_lis) == 0:
        print(valid_log_dic)
        raise RuntimeError("an exception occurred during log reading! failed to write validation log!")
    last_epoch_dic = log_lis[-1]
    last_epoch_dic.update({k: v for k, v in valid_log_dic.items() if k != "epoch"})
    log_lis = log_lis[:-1]
    log_lis.append(last_epoch_dic)
    with open(log_path, "w+", encoding="utf-8") as f:
        for dic in log_lis:
            f.write(str(dic))
            f.write("\n")
def find_most_recent_state_dict(dir_path):
    dic_lis = [i for i in os.listdir(dir_path)]
    dic_lis = [i for i in dic_lis if "model" in i]
    if len(dic_lis) == 0:
        raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
    dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
    return dir_path + "/" + dic_lis[-1]
