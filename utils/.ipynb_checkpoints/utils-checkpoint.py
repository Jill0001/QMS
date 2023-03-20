import math
import time
import torch
import numpy as np
import torch.nn as nn

import random
import os
from tqdm import tqdm
from flashtext import KeywordProcessor


def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


def make_exp_dirs(exp_name):
    day_logs_root = 'generation_logs/' + time.strftime("%Y-%m%d", time.localtime())
    os.makedirs(day_logs_root, exist_ok=True)
    exp_log_path = os.path.join(day_logs_root, exp_name)

    model_save_root = 'saved_models/'
    model_save_path = os.path.join(model_save_root, exp_name)

    os.makedirs(exp_log_path, exist_ok=True)  # log dir make
    os.makedirs(model_save_path, exist_ok=True)  # model save dir make

    return exp_log_path


def send_to_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def writr_gt(val_dataloader, test_dataloader, log_dir):
    gt_file_name_val = os.path.join(log_dir, ('gt4eval.txt'))
    gt_txt_val = open(gt_file_name_val, 'w')

    gt_with_id_file_name_val = os.path.join(log_dir, ('gt_val.txt'))
    gt_with_id_txt_val = open(gt_with_id_file_name_val, 'w')

    gt_file_name_test = os.path.join(log_dir, ('gt4test.txt'))
    gt_txt_test = open(gt_file_name_test, 'w')

    gt_with_id_file_name_test = os.path.join(log_dir, ('gt_test.txt'))
    gt_with_id_txt_test = open(gt_with_id_file_name_test, 'w')
    
    for idx, test_data in tqdm(enumerate(val_dataloader)):
        for i in range(len(test_data['query'])):
            gt_txt_val.write(test_data['all_queries'][i] + '\n')
            gt_with_id_txt_val.write(test_data['item_id'][i] + ' ' + test_data['all_queries'][i] + '\n')

    for idx, test_data in tqdm(enumerate(test_dataloader)):
        for i in range(len(test_data['query'])):
            gt_txt_test.write(test_data['all_queries'][i] + '\n')
            gt_with_id_txt_test.write(test_data['item_id'][i] + ' ' + test_data['all_queries'][i] + '\n')

    for txt in [gt_txt_val, gt_with_id_txt_val, gt_txt_test, gt_with_id_txt_test]:
        txt.flush()
        txt.close()


def de_stop_words(text, ign_kp):
    howto_lis = ["怎么", '怎么做', '怎样', '怎样做', "制作", "怎么制作", "怎样制作",
                 "做法", '方法', '吃法', "教程", '教学', "完整教程", '入门教程',
                 '正宗做法', '家庭做法', '的家庭做法', '家常做法', '正确做法', '制作方法',
                 '怎么做才好吃', '怎么做好吃', '才好吃',
                 '好吃', '最好吃', '好喝', '最好喝',
                 '用法', '画法', '技巧', '基本', '基础', '自制', '配方', '全过程', '全部过程']

    howto_word = ["怎", "么", "做", "法", '的', "教", '程', '制', '作', '是', '最', '和']

    ign_kp = KeywordProcessor()
    ign_kp.add_keywords_from_dict({
        'howto': howto_lis,
        'howtoword': howto_word})

    for word, l, r in reversed(ign_kp.extract_keywords(text, True)):
        text = text[:l] + ' ; ' + text[r:]
    for word, l, r in reversed(ign_kp.extract_keywords(text, True)):
        text = text[:l] + text[r:]
    text = text.replace(' ; ', '')
    for word, l, r in reversed(ign_kp.extract_keywords(text, True)):
        text = text[:l] + text[r:]
    return text
