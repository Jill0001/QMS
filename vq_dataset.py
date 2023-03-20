import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import json
import os
import kshingle as ks
import cv2
import sys

import ast

import hydra
from omegaconf import DictConfig, OmegaConf

from copy import deepcopy
from flashtext import KeywordProcessor
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import BertTokenizer, BertModel
from joblib import hash, Memory
import jieba.posseg

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import collections
from transformers import BertModel

mem = Memory('./.cache')


# ch_tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
# ch_tokenizer = BertTokenizer.from_pretrained("/home/jiamengzhao/.huggingface/bert-base-chinese")


# ch_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


def get_vid_npy(video_path, vid=None):
    video_path = video_path.replace('features', 'clip_feature_23w').replace('.npy', '_clip.npy')  # 用clip feature
    # video_path= video_path.replace('features','resnet_feature_23w').replace('.npy','_resnet.npy') # resnet feature
    # video_path= os.path.join('/home/jiamengzhao/mmud/i3d_feat_5w/',vid+".npy") # i3d feature

    vid_feat = np.load(video_path, allow_pickle=True).astype(np.float32)
    if len(vid_feat.shape) == 4:
        try:
            vid_feat = vid_feat.squeeze(axis=1).squeeze(axis=1)
        except Exception:
            # print(vid_feat.shape)
            vid_feat = vid_feat[:, 1, 1, :]

    dim = vid_feat.shape[1]
    pad_np = np.zeros([32, dim], dtype=np.float32)
    pad_np[:len(vid_feat), :] = vid_feat

    mask_np = np.zeros([32], dtype=np.float32)
    mask_np[:len(vid_feat)] = 1
    return pad_np, mask_np


# def get_asr_text_npy(video_id):
#     bert_asr_23w_feat_dir = '/home/jiamengzhao/mmud/bert_asr100_23w/'
#     t_feat_name = os.path.join(bert_asr_23w_feat_dir,video_id+'.npy')
#     asr_feat = np.load(t_feat_name, allow_pickle=True).astype(np.float32)

#     pad_np = np.zeros([300, 768], dtype=np.float32)
#     pad_np[:len(asr_feat), :] = asr_feat

#     mask_np = np.zeros([300], dtype=np.float32)
#     mask_np[:len(asr_feat)] = 1

#     return pad_np,mask_np

def get_asr_words_label(in_q_srt, text, max_length):
    text = text[:max_length]
    in_q_list = in_q_srt.split(' ')
    res_all = []
    in_text_noun_mask = torch.zeros(max_length)
    for word in in_q_list:
        if word == '':
            continue
        else:
            word_len = len(word)
            res = [i for i in range(len(text)) if text.startswith(word, i)]
            for r in res:
                in_text_noun_mask[r + 1:r + word_len + 1] = 1

    return in_text_noun_mask


# def read_feather(ft_file, split):
#     df = pd.read_feather(ft_file)

#     # df = df[df['category_name'] == '烹饪'].reset_index()[df.columns]
#     df = df[df['type'] == split].reset_index()[df.columns]
#     # df = df.sort_values('keyword')

#     item_ids = llist(df['item_id'])
#     querys = llist(df['kw_less'])

#     # ocr_content_tru = df['ocr_content'].apply(lambda x: x[:max(len(x), 20)], 1)
#     # tag_tru = df['hashtag'].apply(lambda x: x[:max(len(x), 20)], 1)
#     # cap_all = df['caption'].apply(lambda x: x[:max(len(x), 20)], 1)
#     asr_all = df['asr_pure'].apply(lambda x: x[:max(len(x), 300)], 1)
#     # ocr_all = df['ocr_cover'].apply(lambda x: x[:max(len(x), 50)], 1)
#     asr_in_q = df['asr_in_q'].apply(lambda x: ' '.join(x)[:max(len(x), 20)], 1)

#     text_ensemble = llist(asr_all)

#     # asr_raw = df['asr_result']
#     # ocr_raw = df['ocr_result']

#     # text_graph = llist(df['text_graph'])

#     paths = llist(df['path'])
#     # frame_paths = llist(df['frame_paths'])

#     return item_ids, querys, text_ensemble,paths,asr_in_q


@mem.cache
def get_kwgraph(item_ids, querys):
    regraph = {}
    for idx, (item_id, query) in enumerate(zip(item_ids, querys)):
        [regraph.setdefault(i, []).append([idx]) for i in query.split(',')]

    return regraph


class PosSample():
    def __init__(self, regraph, db_pos=None):
        self.regraph = regraph
        self.db_pos = db_pos

    def __call__(self, mem):
        indexs = self.regraph[mem['query']]
        anc = mem['idx']
        pos = [i for i in indexs if i != anc]
        # print(anc, pos)
        if len(pos) > 0:
            pos = random.choice(pos)[0]
        else:
            pos = anc
        mem_pos = self.db_pos[pos]

        newmem = {}

        for k, v in mem.items():
            newmem[k] = v
            if k in mem_pos:
                newmem[f'pos_{k}'] = mem_pos[k]
        return newmem


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class VidCapDataset(Dataset):

    def __init__(self, ft_file, split, tokenizer, cfg):
        self.split = split
        self.cfg = cfg
        df = pd.read_feather(ft_file)

        # df = df[df['category_name'] == '烹饪'].reset_index()[df.columns]
        df = df[df['type'] == split].reset_index()[df.columns]
        # df = df.sort_values('keyword')

        self.item_ids = list(df['item_id'])
        self.querys = list(df['kw_less'])
        self.query_sim = list(df['query_sim'])

        # ocr_content_tru = df['ocr_content'].apply(lambda x: x[:max(len(x), 20)], 1)
        # tag_tru = df['hashtag'].apply(lambda x: x[:max(len(x), 20)], 1)
        # self.cap_all = df['caption'].apply(lambda x: x[:max(len(x), 20)], 1)
        self.asr_all = df['asr_pure'].apply(lambda x: x[:max(len(x), self.cfg.dataset.max_text)], 1)
        # self.ocr_all = df['ocr_cover'].apply(lambda x: x[:max(len(x), 50)], 1)
        self.asr_in_q = df['asr_in_q'].apply(lambda x: ' '.join(x)[:max(len(x), 20)], 1)

        self.text_ensemble = list(self.asr_all)

        # asr_raw = df['asr_result']
        # ocr_raw = df['ocr_result']

        # text_graph = list(df['text_graph'])

        self.paths = list(df['path'])
        # frame_paths = list(df['frame_paths'])

        # self.regraph = get_kwgraph(self.item_ids, self.querys)
        self.cfg = cfg
        self.tokenizer = tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    def __len__(self):
        if self.cfg.debug:
            if self.split == 'train':
                return 500
            else:
                return 50
        else:
            return len(self.item_ids)

    def token_text(self, text, max_length):

        encoded_dict = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt')

        return encoded_dict

    def query_sim_to_probs(self, query_sim):
        mask_th = self.cfg.dataset.mask_th
        exp_dim = self.cfg.dataset.exp_dim
        sim_len = len(query_sim)
        sim_list_mask = query_sim > mask_th
        sim_mask_len = sim_list_mask.sum()
        sim_masked_sub = (((1 / (query_sim))) ** exp_dim) * sim_list_mask
        # sim_masked_sub = ((1/(query_sim-mask_th))**exp_dim)*sim_list_mask
        sim_sumed = sim_masked_sub.sum()
        if sim_sumed == 0:
            sim_probs = np.ones(sim_len) / sim_len
        else:
            sim_probs = sim_masked_sub / sim_sumed

        return sim_probs

    def choose_in_probs(self, probs, query_list):
        query_num = len(query_list)
        indexes = np.array(range(query_num))
        try:
            chose = np.random.choice(a=indexes, size=1, replace=True, p=probs)
        except Exception:
            chose = np.random.choice(a=indexes, size=1, replace=True)

        return query_list[int(chose)]

    def idx2item(self, idx):
        item_id = self.item_ids[idx]

        all_queries = self.querys[idx]
        querys_list = all_queries.split(',')

        query = random.choice(querys_list)
        query2 = random.choice(querys_list)

        query_sim = self.query_sim[idx]
        probs = self.query_sim_to_probs(query_sim)

        chose_query = self.choose_in_probs(probs, querys_list)

        asr_in_q = self.asr_in_q[idx]
        origin_text = self.text_ensemble[idx]

        video_feat, video_feat_mask = get_vid_npy(self.paths[idx], item_id)
        # text_feat,text_feat_mask = get_asr_text_npy(item_id)

        encoded_dict = self.token_text(query, max_length=20)
        tgt_ids = encoded_dict['input_ids'][0]
        tgt_att_mask = encoded_dict['attention_mask'][0]

        encoded_dict = self.token_text(query2, max_length=20)
        tgt_ids2 = encoded_dict['input_ids'][0]

        encoded_dict = self.token_text(chose_query, max_length=20)
        chose_tgt_ids = encoded_dict['input_ids'][0]

        encoded_dict = self.token_text(origin_text, max_length=self.cfg.dataset.max_text)
        text_idx = encoded_dict['input_ids'][0]
        text_att_mask = encoded_dict['attention_mask'][0]

        encoded_dict = self.token_text(asr_in_q, max_length=20)
        asr_in_q_idx = encoded_dict['input_ids'][0]
        asr_in_q_mask = encoded_dict['attention_mask'][0]

        in_text_noun_mask = get_asr_words_label(asr_in_q, origin_text, max_length=self.cfg.dataset.max_text)

        outdic = {'item_id': item_id,
                  "query": query,
                  'all_queries': all_queries,
                  "tgt_ids": tgt_ids,
                  "tgt_ids2": tgt_ids2,
                  'tgt_att_mask': tgt_att_mask,

                  'text': origin_text,
                  'text_idx': text_idx,
                  'text_att_mask': text_att_mask,
                  'asr_in_q_idx': asr_in_q_idx,
                  'asr_in_q_mask': asr_in_q_mask,

                  'video_feat': video_feat,
                  'video_feat_mask': video_feat_mask,

                  "retri_q": query,
                  "retri_c": query2,

                  # 'text_feat':text_feat,
                  # "text_feat_mask":text_feat_mask,
                  'chose_query': chose_query,
                  'chose_tgt_ids': chose_tgt_ids,

                  "in_text_noun_mask": in_text_noun_mask}

        return outdic

    def __getitem__(self, idx):

        origin_dic = self.idx2item(idx)
        # pair_idx = random.choice(self.regraph[origin_dic['query']][0])
        # pair_dic = self.idx2item(pair_idx)
        # pair_dic_rename = {}
        # for k in pair_dic:
        #     pair_dic_rename[f'{k}_pair'] = pair_dic[k]
        # origin_dic.update(pair_dic_rename)

        return origin_dic

# class Coll(CollateBase):

#     def __init__(self) -> None:
#         super().__init__()

#     def before_collate(self, sample_list):
#         return sample_list

#     def after_collate(self, batch):

#         if 'query' in batch.keys():
#             batch['tgt_tokened'] = ch_tokenizer(batch['query'], return_tensors='pt', padding=True, truncation=True,
#                                                 return_token_type_ids=False,
#                                                 max_length=20)
#         if 'query2' in batch.keys():
#             batch['tgt_tokened2'] = ch_tokenizer(batch['query2'], return_tensors='pt', padding=True, truncation=True,
#                                                 return_token_type_ids=False,
#                                                 max_length=20)

#         if 'origin_text' in batch.keys():
#             batch['text_tokened'] = ch_tokenizer(batch['origin_text'], return_tensors='pt', padding='max_length',
#                                                         return_token_type_ids=False,
#                                                         truncation=True, max_length=100)
#         if 'asr_in_q' in batch.keys():
#             batch['asr_in_q_tokened'] = ch_tokenizer(batch['asr_in_q'], return_tensors='pt', padding=True,
#                                                         return_token_type_ids=False,
#                                                         truncation=True, max_length=20)

#         return batch


# @hydra.main(config_path="conf", config_name="basic_cfg")
# def main(cfg : DictConfig):
#     ft_file = '/home/jiamengzhao/share/video_meta/23w_selected_asr_pure_with_path_split_ocr_framepath_lesskw_qnouns_capinq_asrinq_qsim.ft'
#         # ft_file = '/home/jiamengzhao/share/video_meta/dish_v6.ft'

#     dataset = VidCapDataset(ft_file, 'train', cfg)

#     # ch_tk = BertTokenizer.from_pretrained("/home/jiamengzhao/.huggingface/bert-base-chinese")


#     dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

#     for i, data in tqdm(enumerate(tqdm(dataloader))):
#         print(data['choose_query'])
#         print(data['query'])
#         if i >5:
#             exit()


# if __name__ == '__main__':

#    main()

# 用来抽bert特征并存储
# device = torch.device('cuda')
# bert_asr_23w_feat_dir = '/home/jiamengzhao/mmud/bert_asr100_23w/'
# text_embedding_model = BertModel.from_pretrained('bert-base-chinese').to('cuda')
# text_embedding_model.eval()
# for i, data in tqdm(enumerate(tqdm(dataloader))):
# item_id = data['item_id']
# t_feat_name = os.path.join(bert_asr_23w_feat_dir,item_id[0]+'.npy')
# textual_feat = text_embedding_model(input_ids=data['text_tokened']['input_ids'].to('cuda'),
#                         attention_mask=data['text_tokened']['attention_mask'].to('cuda')).last_hidden_state[0].detach().cpu().numpy()

# np.save(t_feat_name, textual_feat)
