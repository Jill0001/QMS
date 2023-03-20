import os
import time
import inspect
import math
import sys
from tqdm import tqdm
from pprint import pformat

import torch
# import torchtext.data
import torch.nn as nn
import torch.optim as optim

from accelerate import Accelerator
# from accelerate.kwargs_handlers import DistributedDataParallelKwargs
from transformers.models.bart.modeling_bart import BartConfig, BartForConditionalGeneration
from transformers.models.bert import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.qms_model import QueryVideoCaptionNet
from models import qms_model as model_file

from torch.utils.data import DataLoader
from vq_dataset import VidCapDataset
from utils.utils import setup_seed, writr_gt, send_to_device
from utils.eval_caption import cal_score_from_txt

# from lumo.calculate.schedule import CosScheduler
from lumo import Meter, Logger

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.utils import make_exp_dirs

log = Logger()


def get_pretrained_model(model, saved_dir):
    saved_models = os.listdir(saved_dir)
    if len(saved_models) != 0:
        saved_models.sort()
        from_ep = saved_models[-1][5] + saved_models[-1][6] + saved_models[-1][7]
        saved_model_path = os.path.join(saved_dir, saved_models[-1])
        state_dict = torch.load(saved_model_path)
        model.load_state_dict(state_dict)
        log.info('Load state dict from %s' % str(saved_model_path))
    else:
        from_ep = -1
        log.info('Initialized randomly (with seed)')
    return model, int(from_ep)


def get_pretrained_model_giving_ep(model, saved_dir, ep):
    saved_models = os.listdir(saved_dir)
    saved_models.sort()
    saved_model_path = os.path.join(saved_dir, saved_models[ep])

    state_dict = torch.load(saved_model_path)
    model.load_state_dict(state_dict)
    log.info('Load state dict from %s' % str(saved_model_path))
    return model


def load_data(ft_file, cfg):
    tkr = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    train_dataset = VidCapDataset(ft_file, 'train', tkr, cfg)
    eval_dataset = VidCapDataset(ft_file, 'eval', tkr, cfg)
    test_dataset = VidCapDataset(ft_file, 'test', tkr, cfg)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, num_workers=8, shuffle=True,
                                  drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg.train.batch_size, num_workers=8, shuffle=False,
                                 drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, num_workers=8, shuffle=False,
                                 drop_last=False)

    return train_dataloader, eval_dataloader, test_dataloader


def eval_net(ep, model, loader, log_path, device, tp='test'):
    log_txt_name = os.path.join(log_path, ('ep_' + str(ep) + '_' + tp + '.txt'))
    log_txt = open(log_txt_name, 'w')

    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(loader)):
            batch = send_to_device(batch, device)
            query_infer = model(**batch, mode='eval')

            log_txt.write('\n'.join(query_infer) + '\n')

    log_txt.flush()
    log_txt.close()

    if tp == 'val':
        gt_name = os.path.join(log_path, 'gt4eval.txt')
        scores = cal_score_from_txt(log_txt_name, gt_name)
        scores_to_print = [(k, scores[k]) for k in sorted(scores.keys())]
    else:
        gt_name = os.path.join(log_path, 'gt4test.txt')
        scores = cal_score_from_txt(log_txt_name, gt_name)

        log.info('-' * 20 + 'Test Stage' + '-' * 20)
        scores_to_print = [(k, scores[k]) for k in sorted(scores.keys())]

    log.info("Epoch: ", ep, scores_to_print)

    return scores


def load_model(model, save_path, epoch):
    model_pth_path = os.path.join(save_path, 'epoch' + str(epoch).zfill(3) + '.pth')
    state_dict = torch.load(model_pth_path)
    model.load_state_dict(state_dict)
    return model


def save_model(model, save_path, epoch, accelerator):
    log.info('Saving Net...')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(),
                     os.path.join(save_path, 'epoch' + str(epoch).zfill(3) + '.pth'))


def run_stage(cfg, model, lr_sche, opt,
              train_loader, eval_loader, test_loader,
              log_path, device,
              accelerator):
    print_every = 100
    eval_every = 1
    save_every = 1

    max_epoch = cfg.train.max_epoch

    best_res_dic = {}
    best_res_dic['B1'] = 0.0
    best_res_dic['B2'] = 0.0
    best_res_dic['B3'] = 0.0
    best_res_dic['B4'] = 0.0
    best_res_dic['RL'] = 0.0
    best_res_dic['Cdr'] = 0.0
    best_B2_ep = 0

    step = 0
    for epoch in range(max_epoch):

        log.info('-' * 20, ' Current Epoch: ', epoch, '-' * 20)
        # torch.cuda.empty_cache()
        # avgmeter = AvgMeter()
        meter = Meter()
        time_now = time.time()
        show_loss = 0

        for idx, batch in enumerate(train_loader):
            model.train()
            step += 1
            opt.zero_grad()

            batch = send_to_device(batch, device)
            loss = model(**batch)
            meter.gen_loss = loss[0]
            if len(loss) > 1:
                meter.cls_loss = loss[-1]
                weight = cfg.model.weight_cls
                loss_mean = ((1 - weight) * loss[0]) + (weight * loss[1])
            else:
                loss_mean = loss[0]
            accelerator.backward(loss_mean)
            opt.step()
            show_loss += loss_mean
            if idx % print_every == print_every - 1 and accelerator.is_main_process:
                cost_time = time.time() - time_now
                time_now = time.time()
                log.info(
                    f'lr: {cur_lr:.6f} | step: {idx + 1}/{len(train_loader) + 1} | time cost {cost_time:.2f}s | loss: {(show_loss / print_every):.4f}')
                show_loss = 0
                log.newline()

            cur_lr = opt.param_groups[-1]['lr']
            meter.lr = f"{cur_lr:.5f}"

            lr_sche.step()

        log.info('current lr: ', opt.param_groups[-1]['lr'])
        log.info(cfg.name)

        if (epoch % eval_every) == (eval_every - 1) and epoch >= 0:

            log.info('Evaluating Net...')
            scores = eval_net(epoch, model, eval_loader, log_path, device, 'eval')

            if float(scores['Bleu1']) >= best_res_dic['B1']:
                best_res_dic['B1'] = float(scores['Bleu1'])
            if float(scores['Bleu2']) >= best_res_dic['B2']:
                best_res_dic['B2'] = float(scores['Bleu2'])
                best_B2_ep = epoch
            if float(scores['Bleu3']) >= best_res_dic['B3']:
                best_res_dic['B3'] = float(scores['Bleu3'])
            if float(scores['Bleu4']) >= best_res_dic['B4']:
                best_res_dic['B4'] = float(scores['Bleu4'])
            if float(scores['Rouge']) >= best_res_dic['RL']:
                best_res_dic['RL'] = float(scores['Rouge'])
            if float(scores['Cider']) >= best_res_dic['Cdr']:
                best_res_dic['Cdr'] = float(scores['Cider'])

        if (epoch % save_every) == (save_every - 1):
            save_model(model, log_path, epoch, accelerator=accelerator)

    ## generate and calculate metrics
    model = load_model(model, log_path, best_B2_ep)
    log.info(f'Model Loaded! Best ep: {best_B2_ep}')
    test_scores = eval_net(best_B2_ep, model, test_loader, log_path, device, 'test')

    return test_scores


@hydra.main(config_path="conf", config_name="basic_cfg", version_base='1.2.0')
def main(cfg: DictConfig):
    accelerator = Accelerator()
    device = accelerator.device

    log_path = make_exp_dirs(cfg.name)
    log.add_log_dir(log_path)

    model = QueryVideoCaptionNet(cfg)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    # optimizer = optim.AdamW(nn.ModuleList(
    #     [model.pt_bert]).parameters(), lr=5e-5, weight_decay=1e-5)
    # optimizer2 = optim.AdamW(nn.ModuleList(
    #     [model.model_gen,model.visual_encoder]).parameters(), lr=2e-4, weight_decay=1e-5)
    # optimizer2 = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    ft_file = 'dataset/qms_dataset.ft'
    train_dataloader, val_dataloader, test_dataloader = load_data(ft_file, cfg)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                        T_max=int(cfg.train.max_epoch) * len(train_dataloader))

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # log model file into log
    log.info(inspect.getsource(model_file))
    log.info('Found device: %s' % device)

    str_cfg = OmegaConf.to_yaml(cfg)
    log.info('Config: \n', (str_cfg))

    log.info('train data: ', cfg.train.batch_size * len(train_dataloader))
    log.info('val data: ', cfg.train.batch_size * len(val_dataloader))
    log.info('test data: ', cfg.train.batch_size * len(test_dataloader))

    writr_gt(val_dataloader, test_dataloader, log_path)

    best_res_dic = run_stage(cfg=cfg,
                             model=model,
                             lr_sche=lr_scheduler, opt=optimizer,
                             train_loader=train_dataloader,
                             eval_loader=val_dataloader,
                             test_loader=test_dataloader,
                             log_path=log_path, device=device,
                             accelerator=accelerator)

    log.info('Config: \n', (str_cfg))
    log.info(pformat(best_res_dic))
    os.system('rm -rf qms_train.log')


if __name__ == '__main__':
    main()
