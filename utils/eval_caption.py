from pycocoevalcap.bleu.bleu import Bleu as _Bleu
from pycocoevalcap.cider.cider import Cider as _Cider
from pycocoevalcap.rouge.rouge import Rouge as _Rouge
from typing import Dict
from concurrent import futures
from lumo import Params, Logger
import shutil
from itertools import cycle
from torchtext.data.metrics import bleu_score
from typing import List
import os
import numpy as np
from lumo.contrib.itertools import window2


def adapter_coco(pred: str, gt):
    if isinstance(gt, str):
        gt = gt.split(',')
    if isinstance(pred, str):
        pred = pred.split(',')
    return {'0': pred}, {'0': gt}


def adapter_coco_batched(preds: list, gts: list):
    pred_dict = {}
    gt_dict = {}
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        pred, gt = adapter_coco(pred, gt)
        pred_dict[i] = pred['0']
        gt_dict[i] = gt['0']

    return pred_dict, gt_dict


class EvalPM(Params):
    def __init__(self):
        super().__init__()

        self.eval_file = None
        self.f1 = None  # gen
        self.f2 = None  # gt
        self.log_root = None

        self.user = self.choice('yhz2', 'yhz', 'jmz')
        self.mod = 1
        self.skip = 0
        self.window_size = 64

        self.r = True
        self.c = True
        self.bleu = 4

    def iparams(self):
        if self.log_root is None:
            if self.eval_file == None:
                self.log_root = self.eval_root
            else:
                self.log_root = f'{os.path.dirname(self.eval_file)}'


class Evaluate:
    def __init__(self, n=None) -> None:
        super().__init__()
        self.n = n
        self.metric = None

    def __str__(self):
        return f"{self.__class__.__name__}"

    def computer_score(self, src, tgt) -> Dict[str, np.ndarray]:
        return {str(self): self.metric.compute_score(tgt, src)[0]}

    def __repr__(self) -> str:
        return self.__str__()


class Rouge(Evaluate):

    def __init__(self, n=None) -> None:
        super().__init__(n)
        self.metric = _Rouge()


class Cider(Evaluate):

    def __init__(self, n=None) -> None:
        super().__init__(n)
        self.metric = _Cider()


class Bleu(Evaluate):

    def __init__(self, n=4) -> None:
        super().__init__(n)
        self.metric = _Bleu(n)

    def computer_score(self, src, tgt, n=None):
        old_n = self.metric._n
        if n is not None:
            self.metric = _Bleu(n)

        try:
            res = {f'{str(self)}{i + 1}': np.array(res)
                   for i, res
                   in enumerate(list(self.metric.compute_score( tgt,src))[0])}
        except IndexError as e:
            res = self.computer_score(src, tgt, self.n - 1)
            for i in range(self.n - 1, self.n + 1):
                res[f'{str(self)}{i}'] = np.zeros_like(res[f'{self}1'])
        except Exception as e:
            print(e)
            raise e

        self.metric = _Bleu(old_n)
        return res


def file_for_v1(file):
    with open(file, 'r', encoding='utf-8') as r:
        for line in r:
            a, b, c = line.split(',')
            yield a, c.strip(), b.strip()


def file_for_yhz2(file, file2):
    with open(file, 'r', encoding='utf-8') as r, open(file2, 'r', encoding='utf-8') as r2:
        for line1, line2 in zip(r, r2):
            a, b, c = 'empty', line1.replace(' ', '').strip(), line2.strip()
            yield a, ' '.join(b), ' '.join(c)


def main():
    log = Logger()

    pm = EvalPM()
    pm.from_args()
    pm.iparams()
    # log.raw(pm)

    evals = []
    if int(pm.bleu) > 0:
        evals.append(Bleu(int(pm.bleu)))
    if pm.r:
        evals.append(Rouge())
    if pm.c:
        evals.append(Cider())

    open_fn = file_for_yhz2

    lines = [[item_id, *adapter_coco(pred, gt)] for item_id, pred, gt in open_fn(pm.f1, pm.f2)]

    item_ids, preds, gts = list(zip(*lines))

    execurot = futures.ProcessPoolExecutor(max_workers=len(evals))

    preds = {i: list(val.values())[0] for i, val in enumerate(preds)}
    gts = {i: list(val.values())[0] for i, val in enumerate(gts)}

    ress = []
    for e in evals:
        ress.append(execurot.submit(e.computer_score, preds, gts))

    print('results:')
    for res in futures.as_completed(ress):
        print()
        for k, v in res.result().items():
            print(k, f'{np.mean(v):.3f}')

    #
    # for e in evals:
    #     s = list(execurot.map(e.computer_score, preds, gts))
    #     s = [list(i.values())[0] for i in s]
    #     print(e, s)

def cal_score_from_txt(gen_file,gt_file):
    log = Logger()

    pm = EvalPM()
    # pm.from_args()
    # pm.iparams()
    # log.raw(pm)

    evals = []
    if int(pm.bleu) > 0:
        evals.append(Bleu(int(pm.bleu)))
    if pm.r:
        evals.append(Rouge())
    if pm.c:
        evals.append(Cider())

    open_fn = file_for_yhz2

    # lines = [[item_id, *adapter_coco(pred, gt)] for item_id, pred, gt in open_fn(pm.f1, pm.f2)]
    lines = [[item_id, *adapter_coco(pred, gt)] for item_id, pred, gt in open_fn(gen_file, gt_file)]

    item_ids, preds, gts = list(zip(*lines))

    execurot = futures.ProcessPoolExecutor(max_workers=len(evals))

    preds = {i: list(val.values())[0] for i, val in enumerate(preds)}
    gts = {i: list(val.values())[0] for i, val in enumerate(gts)}

    ress = []
    for e in evals:
        ress.append(execurot.submit(e.computer_score, preds, gts))
    scores_list = []
    # print('results:')
    scores_dic={}
    for res in futures.as_completed(ress):
        # print()
        for k, v in res.result().items():
            # print(k, f'{np.mean(v):.3f}')
            scores_dic[k]=(f'{np.mean(v)*100:.2f}')
    return scores_dic


if __name__ == '__main__':
    main()
