
import json
import numpy as np
from joblib import Parallel, delayed
# from lumo.contrib import pickle
import pandas as pd
from flashtext import KeywordProcessor
import regex as re
from collections import Counter
# from lumo.contrib.itertools import window
from lumo.contrib.string.lcs import LCS, Match
from itertools import combinations, permutations, chain
from lumo.decorators.regist import Register
from typing import List, Sized, Iterable,Sequence

from itertools import zip_longest, chain, accumulate, repeat
from operator import add

cat_regist = Register('cat_process')

from lumo import Logger

from dataclasses import dataclass, field
from collections import namedtuple

Chunk = namedtuple('Chunk', 'w a size')


def window(seq: Sized, n: int, strid: int = 1, drop_last: bool = False):
    for i in range(0, len(seq), strid):
        res = seq[i:i + n]
        if drop_last and len(res) < n:
            break
        yield res

def merge_text(*args):
    return ','.join([i for i in args if len(i) > 0])


def create_mask(*args, txt):
    return np.sum([np.zeros(len(txt))] + [i.mask for i in args], axis=0)


ign_kp = KeywordProcessor()
# stop_words = ["怎么","做法","好吃","做","大全","的","教程","制作","吃","吃法","简单","方法","步骤","么","美味","家常","营养","和","什么","吃","吃货","视频","又","功效","好处",]
# kp.add_keywords_from_list(stop_words)
# howto_lis = ["怎么", "做法", '的做法', '的正宗做法', '正宗做法', '的正确做法', '正确做法',
#               "教程", "制作", "方法", '制作方法', '的制作方法', '怎么做', '使用技巧', '怎样可以', '的操作方法', '的方法',
#               '吃法', "步骤", "功效", "好处", "是什么", '教学', '怎么样',
#               '是什么意思', '怎么办', '图解', '攻略', '配方', '作用', '用途', '技巧',
#               '小技巧', '妙用', '妙招', '小妙招', '讲解', '解释', '教学', '练习', '方式']
howto_lis = ["怎么", '怎么做','怎样' ,'怎样做', "制作",  "怎么制作",  "怎样制作", 
            "做法",'方法', '吃法',"教程", '教学', "完整教程",'入门教程',
            '正宗做法', '家庭做法','的家庭做法','家常做法','正确做法', '制作方法', 
            '怎么做才好吃','怎么做好吃','才好吃','视频教程',
              '好吃' ,'最好吃',  '好喝' ,'最好喝',
               '用法' ,'画法' ,'技巧','基本','基础','自制','配方','全过程','全部过程']

howto_lis = sorted(howto_lis,key=len,reverse=True)

howto_word = ["怎","么", "做","法", '的',"教",'程','制','作','是','最','和','吃']
        
ign_kp.add_keywords_from_dict({
    # 'howto': howto_lis+howto_word,
    # 'howtoword':howto_word
})

def extract_howto(ori, kp):
    """
    循环抽取word，保证所有的关键词都被抽取出来
    """
    nex = ori
    cur = True
    words = []
    while cur:
        cur_words = kp.extract_keywords(nex, span_info=True)
        if len(cur_words) == 0:
            cur = False
        cw = []
        for tp, l, r in cur_words:
            cw.append([nex[l:r], l, r])
            nex = nex[:l] + ' ' * (r - l) + nex[r:]
            break

        words.extend(cw)

    return words, nex

def de_stop_words(text, kp):
    return extract_howto(text,kp)[1].replace(' ','')
    for word, l, r in reversed(kp.extract_keywords(text, True)):
        text = text[:l] + ' ; ' + text[r:]
    for word, l, r in reversed(kp.extract_keywords(text, True)):
        text = text[:l] + text[r:]
    text = text.replace(' ; ', '')
    for word, l, r in reversed(kp.extract_keywords(text, True)):
        text = text[:l] + text[r:]
    return text


@dataclass()
class MatchResult:
    ori: str
    lcs: int = 0  # 最长公共子序列长度
    r: float = 0  # 相似比例，lcs/len(kw)
    c: float = 0  # 完整程度，1 为完全相同，r =1 不代表 c =1 ，一个长度为 9 的 kw，里面有一个 4 和 5 的匹配，则 c = ((4*4)+(5*5))/(9*9)
    dist: int = 0  # 在 nt 中的相对位置关系，越大代表越靠后
    raw_lcs: int = 0
    raw_r: float = 0
    sub: List[str] = field(default_factory=list)  # 子序列
    loop: int = 0
    span: int = 0  # 匹配的keyword在原文中的 跨度
    mask: np.ndarray = None  # 匹配的keyword在原文中的 跨度


def _merge_match_blocks(ori, res):
    """
    a,b = '鬼步舞校长视频中领舞的人是网红学校的校长张鹏飞,也带领的习惯小学创办的','网红校长张鹏飞校长鬼步舞'
    res = lcs2('鬼步舞校长视频中领舞的人是网红学校的校长张鹏飞,也带领的习惯小学创办的','网红校长张鹏飞校长鬼步舞')
    ress = _merge_match_blocks(a,res)


    :param ori:
    :param blocks:
    :return:
    """
    if len(res) == 0:
        return []

    li, lj = res[0][1], res[0][3]
    size = 1
    ress = []
    sub = [[res[0][2], res[0][3]]]

    for x, i, y, j in res[1:]:
        offset = (i - li)
        if offset == 1:
            li, lj = i, j
            size += 1
        else:
            end = size - 1
            ress.append([[ori[li - end:li - end + size], li - end, size], sub])
            sub = []

            li, lj = i, j
            size = 1
        sub.append([y, j])
    end = size - 1
    ress.append([[ori[li - end:li - end + size], li - end, size], sub])

    for i in range(len(ress)):
        m = ress[i]
        l = ress[i - 1] if i > 0 else None
        r = ress[i + 1] if i < len(ress) - 1 else None
        if m[0][-1] == 1:
            if l is not None and len(l) > 0 and ori[l[0][1] + l[0][2]] == m[0][0]:
                l[0][0] = l[0][0] + m[0][0]
                l[0][2] += 1
                l[1] = l[1] + m[1]
                m.clear()
            elif r is not None and len(r) > 0 and ori[r[0][1] - 1] == m[0][0]:
                r[0][0] = m[0][0] + r[0][0]
                r[0][1] -= 1
                r[0][2] += 1
                r[1] = m[1] + r[1]
                m.clear()

    ress = [[Chunk(m[0][0], m[0][1], m[0][2]), [Chunk(mm[0], mm[1], 1) for mm in m[1]]] for m in ress if len(m) > 0]
    return ress


def _match_equal(txt, kw):
    """
    a 和 b 的匹配程度，返回四个分数

    :param txt: 文本
    :param kw:
    :return:
    """
    _ph_ = '_'  # placehold,占位符
    lcs = 0
    raw_lcs = 0
    raw_len = len(kw)
    skw = kw
    skw = de_stop_words(skw, ign_kp)
    txt = de_stop_words(txt, ign_kp)
    # devi_len = len(txt)

    res = MatchResult(kw)
    res.mask = np.zeros(len(txt))
    if len(skw) == 0:
        return res

    dist = 0
    c = 0

    sub = []  # 存匹配上的 keyword，用于抽取
    loop = 0

    nskw = skw
    sm = _merge_match_blocks(txt, LCS(txt, nskw).lcs2())

    spans = []

    while len(sm) > 0:
        loop += 1
        edit = False
        for m, cs in sm:
            edit = True
            raw_lcs += m.size

            txt = txt[:m.a] + '=' * m.size + txt[m.a + m.size:]
            res.mask[m.a:m.a + m.size] = 1

            for cc in cs:
                nskw = nskw[:cc.a] + _ph_ * cc.size + nskw[cc.a + cc.size:]

            if m.size <= 1:
                continue
            spans.append(m)

            lcs += m.size
            sub.append(m.w)
            dist += (m.a * m.size)
            for cc in cs:
                dist += (cc.a * cc.size)
            c += m.size ** 2

        if not edit or nskw.strip(_ph_) == '':
            break

        # 之前会有莫名的死循环现象，目前应该已经排除，但为了保险加了这一行
        if c > 100:
            break

        sm = _merge_match_blocks(txt, LCS(txt, nskw).lcs2())

    spans = sorted(spans, key=lambda x: x.a)
    span = 0
    for a, b in window(spans, 2, drop_last=True):
        span = max(b.a - a.a - a.size, span)

    res.lcs = lcs
    res.raw_lcs = raw_lcs
    res.raw_r = raw_lcs / len(skw)
    res.r = lcs / len(skw)
    res.c = c / (len(skw) ** 2)
    res.dist = dist
    res.sub = sub
    res.loop = loop
    res.span = span
    return res

def match_equal(txt, kw):
    rs = []
    for sub in txt.split(','):
        r = _match_equal(sub,kw)
        rs.append(r)
    max_rawr = max([i.raw_r for i in rs])
    rs[0].raw_r = max_rawr
    return rs[0]
    


def batch_match_equal(txt, kw):
    assert len(txt) == len(kw)
    batch_raw_r=[]
    for sample_id in range(len(kw)):
        sample_raw_score =match_equal(txt[sample_id].replace(' ', ''),kw[sample_id].replace(' ', '')).raw_r
        batch_raw_r.append(sample_raw_score)
    return batch_raw_r

def de_duplicated(keywords):
    hists = set()
    res = []
    for kw in sorted(keywords, key=len, reverse=True):
        if len(set(kw) - hists) > 0:
            res.append(kw)
            hists.update(kw)
    return res


del_kp = KeywordProcessor()
# stop_words = ["怎么","做法","好吃","做","大全","的","教程","制作","吃","吃法","简单","方法","步骤","么","美味","家常","营养","和","什么","吃","吃货","视频","又","功效","好处",]
# kp.add_keywords_from_list(stop_words)
del_kp.add_keywords_from_dict({
    'stop': ['电影', '电视剧', '有意思', '点击头像', '点击右侧头像', '右侧'],
    'epi': ['大全', '全集', '完整版'],
    'kwai': [
        '感谢家人们一路的支持与陪伴',
    ]
})
# stop_kp.add_keyword_from_file(os.path.join(os.path.dirname(__file__), 'stop_kp.txt'))

match_book = re.compile('[《》]')


def match_keyword(nt, kws, hashtag=None, kp=None):
    """
    硬匹配 keyword，只有和 nt 在全部程度上匹配起来的 keyword 才保留
    :param nt:
    :param kws:
    :return:
    """
    if hashtag is None:
        hashtag = set()
    else:
        hashtag = set(hashtag.split(','))
    subs = kws.split(',')

    # 去掉单字
    subs = [i for i in subs if len(i) >= 2]
    ress = [match_equal(nt, de_stop_words(kw, del_kp)) for kw in subs]

    ress = sorted(ress, key=lambda x: [-x.c, -x.lcs])

    sub = set()
    pos = []
    sneg = set()

    txt_mask = [np.zeros(len(nt))]

    for res in ress:
        sub.update(res.sub)
        if len(kp.extract_keywords(res.ori)) > 0 and res.r == 1 and res.span <= 10:
            pos.append(res.ori)
            txt_mask.append(res.mask)
        elif res.raw_r > 0.5:  # 全字匹配 50% 以上
            sneg.add(res.ori)
        elif len(kp.extract_keywords(res.ori)) > 0:
            sneg.add(res.ori)
        else:
            sneg.add(res.ori)

    txt_mask = np.sum(txt_mask, axis=0)

    hashkey = []
    npos = []
    for kw in pos:
        if kw in hashtag:
            hashkey.append(kw)
        else:
            npos.append(kw)

    pos = de_duplicated(npos)

    return [txt_mask, ','.join(sub), ','.join(hashkey), ','.join(pos), ','.join(sneg)]


if __name__ == '__main__':
    print(match_equal('大锅鱼的法公司', '大锅菜做法'))
