import torch.nn as nn
import torch
from torch import argmax

from torch.nn import functional, CrossEntropyLoss, MSELoss, Softmax, BCEWithLogitsLoss
from torchtext.data.metrics import bleu_score
from torch.distributions import Categorical, kl_divergence
import torch.nn.functional as F
from transformers.models.bart.modeling_bart import BartConfig, BartForConditionalGeneration, BartClassificationHead, \
    BartEncoder, BartAttention, BartDecoderLayer, BartEncoderLayer, BartForSequenceClassification, BartModel, \
    shift_tokens_right
from transformers.models.bert import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification

from utils.cal_matches import batch_match_equal
from utils.utils import _init_weights

from torch.nn import MultiheadAttention

from transformers.modeling_outputs import BaseModelOutput


class QueryVideoCaptionNet(nn.Module):

    def __init__(self, cfg):
        super(QueryVideoCaptionNet, self).__init__()

        self.cfg = cfg
        # self.ch_tk = BertTokenizer.from_pretrained("/home/jiamengzhao/.huggingface/bert-base-chinese")
        self.ch_tk = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
        self.model_gen = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")

        # self.bart_cfg = BartConfig(encoder_layers=0, decoder_layers=3, d_model=768, return_dict=True,
        #                            max_position_embeddings=180, vocab_size=21128,
        #                            pad_token_id=0, bos_token_id=101, eos_token_id=102, forced_eos_token_id=102,
        #                            decoder_start_token_id=101)
        # self.model_gen = BartForConditionalGeneration(self.bart_cfg)
        self.model_gen.apply(_init_weights)

        self.model_gen.init_weights()

        enc_cfg = BartConfig(encoder_layers=3, d_model=768, return_dict=True,
                             max_position_embeddings=100, vocab_size=21128)
        self.visual_encoder = BartEncoder(enc_cfg)
        self.visual_encoder.apply(_init_weights)

        # self.textual_encoder = BartEncoder(enc_cfg)
        # self.textual_encoder.apply(_init_weights)
        self.textual_encoder = self.model_gen.get_encoder()

        self.cross_modal_encoder = MultiheadAttention(embed_dim=768, num_heads=cfg.model.num_heads, kdim=768, vdim=768)
        self.cross_modal_encoder.apply(_init_weights)

        self.sgm = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction='none')  # already with a sigmoid layer

        self.proj = nn.Sequential(nn.Linear(768, cfg.model.proj_dim_1),
                                  nn.LeakyReLU(negative_slope=cfg.model.neg_slope, inplace=True),
                                  nn.Linear(cfg.model.proj_dim_1, cfg.model.proj_dim_2),
                                  nn.LeakyReLU(negative_slope=cfg.model.neg_slope, inplace=True),
                                  nn.Linear(cfg.model.proj_dim_2, 1))
        self.proj.apply(_init_weights)

        # self.pt_bert = BertModel.from_pretrained('bert-base-chinese')

    def forward(self, video_feat, video_feat_mask, text_idx, text_att_mask, tgt_ids, tgt_ids2,
                asr_in_q_idx, asr_in_q_mask, all_queries, chose_tgt_ids,
                mode='train', in_text_noun_mask=None,
                **kwargs):

        device = video_feat.device
        # text_inputs_ids = text_tokened['input_ids']
        # text_attention_mask = text_tokened['attention_mask']

        # visual (clip)
        visual_encoded = self.visual_encoder(inputs_embeds=video_feat, attention_mask=video_feat_mask).last_hidden_state

        # textual_encoded = self.textual_encoder(inputs_embeds=text_feat, attention_mask=text_feat_mask).last_hidden_state
        textual_encoded = self.textual_encoder(input_ids=text_idx, attention_mask=text_att_mask).last_hidden_state
        # mm_feat = visual_encoded
        # mm_att_mask = video_feat_mask

        mm_feat = torch.cat([visual_encoded, textual_encoded], dim=1)
        mm_att_mask = torch.cat([video_feat_mask, text_att_mask], dim=1)

        q_from_text = textual_encoded.transpose(1, 0)
        k_from_img = visual_encoded.transpose(1, 0)
        v_from_img = visual_encoded.transpose(1, 0)

        cross_modal_feat = self.cross_modal_encoder(query=q_from_text, key=k_from_img, value=v_from_img)[0].transpose(1,
                                                                                                                      0)
        if mode == 'train':
            cross_modal_feat_for_bce = self.sgm(self.proj(cross_modal_feat).view([-1, 1]))
            cap_words_label_for_bce = in_text_noun_mask.view([-1, 1])

            cls_loss = self.bceloss(cross_modal_feat_for_bce, cap_words_label_for_bce)
            cls_att_mask = text_att_mask.view([-1, 1])
            cls_loss_masked = cls_loss * cls_att_mask
            cls_loss_mean = torch.sum(cls_loss_masked) / torch.sum(cls_att_mask)

        cross_modal_feat_for_pick = self.sgm(self.proj(cross_modal_feat))
        selceted_mask = (cross_modal_feat_for_pick > self.cfg.model.cls_threshold).long().squeeze(dim=-1)
        selceted_idx = selceted_mask * text_idx

        selceted_list = self.ch_tk.batch_decode(selceted_idx, skip_special_tokens=True)

        selceted_str_retoken = self.ch_tk(selceted_list, max_length=20, padding='max_length', truncation=True,
                                          return_tensors='pt')
        selceted_str_retoken_ids = selceted_str_retoken['input_ids'].to(device)
        selceted_str_retoken_att_mask = selceted_str_retoken['attention_mask'].to(device)

        # noun_feat = self.model_gen.get_input_embeddings()(selceted_str_retoken_ids)
        noun_feat = self.textual_encoder(input_ids=selceted_str_retoken_ids,
                                         attention_mask=selceted_str_retoken_att_mask).last_hidden_state

        mm_feat = torch.cat([visual_encoded, noun_feat, textual_encoded], dim=1)
        mm_att_mask = torch.cat([video_feat_mask, selceted_str_retoken_att_mask, text_att_mask], dim=1)

        textual_encoded = torch.cat([noun_feat, textual_encoded], dim=1)
        text_feat_mask = torch.cat([selceted_str_retoken_att_mask, text_att_mask], dim=1)


        v_enc_output = BaseModelOutput(last_hidden_state=visual_encoded)
        t_enc_output = BaseModelOutput(last_hidden_state=textual_encoded)
        mm_enc_output = BaseModelOutput(last_hidden_state=mm_feat)

        # textual (bart embedding)
        # textual_feat = self.model_gen.get_input_embeddings()(text_inputs_ids)

        target_ids_masked = chose_tgt_ids.masked_fill(chose_tgt_ids == self.ch_tk.pad_token_id, -100)

        if mode == 'eval':
            with torch.no_grad():
                # encoder_output = self.model_gen.get_encoder()(**text_tokened)
                # encoder_output = self.model_gen.get_encoder()(inputs_embeds=mm_feat, attention_mask=mm_attention_mask)
                # decoder_start_token_id = torch.cat([decoder_input_ids_noun,decoder_input_ids_label[:,:1]],dim=1)
                # gen_result = self.model_gen.generate(decoder_start_token_id=decoder_start_token_id, encoder_outputs=mm_enc_output,
                #                                      attention_mask=mm_att_mask,max_length=40)
                # gen_result_out = gen_result[:,20:]
                # query_infer = self.ch_tk.batch_decode(gen_result_out, skip_special_tokens=True)
                gen_result = self.model_gen.generate(encoder_outputs=mm_enc_output,
                                                     attention_mask=mm_att_mask, max_length=20)
                query_infer = self.ch_tk.batch_decode(gen_result, skip_special_tokens=True)

            return query_infer

        # gen = self.model_gen(**text_tokened, labels=target_ids_masked)
        # gen = self.model_gen(inputs_embeds=mm_feat, attention_mask=mm_attention_mask, labels=target_ids_masked2)

        # gen = self.model_gen(encoder_outputs=mm_enc_output, attention_mask=mm_att_mask,
        #                 decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)

        gen = self.model_gen(encoder_outputs=mm_enc_output,
                             attention_mask=mm_att_mask, labels=target_ids_masked)


        genv = self.model_gen(encoder_outputs=v_enc_output,
                              attention_mask=video_feat_mask, labels=target_ids_masked)
        gent = self.model_gen(encoder_outputs=t_enc_output,
                              attention_mask=text_feat_mask, labels=target_ids_masked)

        vid_out_ids = argmax(genv.logits, dim=2)
        text_out_ids = argmax(gent.logits, dim=2)
        vid_out_query = self.ch_tk.batch_decode(vid_out_ids, skip_special_tokens=True)
        text_out_query = self.ch_tk.batch_decode(text_out_ids, skip_special_tokens=True)

        vid_score = (torch.Tensor(batch_match_equal(all_queries, vid_out_query)).to(device))
        text_score = (torch.Tensor(batch_match_equal(all_queries, text_out_query)).to(device))

        loss_fct = CrossEntropyLoss(reduction='none')

        masked_lm_loss_v = loss_fct(genv.logits.transpose(1, 2), target_ids_masked)
        masked_lm_loss_t = loss_fct(gent.logits.transpose(1, 2), target_ids_masked)

        logits_for_loss = gen.logits.transpose(1, 2)
        masked_lm_loss = loss_fct(logits_for_loss, target_ids_masked)
        batch_loss_v = torch.mean(masked_lm_loss_v, dim=1)
        batch_loss_t = torch.mean(masked_lm_loss_t, dim=1)
        batch_loss = torch.mean(masked_lm_loss, dim=1)

        mean_v_loss = torch.mean(batch_loss_v * vid_score)
        mean_t_loss = torch.mean(batch_loss_t * text_score)

        mean_loss = torch.mean(batch_loss)

        gen_loss = (mean_loss + mean_v_loss + mean_t_loss) / 3

        loss = [gen_loss, cls_loss_mean]
        # loss = [gen_loss]

        return loss
