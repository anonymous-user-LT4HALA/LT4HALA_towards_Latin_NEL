# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from tqdm import tqdm
from pytorch_transformers.modeling_utils import CONFIG_NAME, WEIGHTS_NAME

# from pytorch_transformers.modeling_bert import (
#     BertPreTrainedModel,
#     BertConfig,
#     BertModel,
# )

# from pytorch_transformers.modeling_roberta import (
#     RobertaConfig,
#     RobertaModel,
# )

from transformers import AutoTokenizer, AutoModel

# from pytorch_transformers.tokenization_bert import BertTokenizer
# from pytorch_transformers.tokenization_roberta import RobertaTokenizer

from blink.common.ranker_base import XLMRobertaEncoder, get_model_obj
from blink.common.optimizer import get_xlm_roberta_optimizer
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG


def load_crossencoder(params):
    # Init model
    crossencoder = CrossEncoderRanker(params)
    return crossencoder


class CrossEncoderModule(torch.nn.Module):
    def __init__(self, params, tokenizer):
        super(CrossEncoderModule, self).__init__()
        model_path = params["bert_model"]

        encoder_model = AutoModel.from_pretrained(model_path)
        encoder_model.resize_token_embeddings(len(tokenizer))
        self.encoder = XLMRobertaEncoder(
            encoder_model,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = self.encoder.xlm_roberta_model.config

        hidden_size = self.encoder.xlm_roberta_model.config.hidden_size

        # +1 for pageweight
        self.scorer = torch.nn.Linear(hidden_size + 1, 1)

    # def scorer(self, embedding_ctxt):
    #     model_output_dim = self.encoder.embeddings.word_embeddings.weight.size(1)
    #     self.additional_linear = nn.Linear(model_output_dim + 1, self.output_dim)
    #     self.dropout = nn.Dropout(0.1)
    #     results = self.additional_linear(self.dropout(embedding_ctxt))
    #     return results

    def forward(
        self, token_idx_ctxt, mask_ctxt, pageweight=None
    ):
        embedding_ctxt = self.encoder(token_idx_ctxt, mask_ctxt)
        if pageweight != None:
            embedding_ctxt = torch.cat([embedding_ctxt, pageweight], dim=1)

            score = self.scorer(embedding_ctxt) 
            return score.squeeze(-1)
        else:
            return embedding_ctxt.squeeze(-1)


class CrossEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(CrossEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()

        if params['tokenizer_path']:
            self.tokenizer = AutoTokenizer.from_pretrained(
                params["tokenizer_path"], do_lower_case=params["lowercase"], use_fast=False
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                params["bert_model"], do_lower_case=params["lowercase"], use_fast=False
            )

        # if params.get("roberta"):
        #     self.tokenizer = RobertaTokenizer.from_pretrained(params["bert_model"], do_lower_case=params["lowercase"])
        # else:
        #     self.tokenizer = BertTokenizer.from_pretrained(
        #         params["bert_model"], do_lower_case=params["lowercase"]
            # )

        special_tokens_dict = {
            "additional_special_tokens": [
                ENT_START_TAG,
                ENT_END_TAG,
                ENT_TITLE_TAG,
            ],
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        # special_tokens_dict = {
        #     "additional_special_tokens": [
        #         '[TEST]',
        #         '[TEST2]',
        #         '[TEST3]'
        #     ],
        # }
        # self.tokenizer.add_special_tokens(special_tokens_dict)
        self.NULL_IDX = self.tokenizer.pad_token_id
        self.START_TOKEN = self.tokenizer.cls_token
        self.END_TOKEN = self.tokenizer.sep_token
        
        # init model
        self.build_model()
        if params["path_to_model"] is not None:
            self.load_model(params["path_to_model"])

        self.model = self.model.to(self.device)

        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.model.load_state_dict(state_dict)

    def save(self, output_dir):
        self.save_model(output_dir)
        self.tokenizer.save_vocabulary(output_dir)

    def build_model(self):
        self.model = CrossEncoderModule(self.params, self.tokenizer)
    
    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = get_model_obj(self.model) 
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def get_optimizer(self, optim_states=None, saved_optim_type=None):
        return get_xlm_roberta_optimizer(
            [self.model],
            self.params["type_optimization"],
            self.params["learning_rate"],
            fp16=self.params.get("fp16"),
        )

    # def score_candidate(self, text_vecs, context_len):
    #     # Encode contexts first
    #     num_cand = text_vecs.size(1)
    #     print(f'num cand be {num_cand}')
    #     text_vecs = text_vecs.view(-1, text_vecs.size(-1))
    #     token_idx_ctxt, mask_ctxt = to_bert_input(
    #         text_vecs, self.NULL_IDX
    #     )
    #     embedding_ctxt = self.model(token_idx_ctxt, mask_ctxt)
    #     print(f'embedding context shape = {embedding_ctxt.shape}')
    #     print(f'embedding context shape after reshape = {embedding_ctxt.reshape(-1, num_cand).shape}')
    #     return embedding_ctxt.reshape(-1, num_cand)


    
    def score_candidate(self, text_vecs, context_len, pageweight=None):
        # Encode contexts first
        num_cand = text_vecs.size(1)
        # print(num_cand)

        text_vecs = text_vecs.view(-1, text_vecs.size(-1))
        token_idx_ctxt, mask_ctxt = to_roberta_input(
                self.tokenizer, text_vecs, context_len,
            )
        
        # flatten the pageweight in a similar way
        # print(pageweight)
        if pageweight != None:
            pageweight = pageweight.view(-1, 1)
            
            # these should be already pooled
            embedding_ctxt = self.model(token_idx_ctxt, mask_ctxt, pageweight)
        else:
            embedding_ctxt = self.model(token_idx_ctxt, mask_ctxt)


        return embedding_ctxt.reshape(-1, num_cand)

    
    def forward(self, input_idx, label_input, context_len, pageweight=None):
        scores = self.score_candidate(input_idx, context_len, pageweight)
        loss = F.cross_entropy(scores, label_input, reduction="mean")
        return loss, scores


def to_roberta_input(tokenizer, text_vecs, context_len):
    # Token indices remain the same
    token_idx_ctxt = text_vecs

    # Generate attention mask (1 for actual tokens, 0 for padding)
    mask_ctxt = (text_vecs != tokenizer.convert_tokens_to_ids(tokenizer.pad_token)).long()
    return token_idx_ctxt, mask_ctxt

# def to_bert_input(token_idx, null_idx, segment_pos):
#     """ token_idx is a 2D tensor int.
#         return token_idx, segment_idx and mask
#     """
#     segment_idx = token_idx * 0
#     if segment_pos > 0:
#         segment_idx[:, segment_pos:] = token_idx[:, segment_pos:] > 0

#     mask = token_idx != null_idx
#     # nullify elements in case self.NULL_IDX was not 0
#     # token_idx = token_idx * mask.long()
#     return token_idx, segment_idx, mask

# def to_bert_input(token_idx, null_idx):
#     """ token_idx is a 2D tensor int.
#         return token_idx, segment_idx and mask
#     """
#     # segment_idx = token_idx * 0
#     mask = token_idx != null_idx
#     # nullify elements in case self.NULL_IDX was not 0
#     token_idx = token_idx * mask.long()
#     return token_idx, mask
