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
from tqdm import tqdm
import Levenshtein

from transformers import (
    # BertPreTrainedModel,
    # BertConfig,
    AutoModel, AutoTokenizer, XLMRobertaTokenizer
    #
)

# from transformers import XLMRobertatokenizer

from blink.common.ranker_base import XLMRobertaEncoder, get_model_obj
from blink.common.optimizer import get_xlm_roberta_optimizer

from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG


def load_biencoder(params):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params, tokenizer):
        super(BiEncoderModule, self).__init__()
        if params["cand_bert"] != None and params['ctxt_bert'] != None:
            ctxt_bert = AutoModel.from_pretrained(params["ctxt_bert"])
            cand_bert = AutoModel.from_pretrained(params['cand_bert'])
        else:
            ctxt_bert = AutoModel.from_pretrained(params["bert_model"])
            cand_bert = AutoModel.from_pretrained(params['bert_model'])
        cand_bert.resize_token_embeddings(len(tokenizer))
        ctxt_bert.resize_token_embeddings(len(tokenizer))
        self.context_encoder = XLMRobertaEncoder(
            ctxt_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.cand_encoder = XLMRobertaEncoder(
            cand_bert,
            params["out_dim"],
            layer_pulled=params["pull_from_layer"],
            add_linear=params["add_linear"],
        )
        self.config = ctxt_bert.config

    def forward(
        self,
        token_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands


class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params, shared=None):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu"
        )
        self.n_gpu = torch.cuda.device_count()
        # init tokenizer
        # self.NULL_IDX = 0
        # self.START_TOKEN = "[CLS]"
        # self.END_TOKEN = "[SEP]"
        if params['tokenizer_path']:
            self.tokenizer = AutoTokenizer.from_pretrained(
                params["tokenizer_path"], do_lower_case=params["lowercase"], use_fast=False
            )
        else:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                params["bert_model"], do_lower_case=params["lowercase"], use_fast=False
            )
        special_tokens_dict = {
            "additional_special_tokens": [
                ENT_START_TAG,
                ENT_END_TAG,
                ENT_TITLE_TAG,
            ],
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)

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

    def build_model(self):
        self.model = BiEncoderModule(self.params, self.tokenizer)

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
 
    def encode_context(self, cands):
        token_idx_cands, mask_cands = to_bert_input(
            self.tokenizer, cands
        )

        embedding_context, _ = self.model(
            token_idx_cands, mask_cands, None, None
        )
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, mask_cands = to_bert_input(
            self.tokenizer, cands
        )
        _, embedding_cands = self.model(
            None, None, token_idx_cands, mask_cands
        )
        return embedding_cands.cpu().detach()
        # TODO: why do we need cpu here?
        # return embedding_cands

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def score_candidate(
        self,
        text_vecs,
        cand_vecs,
        text_meta=None,
        cand_meta=None,
        random_negs=True,
        cand_encs=None, # pre-computed candidate encoding.
        add_metadata=True,
    ):
        if add_metadata:
            print('doing the metadata thing')
            assert len(cand_encs) == len(cand_meta)
        # Encode contexts first
        token_idx_ctxt, mask_ctxt = to_bert_input(
            self.tokenizer, text_vecs
        )
        # print(token_idx_ctxt)
        # print(mask_ctxt)
        embedding_ctxt, _ = self.model(
            token_idx_ctxt.to('cuda'), mask_ctxt.to('cuda'), None, None
        )

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            scores = embedding_ctxt.mm(cand_encs.to('cuda').t())
        else:
            # Train time. We compare with all elements of the batch
            token_idx_cands, mask_cands = to_bert_input(
                self.tokenizer, cand_vecs
            )
            _, embedding_cands = self.model(
                None, None, token_idx_cands.to('cuda'), mask_cands.to('cuda')
            )
            embedding_ctxt.to('cuda')
            embedding_cands.to('cuda')
            if random_negs:
                # train on random negatives
                return embedding_ctxt.mm(embedding_cands.t())
            else:
                # train on hard negatives
                embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
                embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 2
                scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
                scores = torch.squeeze(scores)

        if add_metadata:
            pass

        return scores

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(self, context_input, cand_input, label_input=None, add_metadata=True):
        flag = label_input is None
        scores = self.score_candidate(context_input, cand_input, flag, add_metadata=add_metadata)
        bs = scores.size(0)
        if label_input is None:
            target = torch.LongTensor(torch.arange(bs))
            target = target.to(self.device)
            loss = F.cross_entropy(scores, target, reduction="mean")
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            # TODO: add parameters?
            loss = loss_fct(scores, label_input)
        return loss, scores


def to_bert_input(tokenizer, token_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    # print(f'{tokenizer.convert_to_ids(pad_token)')
    # segment_idx = token_idx * 0
    mask = token_idx != tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    # nullify elements in case self.NULL_IDX was not 0
    # token_idx = token_idx * mask.long()
    return token_idx, mask

    # return token_idx, segment_idx, mask
