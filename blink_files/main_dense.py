# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json

from tqdm import tqdm
import logging
import torch
import numpy as np
from colorama import init
# from termcolor import colored

# import blink.ner as NER
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
from blink.crossencoder.crossencoder import CrossEncoderRanker, load_crossencoder
from blink.biencoder.data_process import (
    process_mention_data,
    get_candidate_representation,
)
import blink.candidate_ranking.utils as utils
from blink.crossencoder.train_cross import modify, evaluate
from blink.crossencoder.data_process import prepare_crossencoder_data
from blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer
import pandas as pd
from torch.nn import Softmax
import os

HIGHLIGHTS = [
    "on_red",
    "on_green",
    "on_yellow",
    "on_blue",
    "on_magenta",
    "on_cyan",
]

BLACKLISTED_NAMES = {
    "Theodoros", "Dionysios", "Demetrios", "Apollonios",
    "Maximus", "Petrus", "Archias", "Alexandros",
    "Apollodoros", "Valerius", "Marcianus", "Timotheos",
    "Ariston", "Philippos", "Uliades", "Herakleides",
    "Philon", "Palladios", "Diodoros", "Theodosios",
    "Ptolemaios", "Theophilos", "Diokles", "Theodotos",
    "Antiochos", "Victor", "Ulpius", "Olympios",
    "Leontius", "Marcellus", "Diogenes", "Asklepiades",
    "Kallistratos", "Archelaos", "Marcellinus", "Seleukos",
    "Valentinus", "Iulianos", "Antipatros", "Aristodemos",
    "Agathokles", "Maximinus", "Glaukos", "Artemidoros",
    "Philippus", "Menekrates", "Proculus", "Thomas",
    "Priscus", "Apollonides", "Nikostratos", "Marcus",
    "Severus", "Metrodoros", "Valerianus", "Nikias",
    "Lykos", "Theon", "Marinus", "Sallustius",
    "Eusebios", "Dorotheos", "Nikanor", "Kallias",
    "Tryphon", "Bassus", "Mithridates", "Pamphilos",
    "Aristarchos", "Antimachos", "Ioannes", "Amyntas",
    "Laodike"
}


def _print_colorful_text(input_sentence, samples):
    init()  # colorful output
    msg = ""
    if samples and (len(samples) > 0):
        msg += input_sentence[0 : int(samples[0]["start_pos"])]
        for idx, sample in enumerate(samples):
            msg += colored(
                input_sentence[int(sample["start_pos"]) : int(sample["end_pos"])],
                "grey",
                HIGHLIGHTS[idx % len(HIGHLIGHTS)],
            )
            if idx < len(samples) - 1:
                msg += input_sentence[
                    int(sample["end_pos"]) : int(samples[idx + 1]["start_pos"])
                ]
            else:
                msg += input_sentence[int(sample["end_pos"]) :]
    else:
        msg = input_sentence
        print("Failed to identify entity from text:")
    print("\n" + str(msg) + "\n")


def _print_colorful_prediction(
    idx, sample, e_id, e_title, e_text, e_url, show_url=False
):
    print(colored(sample["mention"], "grey", HIGHLIGHTS[idx % len(HIGHLIGHTS)]))
    to_print = "id:{}\ntitle:{}\ntext:{}\n".format(e_id, e_title, e_text[:256])
    if show_url:
        to_print += "url:{}\n".format(e_url)
    print(to_print)


def _annotate(ner_model, input_sentences):
    ner_output_data = ner_model.predict(input_sentences)
    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]
    samples = []
    for mention in mentions:
        record = {}
        record["label"] = "unknown"
        record["label_id"] = -1
        # LOWERCASE EVERYTHING !
        record["context_left"] = sentences[mention["sent_idx"]][
            : mention["start_pos"]
        ].lower()
        record["context_right"] = sentences[mention["sent_idx"]][
            mention["end_pos"] :
        ].lower()
        record["mention"] = mention["text"].lower()
        record["start_pos"] = int(mention["start_pos"])
        record["end_pos"] = int(mention["end_pos"])
        record["sent_idx"] = mention["sent_idx"]
        samples.append(record)
    return samples


def _load_candidates(
    entity_catalogue, entity_encoding, faiss_index=None, index_path=None, logger=None
):
    # only load candidate encoding if not using faiss index
    if faiss_index is None:
        candidate_encoding = torch.load(entity_encoding)
        if os.path.exists(entity_encoding.replace('encoding', 'pageweights')):
            pageweights = torch.load(entity_encoding.replace('encoding', 'pageweights'))
        else:
            pageweights = None
        indexer = None
    else:
        if logger:
            logger.info("Using faiss index to retrieve entities.")
        candidate_encoding = None
        assert index_path is not None, "Error! Empty indexer path."
        if faiss_index == "flat":
            indexer = DenseFlatIndexer(1)
        elif faiss_index == "hnsw":
            indexer = DenseHNSWFlatIndexer(1)
        else:
            raise ValueError("Error! Unsupported indexer type! Choose from flat,hnsw.")
        indexer.deserialize_from(index_path)

    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    id2metadata = {}

    local_idx = 0
    with open(entity_catalogue, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]
            if 'metadata' in list(entity.keys()):
                metadata = entity["metadata"]

                ##TODO remove spaghetti
                import math

                val = metadata.get('begin_date')

                def is_missing(x):
                    if x is None or x == '':
                        return True
                    # Check numeric NaN
                    if isinstance(x, float) and math.isnan(x):
                        return True
                    # Sometimes NaN leaks in as a string like "nan" or "NaN"
                    if isinstance(x, str) and x.strip().lower() in ('nan', 'nat', 'none'):
                        return True
                    return False

                if is_missing(val):
                    metadata['begin_date'] = None
                else:
                    try:
                        metadata['begin_date'] = int(float(val))
                    except (ValueError, TypeError):
                        metadata['begin_date'] = None


                id2metadata[local_idx] = metadata

            local_idx += 1
    
    return (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        indexer,
        id2metadata,
        pageweights
    )


def __map_test_entities(test_entities_path, title2id, logger):
    # load the 732859 tac_kbp_ref_know_base entities
    kb2id = {}
    missing_pages = 0
    n = 0
    with open(test_entities_path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)
            if entity["title"] not in title2id:
                missing_pages += 1
            else:
                kb2id[entity["entity_id"]] = title2id[entity["title"]]
            n += 1
    if logger:
        logger.info("missing {}/{} pages".format(missing_pages, n))
    return kb2id


def __load_test(test_filename, kb2id, wikipedia_id2local_id, logger):
    test_samples = []
    with open(test_filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            record = json.loads(line)
            # print(record)
            record["label"] = str(record["label_id"])
            # print(record)
            # # for tac kbp we should use a separate knowledge source to get the entity id (label_id)
            # if kb2id and len(kb2id) > 0:
            #     if record["label"] in kb2id:
            #         record["label_id"] = kb2id[record["label"]]
            #     else:
            #         continue

            # # check that each entity id (label_id) is in the entity collection
            # elif wikipedia_id2local_id and len(wikipedia_id2local_id) > 0:
            #     try:
            #         key = int(record["label"].strip())
            #         if key in wikipedia_id2local_id:
            #             record["label_id"] = wikipedia_id2local_id[key]
            #         else:
            #             continue
            #     except:
            #         continue

            # LOWERCASE EVERYTHING !
            record["context_left"] = record["context_left"].lower()
            record["context_right"] = record["context_right"].lower()
            record["mention"] = record["mention"].lower()
            test_samples.append(record)
    if logger:
        logger.info("{}/{} samples considered".format(len(test_samples), len(lines)))
    return test_samples


def _get_test_samples(
    test_filename, test_entities_path, title2id, wikipedia_id2local_id, logger
):
    kb2id = None
    if test_entities_path:
        kb2id = __map_test_entities(test_entities_path, title2id, logger)
        # logging.info(f'first items from the kb = {[(key, value) for key, value in kb2id.items()][:5]}')
    logging.info(f'type kb2id {type(kb2id)}')
    test_samples = __load_test(test_filename, kb2id, wikipedia_id2local_id, logger)
    return test_samples


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    data, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        add_metadata=False,
        silent=True,
        logger=None,
        debug=biencoder_params["debug"],
    )

    class CombinedDataset(torch.utils.data.Dataset):
        def __init__(self, d, td):
            self.d = d
            self.td = td

        def __len__(self): return len(self.td)

        def __getitem__(self, i): return {k: self.d[k][i] for k in self.d.keys()}, self.td[i]

    ds = CombinedDataset(data, tensor_data)

    sampler = SequentialSampler(ds)
    dataloader = DataLoader(
        ds, sampler=sampler, batch_size=biencoder_params["eval_batch_size"]
    )
    return dataloader

import random


def _run_biencoder(biencoder, dataloader, candidate_encoding, id2metadata=None, top_k=100, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    for batch in tqdm(dataloader):
        data, (context_input, _, label_ids) = batch

        if "date" in data.keys():
            meta = [{"date": data["date"][sample_idx].item(), 
                    "mention": data["mention"][sample_idx]}
                for sample_idx in range(label_ids.shape[0])]
        else:
            meta = None


        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            

            elif meta:
                all_bien_scores, levenstein_scores = biencoder.score_candidate(
                    context_input,None, add_metadata=False, text_meta=meta, cand_encs=candidate_encoding, cand_meta=id2metadata# .to(device)
                )
                bien_scores, bien_indicies = all_bien_scores.topk(top_k)
                # comment out to ignore levenstein
                bien_scores = bien_scores.data.cpu().numpy()
                bien_indicies = bien_indicies.data.cpu().numpy()


                ##
                # Top 64
                ##

                bien_scores   = [row.copy() for row in bien_scores]
                bien_indicies  = [row.copy() for row in bien_indicies]

                
                for sentence_idx in range(len(bien_scores)):
                    #sort and retrieve the highest score and the corresponding name_complete
                    top1_lev_name = sorted(levenstein_scores[sentence_idx], key=lambda x:x[2])[0]
                    #if the name_complete isn't blacklisted
                    if top1_lev_name[0] not in BLACKLISTED_NAMES:
                        for item in levenstein_scores[sentence_idx]:
                            lev_ind, lev_name_and_score = item
                            # add ALL entities where the name_complete AND score == the top one
                            # we manually altered the scores of temporally implausible entities
                            # my life is a joke
                            if lev_name_and_score == top1_lev_name and lev_ind not in bien_indicies[sentence_idx]:
                                bien_indicies[sentence_idx] = np.append(
                                    bien_indicies[sentence_idx], 
                                    lev_ind
                                )

                                bien_scores[sentence_idx] = np.append(
                                    bien_scores[sentence_idx],
                                    all_bien_scores[sentence_idx, lev_ind].data.cpu().numpy()
                                )
            else:
                all_bien_scores = biencoder.score_candidate(
                    context_input,None, add_metadata=False, text_meta=meta, cand_encs=candidate_encoding, cand_meta=id2metadata# .to(device)
                )
                bien_scores, bien_indicies = all_bien_scores.topk(top_k)
                # comment out to ignore levenstein
                bien_scores = bien_scores.data.cpu().numpy()
                bien_indicies = bien_indicies.data.cpu().numpy()



        labels.extend(label_ids.data.numpy())
        nns.extend(bien_indicies)
        all_scores.extend(bien_scores)
    biencoder_found = []
    for label, nn in zip(labels, nns):
        if label in nn:
            biencoder_found.append(True)
        else:
            biencoder_found.append(False)
    # print(labels)
    # print(nns)
    # print(all_scores)
    return labels, nns, all_scores, biencoder_found


def _process_crossencoder_dataloader(context_input, label_input, crossencoder_params, pageweight=None):
    if pageweight != None:
        tensor_data = TensorDataset(context_input, label_input, pageweight)
    else:
        tensor_data = TensorDataset(context_input, label_input)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=crossencoder_params["eval_batch_size"]
    )
    return dataloader


def _run_crossencoder(crossencoder, dataloader, logger, context_len, device="cuda"):
    crossencoder.model.eval()
    accuracy = 0.0
    crossencoder.to(device)

    res = evaluate(crossencoder, dataloader, device, logger, context_len, zeshel=False, silent=False)
    accuracy = res["normalized_accuracy"]
    logits = res["logits"]

    if accuracy > -1:
        predictions = np.argsort(logits, axis=1)
    else:
        predictions = []

    return accuracy, predictions, logits

import ast

def load_models(args, logger=None):

    # load biencoder model
    if logger:
        logger.info("loading biencoder model")
    with open(args.biencoder_config) as json_file:
        text = json_file.read()
        # text = text.replace('\'', '"')
        biencoder_params = ast.literal_eval(text)
        # print(text)
        # biencoder_params = json.loads(text)
        biencoder_params["path_to_model"] = args.biencoder_model
    biencoder = load_biencoder(biencoder_params)

    crossencoder = None
    crossencoder_params = None
    if not args.fast:
        # load crossencoder model
        if logger:
            logger.info("loading crossencoder model")
        with open(args.crossencoder_config) as json_file:
            text = json_file.read()
            # text = text.replace('\'', '"')
            crossencoder_params = ast.literal_eval(text)
            # crossencoder_params = json.load(json_file)
            crossencoder_params["path_to_model"] = args.crossencoder_model
        crossencoder = load_crossencoder(crossencoder_params)

    # load candidate entities
    if logger:
        logger.info("loading candidate entities")
    (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
        id2metadata,
        pageweights
    ) = _load_candidates(
        args.entity_catalogue, 
        args.entity_encoding, 
        faiss_index=getattr(args, 'faiss_index', None), 
        index_path=getattr(args, 'index_path' , None),
        logger=logger,
    )

    # print(f'first entity in the encoding {len(candidate_encoding[0])}')
    # print(f'in human language {id2title[0]}, {id2text[0]}')

    # print(f'second entity in the encoding {len(candidate_encoding[1])}')
    # print(f'in human language {id2title[1]}, {id2text[1]}')

    # print(f'in human language {id2title[0]}, {id2text[0]}')


    return (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        id2metadata,
        pageweights,
        wikipedia_id2local_id,
        faiss_indexer
    )


def run(
    args,
    logger,
    biencoder,
    biencoder_params,
    crossencoder,
    crossencoder_params,
    candidate_encoding,
    title2id,
    id2title,
    id2text,
    id2metadata,
    pageweights,
    wikipedia_id2local_id,
    faiss_indexer=None,
    test_data=None,
):

    if not test_data and not args.test_mentions and not args.interactive:
        msg = (
            "ERROR: either you start BLINK with the "
            "interactive option (-i) or you pass in input test mentions (--test_mentions)"
            "and test entities (--test_entities)"
        )
        raise ValueError(msg)

    id2url = {
        v: "https://en.wikipedia.org/wiki?curid=%s" % k
        for k, v in wikipedia_id2local_id.items()
    }

    stopping_condition = False
    while not stopping_condition:

        samples = None

        if args.interactive:
            logger.info("interactive mode")

            # # biencoder_params["eval_batch_size"] = 1

            # # Load NER model
            # ner_model = NER.get_model()

            # # Interactive
            # text = input("insert text:")

            # # Identify mentions
            # samples = _annotate(ner_model, [text])

            # _print_colorful_text(text, samples)

        else:
            if logger:
                logger.info("test dataset mode")

            if test_data:
                samples = test_data
            else:
                logger.info('loading mentions')
                # Load test mentions
                samples = _get_test_samples(
                    args.test_mentions,
                    args.test_entities,
                    title2id,
                    wikipedia_id2local_id,
                    logger,
                )
            samples = samples
            # print(samples[0])
            stopping_condition = True

        # don't look at labels
        # keep_all = (
        #     args.interactive
        #     or samples[0]["label"] == "unknown"
        #     or samples[0]["label_id"] < 0
        # )

        keep_all = args.keep_all

        # prepare the data for biencoder
        if logger:
            logger.info("preparing data for biencoder")
        dataloader = _process_biencoder_dataloader(
            samples, biencoder.tokenizer, biencoder_params
        )

        # run biencoder
        if logger:
            logger.info(f"run biencoder on {len(samples)} samples")
        top_k = args.top_k
        #what is nns?
        labels, nns, scores, biencoder_found = _run_biencoder(
            biencoder, dataloader, candidate_encoding, id2metadata, top_k, faiss_indexer
        )
        if logger:
            logger.info(f"received {len(labels)}")
        # print(labels, nns, scores)

        ## UNTILL HERE (BIENCODER) EVERYTHING WORKS

        if args.interactive:

            print("\nfast (biencoder) predictions:")

            _print_colorful_text(text, samples)

            # print biencoder prediction
            idx = 0
            for entity_list, sample in zip(nns, samples):
                e_id = entity_list[0]
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                e_url = id2url[e_id]
                _print_colorful_prediction(
                    idx, sample, e_id, e_title, e_text, e_url, args.show_url
                )
                idx += 1
            print()

            if args.fast:
                # use only biencoder
                continue

        else:
            print('How much of the time in biencoder?')
            print(f'{sum(map(int,biencoder_found))}/{len(labels)}')

            biencoder_accuracy = -1
            recall_at = -1
            if not keep_all:
                # get recall values
                top_k = args.top_k
                x = []
                y = []
                for i in range(1, top_k + 1):
                    temp_y = 0.0
                    for label, top in zip(labels, nns):
                        if label in top[:i]:
                            temp_y += 1
                    if len(labels) > 0:
                        temp_y /= len(labels)
                    x.append(i)
                    y.append(temp_y)
                # plt.plot(x, y)
                biencoder_accuracy = y[0]
                recall_at = y[-1]
                print("biencoder accuracy: %.4f" % biencoder_accuracy)
                print("biencoder recall@%d: %.4f" % (top_k, y[-1]))

            # if args.fast:

            #     predictions = []
            #     for entity_list in nns:
            #         sample_prediction = []
            #         for e_id in entity_list:
            #             e_title = id2title[e_id]
            #             sample_prediction.append(e_title)
            #         predictions.append(sample_prediction)

            #     # use only biencoder
            #     return (
            #         biencoder_accuracy,
            #         recall_at,
            #         -1,
            #         -1,
            #         len(samples),
            #         predictions,
            #         scores,
            #     )

        #prepare crossencoder data
        if logger:
            logger.info("prep_crossencoder")
        if pageweights != None:
            context_input, candidate_input, label_input, pageweight_input = prepare_crossencoder_data(
                crossencoder.tokenizer, samples, labels, nns, id2title, id2text, pageweights, keep_all, biencoder_params["max_context_length"]
            )
        else:
            print('no pageweight indeedeth.')
            context_input, candidate_input, label_input = prepare_crossencoder_data(
                crossencoder.tokenizer, samples, labels, nns, id2title, id2text, pageweights, keep_all, biencoder_params["max_context_length"]
            )
            pageweight_input = None
        # logging.info(f"""first sample = {samples[0]}""")
        # logging.info(f"""label_input = {label_input}""")
        # logging.info(f"""as received by the crossencoder data\n lenght of the context_vecs:{len(context_input[0])}
        #             len of the correct candidate_vec: {len(candidate_input[0][label_input[0]])}
        #             first candidate_vec: {len(candidate_input[0][0])}
        #             labels: {label_input[0]}""")
        # max_n = 200
        # context_input = context_input[:max_n]
        # candidate_input = candidate_input[:max_n]
        # label_input = label_input[:max_n]

        context_input_modified = modify(
            context_input, candidate_input, crossencoder_params["max_seq_length"]
        )

        # logging.info(f"first sample thing passed to evaluate function: {context_input_modified[0]}, {label_input[0]} \n len {context_input_modified[0][61]}")


        dataloader = _process_crossencoder_dataloader(
            context_input_modified, label_input, crossencoder_params, pageweight_input
        )

        # run crossencoder and get accuracy
        accuracy, index_array, unsorted_scores = _run_crossencoder(
            crossencoder,
            dataloader,
            logger,
            context_len=biencoder_params["max_context_length"],
        )

        if args.interactive:

            print("\naccurate (crossencoder) predictions:")

            _print_colorful_text(text, samples)

            # print crossencoder prediction
            idx = 0
            for entity_list, index_list, sample in zip(nns, index_array, samples):
                e_id = entity_list[index_list[-1]]
                e_title = id2title[e_id]
                e_text = id2text[e_id]
                e_url = id2url[e_id]
                _print_colorful_prediction(
                    idx, sample, e_id, e_title, e_text, e_url, args.show_url
                )
                idx += 1
            print()
        else:

            scores = []
            predictions = []
            prediction_ids = []
            for entity_list, index_list, scores_list in zip(
                nns, index_array, unsorted_scores
            ):

                index_list = index_list.tolist()

                # descending order
                index_list.reverse()

                sample_prediction = []
                sample_prediction_id = []
                sample_scores = []
                for index in index_list:
                    e_id = entity_list[index]
                    # print(e_id)
                    e_title = id2title[e_id]
                    # print(e_title)
                    sample_prediction.append(e_title)
                    sample_prediction_id.append(e_id)
                    sample_scores.append(scores_list[index])
                predictions.append(sample_prediction)
                scores.append(sample_scores)
                prediction_ids.append(sample_prediction_id)

            crossencoder_normalized_accuracy = -1
            overall_unormalized_accuracy = -1
            if not keep_all:
                crossencoder_normalized_accuracy = accuracy
                print(
                    "crossencoder normalized accuracy: %.4f"
                    % crossencoder_normalized_accuracy
                )

                if len(samples) > 0:
                    overall_unormalized_accuracy = (
                        crossencoder_normalized_accuracy * len(label_input) / len(samples)
                    )
                print(
                    "overall unnormalized accuracy: %.4f" % overall_unormalized_accuracy
                )
            return (
                biencoder_accuracy,
                recall_at,
                crossencoder_normalized_accuracy,
                overall_unormalized_accuracy,
                samples,
                predictions,
                scores,
                biencoder_found,
                prediction_ids
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode."
    )

    # test_data
    parser.add_argument(
        "--test_mentions", dest="test_mentions", type=str, help="Test Dataset."
    )
    parser.add_argument(
        "--test_entities", dest="test_entities", type=str, help="Test Entities."
    )

    # biencoder
    parser.add_argument(
        "--biencoder_model",
        dest="biencoder_model",
        type=str,
        default="models/biencoder_wiki_large.bin",
        help="Path to the biencoder model.",
    )
    parser.add_argument(
        "--biencoder_config",
        dest="biencoder_config",
        type=str,
        default="models/biencoder_wiki_large.json",
        help="Path to the biencoder configuration.",
    )
    parser.add_argument(
        "--entity_catalogue",
        dest="entity_catalogue",
        type=str,
        # default="models/tac_entity.jsonl",  # TAC-KBP
        default="models/entity.jsonl",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )
    parser.add_argument(
        "--entity_encoding",
        dest="entity_encoding",
        type=str,
        # default="models/tac_candidate_encode_large.t7",  # TAC-KBP
        default="models/all_entities_large.t7",  # ALL WIKIPEDIA!
        help="Path to the entity catalogue.",
    )

    # crossencoder
    parser.add_argument(
        "--crossencoder_model",
        dest="crossencoder_model",
        type=str,
        default="models/crossencoder_wiki_large.bin",
        help="Path to the crossencoder model.",
    )
    parser.add_argument(
        "--crossencoder_config",
        dest="crossencoder_config",
        type=str,
        default="models/crossencoder_wiki_large.json",
        help="Path to the crossencoder configuration.",
    )

    parser.add_argument(
        "--top_k",
        dest="top_k",
        type=int,
        default=10,
        help="Number of candidates retrieved by biencoder.",
    )

    # output folder
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        default="output",
        help="Path to the output.",
    )

    parser.add_argument(
        "--fast", dest="fast", action="store_true", help="only biencoder mode"
    )

    parser.add_argument(
        "--show_url",
        dest="show_url",
        action="store_true",
        help="whether to show entity url in interactive mode",
    )

    parser.add_argument(
        "--faiss_index", type=str, default=None, help="whether to use faiss index",
    )

    parser.add_argument(
        "--index_path", type=str, default=None, help="path to load indexer",
    )
        
    parser.add_argument(
        "--keep_all", type=bool, default=False, help="path to load indexer",
    )

    args = parser.parse_args()

    logger = utils.get_logger(args.output_path)

    models = load_models(args, logger)
    biencoder_accuracy, recall_at, crossencoder_normalized_accuracy, overall_unormalized_accuracy, samples, predictions, scores, biencoder_found, prediction_ids = run(args, logger, *models)
    
    mode = "_".join(args.test_mentions.split('/')[-3:]).replace('.jsonl', '')
    logging.info(f'mentions in mode {mode}')

    if args.keep_all == False:
        outfile = args.output_path + f'/{mode}_scores.txt'
        with open(args.output_path + f'/{mode}_scores.txt', 'w', encoding='UTF-8') as f:
            f.write('evaluation_results of this model\n\n')
            f.write(f'biencoder_accuracy {biencoder_accuracy}\n',)
            f.write(f'recall_at, {recall_at}\n')
            f.write(f'crossencoder_normalized_accuracy, {crossencoder_normalized_accuracy}\n')
            f.write(f'overall_normalized_accuracy, {overall_unormalized_accuracy}\n')
        print(f'recall and accuracy saved to {outfile}')
    else:
        final_results = []
        for sample, prediction, score, biencoder_foud, prediction_id in zip(samples, predictions, scores, biencoder_found, prediction_ids):
            print(sample)
            m = Softmax(dim=0)
            output = m(torch.Tensor(score))
            sample['prediction'] = prediction[0]
            sample['prediction_id'] = prediction_id[0]
            sample['biencoder_found'] = biencoder_foud
            sample['prediction_score'] = output[0].float()
            sample['top_5'] = prediction[:5]
            sample['top_5_scores'] = output[:5]
            final_results.append(sample)
        df = pd.DataFrame(final_results)
        outfile = args.output_path + f'/{mode}_predictions.csv'
        df.to_csv(outfile)
        print(print(f'predictions and scores saved to {outfile}'))
        



