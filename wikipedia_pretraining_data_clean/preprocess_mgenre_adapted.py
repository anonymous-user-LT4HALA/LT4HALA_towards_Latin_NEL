# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append('..')
sys.path.append('GENRE')
import html

import jsonlines
import numpy as np
from genre.utils import create_input
from tqdm.auto import tqdm, trange
import json

def create_input_blink(doc, max_length, label, label_id, label_title):
    if all(
        e in doc for e in ("context_left", "mention", "context_right")
    ):
        doc_Input = doc['context_left'].split() + doc['mention'].split() + doc['context_right'].split()
        if len(doc_Input) <= max_length:
            input_ = {
                'context_left': doc["context_left"],
                'mention': doc["mention"],
                'context_right': doc["context_right"],
                'wikidata_id': doc["wikidata_id"],
                'text': label,
                'label_id': label_id,
                'label_title': label_title
            }
        elif len(doc["context_left"].split(" ")) <= max_length // 2:
            input_ =  {
                'context_left': doc["context_left"],
                'mention': doc["mention"],
                'context_right': " ".join(
                    doc["context_right"].split(" ")[
                        : max_length - len(doc["context_left"].split(" "))
                    ]
                ),
                'wikidata_id': doc["wikidata_id"],
                'text': label,
                'label_id': label_id,
                'label_title': label_title
            }
        elif len(doc["context_right"].split(" ")) <= max_length // 2:
            input_ =  {
                'context_left': " ".join(
                    doc["context_left"].split(" ")[
                        len(doc["context_right"].split(" ")) - max_length :
                    ]
                ),
                'mention': doc["mention"],
                'context_right': doc['context_right'],
                'wikidata_id': doc["wikidata_id"],
                'text': label,
                'label_id': label_id,
                'label_title': label_title
            }

        else:
            input_ =  {
                'context_left': " ".join(doc["context_left"].split(" ")[-max_length // 2 :]),
                'mention': doc["mention"],
                'context_right': " ".join(doc["context_right"].split(" ")[: max_length // 2]),
                'wikidata_id': doc["wikidata_id"],
                'text': label,
                'label_id': label_id,
                'label_title': label_title
            }
            

    else:
        print('not here')
        input_ = ''

    input_ = html.unescape(input_)

    return input_

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "action",
        type=str,
        choices=[
            "titles_lang",
            "lang_titles",
            "canonical_title",
            "marginal",
        ],
        help="How to process the target.",
    )
    parser.add_argument(
        "--base_wikipedia",
        type=str,
        help="Base folder with Wikipedia data.",
    )
    parser.add_argument(
        "--base_wikidata",
        type=str,
        help="Base folder with Wikidata data.",
    )
    parser.add_argument(
        "--base_tr2016",
        type=str,
        help="Base folder with TR2016 data.",
    )
    parser.add_argument(
        "--langs",
        type=str,
        help="Pipe (|) separated list of language ID to process.",
    )
    parser.add_argument(
        "--allowed_langs",
        type=str,
        default="af|am|ar|as|az|be|bg|bm|bn|br|bs|ca|cs|cy|da|de|el|en|eo|es|et|eu|fa|ff|fi|fr|fy|ga|gd|gl|gn|gu|ha|he|hi|hr|ht|hu|hy|id|ig|is|it|ja|jv|ka|kg|kk|km|kn|ko|ku|ky|la|lg|ln|lo|lt|lv|mg|mk|ml|mn|mr|ms|my|ne|nl|no|om|or|pa|pl|ps|pt|qu|ro|ru|sa|sd|si|sk|sl|so|sq|sr|ss|su|sv|sw|ta|te|th|ti|tl|tn|tr|uk|ur|uz|vi|wo|xh|yo|zh",
        help="Pipe (|) separated list of allowed language ID to use.",
    )
    parser.add_argument(
        "--random_n",
        type=int,
        default=1,
        help="Number or random entity titles to use when the one in the source is unavailable",
    )
    parser.add_argument(
        "--abstracts",
        action="store_true",
        help="Process abstracts only.",
    )
    parser.add_argument(
        "--target_switching",
        action="store_true",
        help="Enables target switching.",
    )
    parser.add_argument(
        "--target_switching_prob",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--monolingual",
        action="store_true",
        help="Only monolingual targets.",
    )
    parser.add_argument(
        "--target_switch_x_lingual",
        type=str,
        default='la|de',
        help="Which cross lingual connections to make",
    )
    parser.add_argument(
        "--filter_tr2016",
        action="store_true",
        help="Filters out TR2016 mention-entities.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    parser.add_argument(
        '--description',
        action="store_true",
        help='add description to dataset'
    )
    parser.add_argument(
        '--create_blink_kb',
        type=str,
        help='create blink kb in the given language',
        default=None
    )

    parser.add_argument(
        '--RE_candidates',
        type=str,
        help='whether or not to have a long or a short description',
        choices=['Kurz', 'Voll'],
        default=None
    )

    parser.add_argument(
        '--output_version',
        type=str,
        help='extra identifier for the output',
        default=None
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)

    args.allowed_langs = set(args.allowed_langs.split("|"))

    assert not (args.monolingual and args.target_switching)

    wikidataID2canonical_lang_title = {}
    wikidataID2lang_title = {}
    if args.action == "canonical_title":
        filename = os.path.join(
            args.base_wikidata, "wikidataID2canonical_lang_title.pkl"
        )
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            wikidataID2canonical_lang_title = pickle.load(f)
    else:
        filename = os.path.join(
            args.base_wikidata, "wikidataID2lang_title.pkl"
        )
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            wikidataID2lang_title = pickle.load(f)

    if args.description or args.create_blink_kb:
        ## Code added by XXX to add short wikipedia description in German 
        # or other languages if few changes are made
        ## and creating a BLINK kb with the appropriate format and label_ids
        filename = os.path.join(
            args.base_wikidata, "wikidataID2label_desc_lang.pkl"
        )
        # load the wikidataID 2 label, description and language dictionary
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            wikidataID2label_desc_lang = pickle.load(f)
        
        filename = os.path.join(
                args.base_wikipedia,
                args.create_blink_kb,
                "WikidataID2{}paragraphs".format(args.create_blink_kb),
            )
        
        with open(filename, "rb") as f:
            WikidataID2langparagraphs = pickle.load(f)

        #use this as a base for the kb
        if args.create_blink_kb:
            import pandas as pd
            importance_metrics = pd.read_csv('../wikipedia_pretraining_data_clean/wikidata_importance_metrics_async.tsv', sep='\t')
            importance_metrics.page_length_bytes.fillna(0, inplace=True)
            id2page_length_bytes = dict(zip(importance_metrics.qid, importance_metrics.page_length_bytes))

            blink_kb = []
            wikidataID2label_id = {}
            for wikidataID, label_desc_langs in wikidataID2label_desc_lang.items():
                for label_desc_lang in label_desc_langs:
                    # iterate over the possible labels, description and languages associated with the wikidataID
                    label, desc, lang = label_desc_lang
                    # if the language == the target language (in this case German)
                    if lang == args.create_blink_kb:
                        # add the relevant information for blink + the wikidata_ID
                        kb_item = {'wikidata_id': wikidataID,
                                   #BLINK requires sequential int entity label ids in the kb
                                   'label_id': len(blink_kb),
                                   'text': WikidataID2langparagraphs.get(wikidataID, desc) if args.RE_candidates == 'Voll' else desc,
                                   'title': label,
                                   'metadata': {'pageweight': id2page_length_bytes.get(wikidataID, 0)}}
                        wikidataID2label_id[wikidataID] = len(blink_kb)
                        blink_kb.append(kb_item)
                        # save a mapping from the wikidataID to the BLINK id            
            #save to jsonl    
            filename = os.path.join(
                args.base_wikidata, f"blink_kb_{args.create_blink_kb}_{args.RE_candidates}.jsonl"
            )

            logging.info("creating kb {} with {} ents".format(filename, len(blink_kb)))
            with open(filename, "w", encoding='UTF-8') as f:
                for item in blink_kb:
                    f.write(json.dumps(item, ensure_ascii=False))
                    f.write('\n')
        else:
            #if the blink kb already exists, load it.
            filename = os.path.join(
                args.base_wikidata, f"blink_kb_de_{args.RE_candidates}.jsonl"
            )
            logging.info("reading kb {}".format(filename))
            blink_kb = []
            with open(filename, "r", encoding='UTF-8') as f:
                for line in f.readlines():
                    blink_dct = json.loads(line)
                    blink_kb.append(blink_dct)
            #recreate the mapping from wikidataID to label_id
            wikidataID2label_id = {item['wikidata_id']: item['label_id'] for item in blink_kb}
            


    # tr2016_data = []
    # for fname in os.listdir(args.base_tr2016):
    #     if "test" in fname:
    #         with jsonlines.open(os.path.join(args.base_tr2016, fname)) as f:
    #             data += list(f)

    # tr2016_mentions = {
    #     (d["meta"]["mention"], wikidataID)
    #     for d in data
    #     for wikidataID in d["output"][0]["answer"]
    # }

    for lang in args.langs.split("|"):
        filename = os.path.join(args.base_wikipedia, "{0}/{0}wiki.pkl".format(lang))
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            wiki = pickle.load(f)

        flag = False
        for page in tqdm(wiki.values(), desc=lang):
            for a in page["anchors"]:
                if (
                    page["paragraphs"][a["paragraph_id"]][a["start"] : a["end"]]
                    != a["text"]
                ):
                    a["paragraph_id"] -= 1
                    flag = True
                assert (
                    page["paragraphs"][a["paragraph_id"]][a["start"] : a["end"]]
                    == a["text"]
                )

        if flag:
            filename = os.path.join(args.base_wikipedia, "{0}/{0}wiki.pkl".format(lang))
            logging.info("Saving {}".format(filename))
            with open(filename, "wb") as f:
                pickle.dump(wiki, f)

        fs_name = os.path.join(args.base_wikipedia, "{}/{}{}{}{}.source").format(
            lang,
            args.action,
            "_abstract" if args.abstracts else "",
            "_target_switching" if args.target_switching else "",
            "_monolingual" if args.monolingual else "",
            "_description" if args.description else ""
        )

        ft_name = os.path.join(args.base_wikipedia, "{}/{}{}{}{}.target").format(
            lang,
            args.action,
            "_abstract" if args.abstracts else "",
            "_target_switching" if args.target_switching else "",
            "_monolingual" if args.monolingual else "",
            "_description" if args.description else ""
        )

        # bl_name = os.path.join(args.base_wikipedia, "{}/{}{}{}{}_blink.jsonl").format(
        #     lang,
        #     args.action,
        #     "_abstract" if args.abstracts else "",
        #     "_target_switching" if args.target_switching else "",
        #     "_monolingual" if args.monolingual else "",
        #     "_description" if args.description else "",
        # )

        bl_name = os.path.join(args.base_wikipedia, "{}/{}_{}_blink.jsonl").format(
            lang,
            args.action,
            args.RE_candidates,
            args.output_version
        )

        logging.info("Creating {}".format(fs_name))
        logging.info("Creating {}".format(ft_name))
        logging.info("Creating {}".format(bl_name))

        with open(fs_name, "w") as fs, open(ft_name, "w") as ft, open(bl_name, "w") as bl:
            
            ent_count = 0


            for page in tqdm(wiki.values()):

                max_paragraph_id = 0 if args.abstracts else len(page["paragraphs"])
                while (
                    max_paragraph_id < len(page["paragraphs"])
                    and "Section::::" not in page["paragraphs"][max_paragraph_id]
                ):
                    max_paragraph_id += 1

                for anchor in page["anchors"]:
                    # print(anchor['wikidata_ids'])
                    if (
                        len(anchor["wikidata_ids"]) == 1 and
                        #added this condition bc smaller knowledge base = more mistakes 
                        #when solving anchors in the "wikidata" way
                        #(which doesn't take the lang into account while matching anchors)   
                        anchor['wikidata_src'] != 'wikidata'
                        and anchor["paragraph_id"] < max_paragraph_id
                        and (
                            list(anchor["wikidata_ids"])[0] in wikidataID2lang_title
                            or list(anchor["wikidata_ids"])[0]
                            in wikidataID2canonical_lang_title
                        )
                    ):
                        #create left_context, mention, right_context
                        left_context = page["paragraphs"][anchor["paragraph_id"]][
                            : anchor["start"]
                        ].strip()
                        mention = page["paragraphs"][anchor["paragraph_id"]][
                            anchor["start"] : anchor["end"]
                        ].strip()
                        right_context = page["paragraphs"][anchor["paragraph_id"]][
                            anchor["end"] :
                        ].strip()

                        if mention == "":
                            continue
                        if (
                            args.filter_tr2016
                            and (mention, list(anchor["wikidata_ids"])[0]) in tr2016_mentions
                        ):
                            continue

                        input_ = (
                            create_input(
                                {
                                    "input": "{} [START] {} [END] {}".format(
                                        left_context, mention, right_context
                                    ).strip(),
                                    "meta": {
                                        "left_context": left_context,
                                        "mention": mention,
                                        "right_context": right_context,
                                    },
                                },
                                128,
                            )
                            .replace("\n", ">>")
                            .replace("\r", ">>")
                        )

                        if args.action == "titles_lang" or args.action == "lang_titles":
                            #create a temporary dict for the language, title mapping of the first wikidataID
                            #in this specific anchor (anchor = link to a different page, sometimes associated with wikidata entries/solved, sometimes not)
                            tmp_dict = dict(
                                wikidataID2lang_title[list(anchor["wikidata_ids"])[0]]
                            )
                            try:
                                #get the label_id for the tgt lang kb
                                label_id = wikidataID2label_id[list(anchor["wikidata_ids"])[0]]
                            except KeyError:
                                #if the label_id isn't here, set it to None
                                print(f"{list(anchor['wikidata_ids'])} not found in german kb.")
                                label_id = None
                            if args.description:
                                try:
                                    lang_desc_dict = {item[2]: item[1] for item in wikidataID2label_desc_lang[list(anchor["wikidata_ids"])[0]]}
                                except KeyError:
                                    lang_desc_dict = {}
                                    print(f'description for {list(anchor["wikidata_ids"])[0]} not found')
                            title = tmp_dict.get(lang, None)


                            if title and (
                                args.monolingual
                                or (not args.target_switching and not args.target_switch_x_lingual)
                                or (
                                    args.target_switching
                                    and np.random.rand() > args.target_switching_prob
                                )
                            ):
                                if args.action == "titles_lang":
                                    output_ = "{} >> {}".format(title, lang)
                                else:
                                    output_ = "{} >> {}".format(lang, title)

                                fs.write(input_ + "\n")
                                ft.write(output_ + "\n")

                            elif not args.monolingual and not args.target_switch_x_lingual:
                                #code from the GENRE researchers to make the answer randomly switch languages
                                if args.action == "titles_lang":
                                    choices = [
                                        "{} >> {}".format(title, lang2)
                                        for lang2, title in tmp_dict.items()
                                        if lang2 in args.allowed_langs and lang2 != lang
                                    ]
                                else:
                                    choices = [
                                        "{} >> {}".format(lang2, title)
                                        for lang2, title in tmp_dict.items()
                                        if lang2 in args.allowed_langs and lang2 != lang
                                    ]

                                for output_ in np.random.choice(
                                    choices,
                                    min(len(choices), args.random_n),
                                    replace=False,
                                ):
                                    fs.write(input_ + "\n")
                                    ft.write(output_ + "\n")
                            elif args.target_switch_x_lingual:
                                # code by xxx for specific crosslingual target switching
                                
                                # get src and tgt langs from the args
                                source_lang, target_lang = args.target_switch_x_lingual.split('|')
                                
                                #procede only if a target lang is in the lang_titles associated \w its wikidataID
                                #and the label_id is found
                                if target_lang in tmp_dict.keys() and label_id!=None:
                                    title = tmp_dict[target_lang]   

                                    #create GENRE input and output for the src and tgt files                                 
                                    if args.action == "titles_lang":
                                        output_ = "{} >> {}".format(title, target_lang)
                                    else:
                                        output_ = "{} >> {}".format(target_lang, title)
                                    if args.description:
                                        description = lang_desc_dict[target_lang] if target_lang in lang_desc_dict.keys() else ''
                                        output_ = output_ + " " + description
                                    
                                    #create blink input
                                    description = lang_desc_dict[target_lang] if target_lang in lang_desc_dict.keys() else ''

                                    blink_input = (
                                            create_input_blink( {
                                                        "context_left": left_context,
                                                        "mention": mention,
                                                        "context_right": right_context,
                                                        "wikidata_id": list(anchor["wikidata_ids"])[0]
                                                },
                                                128,
                                                label_title=title,
                                                label = WikidataID2langparagraphs.get(list(anchor["wikidata_ids"])[0], description) if args.RE_candidates == 'Voll' else description,
                                                label_id = label_id
                                            )
                                        )
                                    ent_count += 1

                                    #write away
                                    fs.write(input_ + "\n")
                                    ft.write(output_ + "\n")
                                    bl.write(json.dumps(blink_input, ensure_ascii=False) + '\n')
                        

                        elif args.action == "canonical_title":

                            if (
                                list(anchor["wikidata_ids"])[0]
                                in wikidataID2canonical_lang_title
                            ):
                                lang, title = wikidataID2canonical_lang_title[
                                    list(anchor["wikidata_ids"])[0]
                                ]
                                fs.write(input_ + "\n")
                                ft.write("{} >> {}".format(title, lang) + "\n")

                        elif args.action == "marginal":
                            output_ = " || ".join(
                                "{} >> {}".format(lang2, title)
                                for lang2, title in wikidataID2lang_title[
                                    list(anchor["wikidata_ids"])[0]
                                ]
                                if lang2 in args.allowed_langs
                            )
                            if output_:
                                fs.write(input_ + "\n")
                                ft.write(output_ + "\n")
    
    print(f'number of ents: {ent_count}')