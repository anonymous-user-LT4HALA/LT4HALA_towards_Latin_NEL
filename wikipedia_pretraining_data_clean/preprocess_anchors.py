# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import pickle
import re
import sys
sys.path.append('GENRE')


from genre.utils import chunk_it, get_wikidata_ids
from tqdm.auto import tqdm, trange


def clean_anchor_lang(anchor, lang):
    if re.match(r"^https%3A//(.*)\.wikipedia\.org/wiki/", anchor):
        anchor = anchor[10:].split("/")
        return clean_anchor_lang(anchor[2], anchor[0].split(".")[0])
    elif anchor.startswith("%3A{}".format(lang)):
        return clean_anchor_lang(anchor[len("%3A{}".format(lang)) :], lang)
    elif anchor.startswith("%3A"):
        return clean_anchor_lang(anchor[len("%3A") :], lang)
    elif anchor.startswith("w%3A{}".format(lang)):
        return clean_anchor_lang(anchor[len("w%3A{}".format(lang)) :], lang)
    elif anchor.startswith("w%3A"):
        return clean_anchor_lang(anchor[len("w%3A") :], lang)
    else:
        return anchor, lang


import html
import re
import unicodedata
from typing import Iterable

def clean_paragraphs(paragraphs: Iterable[str]) -> str:
    """
    Clean and normalize a sequence of paragraphs:
      - Replace newlines/tabs with spaces
      - HTML-unescape entities (e.g., &nbsp;, &amp;)
      - Decode unicode escape sequences (e.g., \\u00a0) if present
      - Replace non-breaking spaces with normal spaces
      - Collapse repeated whitespace
      - Strip leading/trailing spaces
      - Normalize Unicode (NFC)

    Args:
        paragraphs: iterable of paragraph strings

    Returns:
        A single cleaned string, paragraphs joined by a single space.
    """
    cleaned_paragraphs = []
    for para in paragraphs:
        if para is None:
            continue
        # Basic whitespace cleanup  
        para = para.replace('\n', ' ').replace('\t', ' ').strip()
        para = html.unescape(para)
        print(para)

        # Decode unicode escape sequences if present (e.g., "\\u00a0")
        # Guarded so normal text with stray backslashes isn't over-decoded.
        if "\\u" in para or "\\x" in para:
            try:
                # Encode to bytes then decode with 'unicode_escape'
                para = para.encode("utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                print('here')
                # Fall back silently if decoding isn't applicable
                pass

        # Replace non-breaking spaces (and similar) with normal spaces
        para = para.replace("\u00A0", " ").replace("\u202F", " ").replace("\u2007", " ")

        # Normalize Unicode to a canonical form
        para = unicodedata.normalize("NFC", para)

        # Collapse multiple whitespace to a single space
        para = re.sub(r"\s+", " ", para).strip()

        cleaned_paragraphs.append(para)

    # Join paragraphs with a single space
    # one to omit the title
    return " ".join(cleaned_paragraphs[1:])



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "step",
        type=str,
        choices=["prepare", "solve", "fill", "paragraphs"],
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
        "--langs",
        type=str,
        help="Pipe (|) separated list of language ID to process.",
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

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)

    if args.step == "paragraphs":
        for lang in args.langs.split("|"):
            results = {}


            filename = os.path.join(
                args.base_wikipedia,
                "{}".format(lang),
                "{}wiki.pkl".format(lang),
            )
            logging.info('Loading {}'.format(filename))
            with open(filename, "rb") as f:
                wiki = pickle.load(f)
            wikiid2wikititle = {key:value['title'] for key, value in wiki.items()}
            wikititle2wikiid = {value['title']:key for key, value in wiki.items()}

            filename = os.path.join(args.base_wikidata, "lang_title2wikidataID.pkl")
            logging.info("Loading {}".format(filename))
            with open(filename, "rb") as f:
                lang_title2wikidataID = pickle.load(f)

            for itemlang, title in lang_title2wikidataID.keys():
                if itemlang == lang:
                    if title in wikititle2wikiid.keys():
                        paragraphs = wiki[wikititle2wikiid[title]]['paragraphs']
                        paragraphs = clean_paragraphs(paragraphs)
                        wikidata_id = lang_title2wikidataID[(lang, title)]
                        results[list(wikidata_id)[0]] = paragraphs
                    else:
                        print(f"{title} with wikidata_id {lang_title2wikidataID[(lang, title)]} not found in wiki")
                        continue

            filename = os.path.join(
                args.base_wikipedia,
                lang,
                "WikidataID2{}paragraphs".format(lang),
            )

            logging.info("Saving {}".format(filename))
            with open(filename, "wb") as f:
                pickle.dump(results, f)

    elif args.step == "solve":
        filename = os.path.join(args.base_wikidata, "lang_title2wikidataID.pkl")
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            lang_title2wikidataID = pickle.load(f)

        filename = os.path.join(args.base_wikidata, "lang_redirect2title.pkl")
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            lang_redirect2title = pickle.load(f)

        filename = os.path.join(args.base_wikidata, "label_or_alias2wikidataID.pkl")
        logging.info("Loading {}".format(filename))
        with open(filename, "rb") as f:
            label_or_alias2wikidataID = pickle.load(f)

        for lang in args.langs.split("|"):
            filename = os.path.join(
                args.base_wikipedia,
                lang,
                "{}wiki_anchors.pkl".format(lang),
            )
            logging.info("Loading {}".format(filename))
            with open(filename, "rb") as f:
                anchors = pickle.load(f)

            results = {
                anchor: get_wikidata_ids(
                    *clean_anchor_lang(anchor, lang),
                    lang_title2wikidataID,
                    lang_redirect2title,
                    label_or_alias2wikidataID,
                )
                for anchor in tqdm(anchors)
            }

            filename = os.path.join(
                args.base_wikipedia,
                lang,
                "{}wiki_anchors_maps.pkl".format(lang),
            )
            logging.info("Saving {}".format(filename))
            with open(filename, "wb") as f:
                pickle.dump(results, f)

    elif args.step == "fill":
        for lang in args.langs.split("|"):
            filename = os.path.join(
                args.base_wikipedia, lang, "{}wiki_anchors_maps.pkl".format(lang)
            )
            logging.info("Loading {}".format(filename))
            with open(filename, "rb") as f:
                anchors_map = pickle.load(f)

            filename = os.path.join(
                args.base_wikipedia, lang, "{}wiki.pkl".format(lang)
            )
            logging.info("Loading {}".format(filename))
            with open(filename, "rb") as f:
                wiki = pickle.load(f)

            for page in tqdm(wiki.values()):
                page["anchors"] = [
                    {
                        **anchor,
                        "wikidata_ids": anchors_map[anchor["href"]][0],
                        "wikidata_src": anchors_map[anchor["href"]][1],
                    }
                    for anchor in page["anchors"]
                ]

            filename = os.path.join(
                args.base_wikipedia, lang, "{}wiki.pkl".format(lang)
            )
            logging.info("Saving {}".format(filename))
            with open(filename, "wb") as f:
                pickle.dump(wiki, f)

            anchors_solved = sum(
                len(a["wikidata_ids"]) == 1
                for page in wiki.values()
                for a in page["anchors"]
            )
            anchors_total = sum(
                not (len(a["wikidata_ids"]) == 0 and a["wikidata_src"] == "simple")
                for page in wiki.values()
                for a in page["anchors"]
            )
            logging.info(
                "LANG: {} -- Solved {:.2%} of anchors".format(
                    lang, anchors_solved / anchors_total
                )
            )

    elif args.step == "prepare":
        for lang in args.langs.split("|"):
            results = {}
            for rank in trange(32):
                filename = os.path.join(
                    args.base_wikipedia,
                    "{}".format(lang),
                    "{}wiki{}.pkl".format(lang, rank),
                )
                if os.path.exists(filename):
                    logging.info("Loading {}".format(filename))
                    with open(filename, "rb") as f:
                        for k, v in pickle.load(f).items():
                            results[k] = v
            logging.info(f'got {len(results)} results')
            logging.info(f'first results {list(results.items())[0]}')


            filename = os.path.join(
                args.base_wikipedia,
                "{}".format(lang),
                "{}wiki.pkl".format(lang),
            )
            logging.info("Saving {}".format(filename))
            with open(filename, "wb") as f:
                pickle.dump(results, f)

            anchors = []
            iter_ = tqdm(results.items())
            for k, v in iter_:
                anchors += [a["href"] for a in v["anchors"]]
                iter_.set_postfix(anchors=len(anchors), refresh=False)

            anchors = list(set(anchors))
            logging.info(f'got {len(anchors)} anchors')


            filename = os.path.join(
                args.base_wikipedia,
                "{}".format(lang),
                "{}wiki_anchors.pkl".format(lang),
            )
            logging.info("Saving {}".format(filename))
            with open(filename, "wb") as f:
                pickle.dump(anchors, f)
