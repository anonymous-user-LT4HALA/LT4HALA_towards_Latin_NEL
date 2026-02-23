import sys
import logging
import lxml
import os
import lxml.etree


from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('input_file', type=str, help='Path to the input tab file')
parser.add_argument('output_file', type=str, help='Path to the output file')
parser.add_argument('version', type=str, help='version of the dataset')
parser.add_argument('real_version')
parser.add_argument('real_candidates', type=str, default='kurz', help='Whether to use the kurztext or the volltext as the real candidates')
parser.add_argument('subset')
parser.add_argument('--output_format', default='csv')
parser.add_argument('--check_multi_and_number', action='store_true')

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from utils.utils import *


args = parser.parse_args()

print(args.check_multi_and_number)

logging.basicConfig(filename=f'silver_pipeline_latin_{args.version}.log', level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.insert(1, "../../../../glaux_database_connector")
#%%

from conllu import parse
import os
import pandas as pd

def get_kwic_from_conllu(tokenlist,index_start, index_end=None):
    if index_end==None:
        index_end = index_start
    left_context = ' '.join([word['form'] for word in tokenlist[:index_start] if 'tokenuri' in word.keys()])
    right_context = ' '.join([word['form'] for word in tokenlist[index_end+1:] if 'tokenuri' in word.keys()])
    mention = ' '.join([word['form'] for word in tokenlist[index_start:index_end+1] if 'tokenuri' in word.keys()])
    lemma = ' '.join([clean_lemma(word['lemma']) for word in tokenlist[index_start:index_end+1] if 'tokenuri' in word.keys()])

    return mention, lemma, left_context, right_context

# def expand_contexts(left_context, right_context, sentences, sentence_found):
#     # to the left context, add the sentence before:
#     prev_sent = sentences[max(sentence_found-1, 0)]
#     prev_sent = ' '.join([token['form'] for token in prev_sent if 'tokenuri' in token.keys()])
#     left_context = prev_sent + '. ' + left_context
#     # to the right context, add the sentence after:
#     next_sent = sentences[min(sentence_found+1, len(sentences)-1)]
#     next_sent = ' '.join([token['form'] for token in next_sent if 'tokenuri' in token.keys()])
#     right_context = right_context + '. ' + next_sent
#     return left_context, right_context


from typing import List, Tuple

def _sentence_text(sent_tokens: List[dict]) -> str:
    """
    Reconstruct a sentence string from tokens that contain 'tokenuri'.
    Joins token['form'] with spaces.
    """
    return ' '.join(token['form'] for token in sent_tokens if 'tokenuri' in token)

def _word_count(text: str) -> int:
    """
    Counts words by splitting on whitespace. Adjust if you need more rigorous tokenization.
    """
    return len([w for w in text.split() if w.strip()])

def _concat_with_period(left: str, right: str, prefer_space: bool = True) -> str:
    """
    Concatenates two strings with a single period separator if needed.
    Avoids duplicating punctuation like '..'.
    """
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left

    # Check if left already ends with sentence-final punctuation
    if left[-1] in '.!?':
        sep = ' ' if prefer_space else ''
    else:
        sep = '. ' if prefer_space else '.'
    return f"{left}{sep}{right}"

def expand_side_until(
    base_text: str,
    sentences: List[List[dict]],
    start_index: int,
    direction: str,
    min_words: int = 50,
    max_extra: int | None = None,
) -> Tuple[str, int]:
    """
    Expands base_text by adding sentences either to the left or right of start_index
    until it reaches at least min_words words, or we hit boundaries, or max_extra is reached.

    direction: 'left' or 'right'
    Returns: (expanded_text, num_added)
    """
    assert direction in ('left', 'right')
    n = len(sentences)
    expanded = base_text
    added = 0

    # Current pointer just outside the focal sentence in the chosen direction
    idx = start_index - 2 if direction == 'left' else start_index + 2

    def can_continue(i: int) -> bool:
        if direction == 'left':
            return i >= 0
        else:
            return i < n

    while _word_count(expanded) < min_words and can_continue(idx):
        # Guard against optional max expansion
        if max_extra is not None and added >= max_extra:
            break

        sentence_str = _sentence_text(sentences[idx])

        if direction == 'left':
            expanded = _concat_with_period(sentence_str, expanded)
            idx -= 1
        else:
            expanded = _concat_with_period(expanded, sentence_str)
            idx += 1

        added += 1

    return expanded, added

def expand_contexts(
    left_context: str,
    right_context: str,
    sentences: List[List[dict]],
    sentence_found: int,
    min_words: int = 50,
    max_extra_left: int | None = None,
    max_extra_right: int | None = None,
) -> Tuple[str, str]:
    """
    Ensures left_context and right_context each have at least `min_words`.
    Starts by adding the immediate previous/next sentence (like your original),
    then continues expanding outward until min_words is met or boundaries reached.

    - sentences: list of sentences; each sentence is a list of token dicts
    - sentence_found: index of the focal sentence (0-based)
    """
    n = len(sentences)
    # --- Add the immediate previous sentence to the left (original behavior) ---
    if n > 0:
        prev_idx = max(sentence_found - 1, 0)
        prev_sent = _sentence_text(sentences[prev_idx])
        left_context = _concat_with_period(prev_sent, left_context)

    # --- Add the immediate next sentence to the right (original behavior) ---
    if n > 0:
        next_idx = min(sentence_found + 1, n - 1)
        next_sent = _sentence_text(sentences[next_idx])
        right_context = _concat_with_period(right_context, next_sent)

    # --- Now expand further until each side meets min_words ---
    left_context, _ = expand_side_until(
        base_text=left_context,
        sentences=sentences,
        start_index=sentence_found,
        direction='left',
        min_words=min_words,
        max_extra=max_extra_left,
    )

    right_context, _ = expand_side_until(
        base_text=right_context,
        sentences=sentences,
        start_index=sentence_found,
        direction='right',
        min_words=min_words,
        max_extra=max_extra_right,
    )

    return left_context, right_context



def check_multitokens(entity_nam_ids, mention_nam_ids):
    return all(mention_nam_id in entity_nam_ids for mention_nam_id in mention_nam_ids)


def to_basic_mention_df(rows, lemma, left_context, right_context, mention, number):
    blink_mention = {}
    blink_mention["RE_id"] = str(rows.iloc[0, :].TM_Real)
    blink_mention['left_context'] = left_context
    blink_mention['right_context'] = right_context
    blink_mention['mention'] = mention
    blink_mention['token_id'] = ','.join(rows['LILA_token_id'].values.tolist())
    blink_mention['lemma'] = lemma
    blink_mention['mention_nam_ids'] = ','.join(
        nam for nam in rows['nam_id'].values.tolist()
        if isinstance(nam, str)
    )
    blink_mention['authorwork_id'] = rows.authorwork_id.values.tolist()[0]
    if number:
        blink_mention['number_ok'] = False if 'Plur' in number else True
    else:
        blink_mention['number_ok'] = True
    blink_mention['number'] = number
    return blink_mention

mention_df = pd.read_csv(args.input_file, index_col=0, dtype = {'TM_authorWork': str})
mention_df.rename(columns={'TM_authorWork':'authorwork_id'}, inplace=True)
mention_df.dropna(subset='TM_Real', inplace=True)
mention_df['TM_Real'] = mention_df.TM_Real.astype(int)
mention_df.reset_index(drop=True, inplace=True)

mention_df['file_id'] = mention_df.LILA_token_id.apply(lambda x: x.split('.')[1].split('/')[-1])

# Create a helper column to detect breaks in sequence
mention_df['group_flag'] = (mention_df['wordref_id'].diff() != 1) | (mention_df['TM_Real'].diff() != 0)

# Cumulative sum of breaks to create group IDs
mention_df['group_id'] = mention_df['group_flag'].cumsum()
print(len(mention_df))
print(len(list(mention_df.group_id.unique())))

# real = pd.read_csv(f'../../../Paulys_kb/{real_version}/blink_real_dict_{real_version}.csv')

#logging.info(f'merging Volltext and Kurztext')
# = pd.merge(mention_df, real[['RE_id', 'Artikel', 'Volltext', 'Kurztext', 'label_id']], on='RE_id')


conllu_lasla_folder = f"{REPO_ROOT}/../LASLALinkedLila/conllup"
lasla_fields = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC', 'LILA:FLCAT', 'LILA:SENTID', 'LILA:LINE', 'LILA:LEMMA', 'TOKENURI']
lasla_fields = [item.lower() for item in lasla_fields]

flat_csv_prep = []

from tqdm import tqdm

# we open the file in question and parse using the CONLLU library 
# load them all into memory because this is faster

# Preload all files into memory
parsed_files = {}
for file_name in os.listdir(conllu_lasla_folder):
    if file_name.endswith('.conllup'):
        with open(os.path.join(conllu_lasla_folder, file_name)) as f:
            data = f.read()
            parsed_files[file_name.replace('.conllup', '')] = parse(data, fields=lasla_fields)

# def get_data(name):
#     with open(os.path.join(conllu_lasla_folder, name + '.conllup')) as f:
#         data = f.read()
#         parsed_data = parse(data, fields=lasla_fields)
#     return parsed_data

from itertools import islice

## first, we groupby the files
# for name, group in tqdm(islice(mention_df.groupby('file_id'), 1)):
for name, group in tqdm(mention_df.groupby('file_id')):
    sentences = parsed_files[name]
    #uncomment when debugging
    # sentences = get_data(name)
    # we additionally group on the group id we made earlier for multiwords
    for name, mention_group in group.groupby('group_id'):
        token_ids = mention_group.LILA_token_id.values.tolist()
        # for sentence_found, sentence in enumerate(sentences):
        for count, sentence in enumerate(sentences):
            tokenlist = sentence.filter(tokenuri=lambda x: x in token_ids)
            if tokenlist:
                if len(tokenlist) == 1:
                    index = sentence.index(tokenlist[0])
                    number = tokenlist[0]['feats'].get('Number')
                    tokenuri = tokenlist[0]['tokenuri']
                    mention, lemma, left_context, right_context = get_kwic_from_conllu(sentence, index)
                    left_context, right_context = expand_contexts(left_context, right_context, sentences, count)
                # the tokens from the found tokenlist HAVE to belong to the same entity 
                # because we already found split on the entity groups earlier
                else:
                    start_index = sentence.index(tokenlist[0])
                    end_index = sentence.index(tokenlist[-1])
                    number = [token['feats'].get('Number') for token in tokenlist if token['feats'].get('Number') is not None]
                    tokenuri = ','.join([item['tokenuri'] for item in tokenlist])
                    mention, lemma, left_context, right_context = get_kwic_from_conllu(sentence, start_index, end_index)
                    left_context, right_context = expand_contexts(left_context, right_context, sentences, count)

                mention = to_basic_mention_df(mention_group, lemma, left_context, right_context, mention, number)
                flat_csv_prep.append(mention)

print(flat_csv_prep[0])
df = pd.DataFrame(flat_csv_prep)
df['subset'] = args.subset
authorwork_dates_df = pd.read_csv(f'{REPO_ROOT}/data/authorwork_dates.csv', dtype={'authorwork_id': str})
df = df.merge(authorwork_dates_df, on='authorwork_id', how='left')
df.rename(columns={'y1':'mention_start_date','y2':'mention_end_date'}, inplace=True)
print(df[['mention_start_date', 'mention_end_date']])

if args.output_format == 'csv':
    # mention_df.to_csv(os.path.join(output_dir, args.output_file + '.csv'))
    final_df = to_real_csv(df, args.real_version)
    print(final_df)
    print(final_df.columns)


    ## add final multitoken check
    print(len(final_df))
    if args.check_multi_and_number:
                # Preconditions checks (optional but helpful)
        required_cols = {'name_complete_id', 'mention_nam_ids', 'number_ok'}
        missing = required_cols - set(final_df.columns)
        if missing:
            raise KeyError(f"final_df is missing required column(s): {missing}")

        # Compute 'multitoken_ok' safely. If your check_multitokens can handle empty/NaN, great.
        # Otherwise, guard against NaN by replacing with safe defaults.
        def _safe_split(val, sep, default=''):
            # Handles NaN and ensures string before split
            return str(val if pd.notna(val) else default).split(sep)

        final_df = final_df.copy()  # ensure we're working on an owning frame
        final_df['multitoken_ok'] = final_df.apply(
            lambda x: check_multitokens(
                _safe_split(x['name_complete_id'], ' / '),
                _safe_split(x['mention_nam_ids'], ',')
            ),
            axis=1
        )

        # First filter: multitoken_ok
        mask_multis = final_df['multitoken_ok'].fillna(False)
        ok_multis_final = final_df.loc[mask_multis].copy()
        print(f"length after filtering multis: {len(ok_multis_final)}")

        # Second filter: number_ok
        # If 'number_ok' isn't boolean yet, coerce and handle NaN as False
        number_ok_bool = ok_multis_final['number_ok']
        if number_ok_bool.dtype != bool:
            number_ok_bool = number_ok_bool.astype('bool', errors='ignore')
        mask_number = number_ok_bool.fillna(False)

        ok_number_multis_final = ok_multis_final.loc[mask_number].copy()
        print(f"length after filtering number_ok: {len(ok_number_multis_final)}")

        # Save the final filtered frame (was previously saving the first filter)
        ok_number_multis_final.to_csv(args.output_file, index=False)


    else:
        final_df.to_csv(args.output_file)

#     final_df.to_csv(os.path.join(args.output_file))
else:
    blink_mentions_jsonl = to_jsonl_blink_from_df(df, args.real_version, args.real_candidates, language='latin')
    output_file = os.path.join(args.output_file + '.jsonl')
    df_write_jsonl(output_file, blink_mentions_jsonl)       


        


                
# print(mention_df.LILA_token_id[0])
# %%
