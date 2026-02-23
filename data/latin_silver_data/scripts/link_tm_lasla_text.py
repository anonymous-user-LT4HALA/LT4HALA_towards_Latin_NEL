#--------------------
# preparation to connect to GLAUx
#--------------------
import sys
import logging
import lxml
import os
import lxml.etree


from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('version', type=str)
parser.add_argument('real_version', type=str)


args = parser.parse_args()

logging.basicConfig(filename=f'silver_pipeline_latin_{args.version}.log', level=logging.INFO)
logger = logging.getLogger(__name__)
import pandas as pd
import re
import pickle

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Voeg de console handler toe aan de root logger
logging.getLogger().addHandler(console_handler)


import re

def replace_dollars(text):
    # This pattern matches two or more consecutive $ symbols
    pattern = r'\${2,}'
    # Replace each match with the same number of dots
    result = re.sub(pattern, lambda m: '.' * len(m.group()), text)
    return result


def contains_non_zero_numbers(text):
    # This pattern matches any digit from 1 to 9
    pattern = r'[1-9]'
    # Search for the pattern in the text
    match = re.search(pattern, text)
    # Return True if a match is found, otherwise False
    return match is not None


def split_at_middle_dollar(text):
    # Find the middle index of the string
    middle_index = len(text) // 2

    # Initialize variables to track the closest $ position
    closest_dollar_index = -1
    min_distance = len(text)

    # Iterate through the string to find the closest $ to the middle
    for i, char in enumerate(text):
        if char == '$':
            distance = abs(i - middle_index)
            if distance < min_distance:
                closest_dollar_index = i
                min_distance = distance

    # If no $ is found, return the original string
    if closest_dollar_index == -1:
        return text

    # Split the string at the closest $ position
    part1 = text[:closest_dollar_index]
    part2 = text[closest_dollar_index + 1:]

    return part1, part2

def lists_overlap(list1, list2):
    return bool(set(list1) & set(list2))



def main():

    #--------------------
    # Combine author and non-author results from Trismegistos expansion
    #--------------------

    df_tmexpanded_authors = pd.read_csv('../Trismegistos_expansion/output/tm_linked_sources_authors.csv', index_col=0)
    df_tmexpanded_authors['subset'] = 'author'
    df_tmexpanded_non_authors = pd.read_csv('../Trismegistos_expansion/output/tm_linked_sources_non_authors.csv', index_col=0)
    df_tmexpanded_non_authors['subset'] = "non_author"

    df_tmexpanded = pd.concat([df_tmexpanded_authors, df_tmexpanded_non_authors])
    df_tmexpanded.reset_index(drop=True, inplace=True)

    #--------------------
    # Clean Trismegistos expansion
    #--------------------

    #get all cells that have matched using the algorithm (script get_tm_authorwork_ids)
    df_match_found = df_tmexpanded.loc[df_tmexpanded['original_response'].str.startswith("b'\\r\\n\\r\\n\\r\\n\\r\\nAUTHORMatch_CONDITIONALAuthorwork") == True, :].copy()
    df_match_found.drop(columns=[f'start_passage_{n}' for n in range(8)], inplace=True)
    df_match_found.drop(columns=[f'end_passege_{n}' for n in range(8)], inplace=True)
    df_match_found.reset_index(drop=True, inplace=True)


    # Iterate through the DataFrame rows
    for row in df_match_found.itertuples():
        orig_response = getattr(row, 'original_response')
        start, tm_author_work, human_readable_title, machine_readable = orig_response.split('@')

        if not contains_non_zero_numbers(machine_readable):
            row_start_passage = 'missing'
            row_end_passage = 'missing'
        else:
            possible_start_end_passages = machine_readable.split('|')
            row_start_passage = []
            row_end_passage = []

            for possible_start_end_passage in possible_start_end_passages:
                if contains_non_zero_numbers(possible_start_end_passage):
                    split_index = possible_start_end_passage.find('$')
                    machine_readable_passage_info = possible_start_end_passage[:split_index].strip()
                    machine_readable_numbers = possible_start_end_passage[split_index + 1:][:-3].strip()
                    passage1, passage2 = split_at_middle_dollar(machine_readable_numbers)

                    try:
                        if machine_readable_passage_info.split(' - ')[0] == machine_readable_passage_info.split(' - ')[1] and not passage1 == passage2:
                            print(f'some sort of mistake is happening in {machine_readable_passage_info} with {machine_readable_numbers}')
                        else:
                            if '..' not in passage1.replace('$', '.'):
                                row_start_passage.append(passage1.replace('$', '.')) 
                            if '..' not in passage2.replace('$', '.'):
                                row_end_passage.append(passage2.replace('$', '.')) 
                    except IndexError:
                        logging.info(f"error in {row.Index}: - split not possible for {machine_readable}")
                        continue
            if len(row_start_passage) == 0 or len(row_end_passage) == 0:
                row_start_passage = 'missing_or_incomplete'
                row_end_passage = 'missing_or_incomplete'
            else:
                row_start_passage = ','.join(row_start_passage)
                row_end_passage = ','.join(row_end_passage)

        # Add the results to the DataFrame
        df_match_found.at[row.Index, 'start_passage'] = row_start_passage
        df_match_found.at[row.Index, 'end_passage'] = row_end_passage





    logging.info(f"""Number of sources extracted from the RE entries: {len(df_tmexpanded)},
                 Number of those sources the TM algorithm could link to a text:  {len(df_match_found)},
                {len(df_match_found[df_match_found['subset'] == 'author'])} of those are authors, 
                {len(df_match_found[df_match_found['subset'] == 'non_author'])} are non authors""")
    


    logging.info(f"""Number of sources extracted from the RE entries: {len(df_tmexpanded)},
                 Number of those sources the TM algorithm could link to a text:  {len(df_match_found)},
                {len(df_match_found[df_match_found['subset'] == 'author'])} of those are authors, 
                {len(df_match_found[df_match_found['subset'] == 'non_author'])} are non authors""")

    df_match_found.reset_index(drop=True, inplace=True)

    #--------------------
    # Access GLAUx for a mapping from the GLAUx text id to the TM Authorwork id and map
    #--------------------

    columns = ['declined_form', 'lemma', 'word_type', 'LILA_lemma_id', 'LILA_token_id', 'nam_id', 'book', 'section', 'paragraph', 'wordref_id', 'TM_Real', 'TM_Real_checked', 'TM_Real_uncertain', 'TM_authorWork']
    df_lasla_export = pd.read_csv('../export/LASLA_export_v1.tab', sep='\t', header=None, quoting=3, dtype=str)
    df_lasla_export.columns = columns

    #remove all that are in one of the texts that have been manually annotated
    blacklist = {'4509', '5509', '1210', '494', '286', '15725', '9', '442'}

    
    df_lasla_export = df_lasla_export.loc[
        ~df_lasla_export['TM_authorWork'].astype(str).isin(blacklist)
    ].copy()

    #remove all that already have a link to the Real and don't need to be linked additionally
    df_lasla_export_no_re = df_lasla_export[df_lasla_export['TM_Real_checked'] != '1']
    df_lasla_export[df_lasla_export['TM_Real_checked'] == '1'].reset_index(drop=True).to_csv('gold_mentions_trismegistos_v2.csv')

    #remove all that don't have a nam_id to link
    linkable_lasla_tokens = df_lasla_export_no_re[df_lasla_export_no_re['nam_id']!= '0'].dropna(subset='nam_id')
    #clean the nam_ids
    linkable_lasla_tokens['clean_nams'] = linkable_lasla_tokens.nam_id.apply(lambda x: x.replace('multiple: ', '').split(' / '))
    print(f'length linkable lasla tokens = {len(linkable_lasla_tokens)}')

    #sources side: add nam_ids
    with open(f'../mappings/{args.real_version}/re_id2nam_id.pkl', 'rb') as f:
        re_id2nam_id = pickle.load(f)
    
    df_match_found['nam_id'] = df_match_found['RE_id'].map(lambda re_id: re_id2nam_id.get(str(re_id)), na_action='ignore')
    df_match_found.dropna(subset='nam_id', inplace=True)
    df_match_found.reset_index(drop=True, inplace=True)

    ## remove all texts that aren't in lasla
    relevant_texts = set(linkable_lasla_tokens['TM_authorWork'])
    df_relevant_matches = df_match_found[df_match_found.tm_authorwork_id.isin(relevant_texts)]
    print(f'length sources in lasla = {len(df_relevant_matches)}')
    ## relevant_text structures

    
    def get_structure(row):
        structure = []
        for col in ['book', 'section', 'paragraph']:
            if pd.notnull(row[col]) and row[col] != '':
                structure.append(col)
        return ','.join(structure)

    # Get first row per text_id
    first_rows = linkable_lasla_tokens.sort_values(by='TM_authorWork').drop_duplicates('TM_authorWork')

    # Map structure
    structure_map = first_rows.set_index('TM_authorWork').apply(get_structure, axis=1)

    # Assign to original DataFrame
    linkable_lasla_tokens['text_structure'] = linkable_lasla_tokens['TM_authorWork'].map(structure_map)

    #about half do not contain passage information at all
    logging.info(f'Number of entities that contain paragraph information {len(df_relevant_matches[df_relevant_matches.start_passage != "missing"])}')
    df_relevant_matches = df_relevant_matches[df_relevant_matches.start_passage != "missing"]
    df_relevant_matches['start_end_passage'] = df_relevant_matches.apply(lambda x: list(zip(x['start_passage'].split(','), x['end_passage'].split(','))), axis=1)
    logging.info(f"sanity check: \n{df_relevant_matches['start_end_passage'].head()}")
    logging.info(f'length before exploding {len(df_relevant_matches)}')
    
    # we explode, so that if we have multiple possible passages, they all have their own row
    df_relevant_matches_exploded = df_relevant_matches.explode('start_end_passage')
    df_relevant_matches_exploded['start_passage_good'] = df_relevant_matches_exploded.start_end_passage.apply(lambda x: x[0])
    df_relevant_matches_exploded['end_passage_good'] = df_relevant_matches_exploded.start_end_passage.apply(lambda x: x[1])
    logging.info(f'length after exploding{len(df_relevant_matches_exploded)}')
    logging.info(f'sanity check:{df_relevant_matches_exploded.head()}')
    df_relevant_matches_exploded.to_csv('test.csv')

    final_mentions_df = pd.DataFrame()

    for item in df_relevant_matches_exploded.to_dict(orient='records'):
        if item.get('start_passage_good') == item.get('end_passage_good'):
            passage = item.get('start_passage_good')
            text = item.get('tm_authorwork_id')
            nam_id = item.get('nam_id')
            RE_id = item.get('RE_id')
            artikel = item.get('RE_artikel')
            source = item.get('source')
            # These are all items where we could find the passage
            if len(passage.split('.')) == len(structure_map[text].split(',')) or structure_map[text].split(',')[0] !='book':
                # first, filter the linkable lasla tokens on the correct text
                mentions_in_text = linkable_lasla_tokens.loc[linkable_lasla_tokens['TM_authorWork'] == text]
                mentions_in_text.reset_index(drop=True, inplace=True)
                # second, filter on all tokens with a relevant passage
                filters = dict(zip(structure_map[text].split(','), passage.split('.')))
                for col, val in filters.items():
                    mentions_in_text = mentions_in_text.loc[mentions_in_text[col] == val]
                    mentions_in_text.reset_index(drop=True, inplace=True)
                # third, for the tokens that are left, check if any of them contain (one of) the desired nam ids
                if len(mentions_in_text) > 0:
                    # if yes, add relevant metadata and add to final df
                    lasla_mentions = mentions_in_text.loc[mentions_in_text.clean_nams.apply(lambda x: lists_overlap(x, nam_id))]      
                    lasla_mentions = lasla_mentions.copy()  # Make a deep copy
                    lasla_mentions['TM_Real_tm'] = lasla_mentions['TM_Real']
                    lasla_mentions['TM_Real'] = RE_id
                    lasla_mentions['RE_artikel'] = artikel
                    lasla_mentions['passage_searched_silver'] = passage
                    lasla_mentions['source'] = source
                    final_mentions_df = pd.concat([final_mentions_df, lasla_mentions])
    
    # remove duplicates 
    # (if the same token manages to get two (or more) entities assigned to it, all are removed)
    final_mentions_df.drop_duplicates(subset='LILA_token_id', keep=False, inplace=True)
    final_mentions_df.reset_index(drop=True, inplace=True)
    final_mentions_df.to_csv('test_silver_latin_v2.csv')

            

if __name__ == '__main__':
    main()
