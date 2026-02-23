# %% [markdown]
# # Find references to Authors starting from the real

# %%
import requests
from urllib.parse import quote
import re
tm_interpreter = 'FORBIDDEN'

def clean_source(source):
    if source.endswith(' ,') or source.endswith(' .'):
        source = source[:-2]
    return source

def request_tm_source(query):
    clean_query = quote(query)
    response = requests.get(tm_interpreter+clean_query)
    if response.status_code == 200:
        return str(response.content)
    else:
        print(f'{query} not found, exited with status_code {response.status_code}')
        return None

def parse_start_end(passages):
    start_end = passages.split('$')[1:-1]
    split_index = int(len(start_end)/2)
    start_passage = '.'.join(start_end[0:split_index])
    end_passage = '.'.join(start_end[split_index:])
    return start_passage, end_passage

def parse_result(response):
    if 'NOT' in response:
        return None
    else:
        parts = response.split('@')
        try:
            tm_authorwork_id = parts[1].replace('https://www.trismegistos.org/authorwork/', '')
            work_name = parts[2]
            passages = parts[3]
            if '|' in passages:
                options = passages.split('|')
            else:
                options = [passages]
            result_dict = {'tm_authorwork_id': tm_authorwork_id,
                           'work_name': work_name}
            for num, option in enumerate(options):
                start_passage, end_passage = parse_start_end(option)
                result_dict[f'start_passage_{num}'] = start_passage
                result_dict[f'end_passege_{num}'] = end_passage
            return result_dict
        except IndexError:
            print(f'problem in {response}')


test_query = 'Appian. 248f. ,'
# test_query = 'BLA'

parse_result(request_tm_source(clean_source(test_query)))

# %%
import pandas as pd

df = pd.read_csv('negative_data/confirmed_sources_from_non_authors.csv', header=None)
df.rename(columns={num: name for num, name in enumerate(['RE_id', 'tm_authorwork_id', 'source', 'RE_artikel', 'tm_author_id'])}, inplace=True)
df.head()

# %%
import tqdm
import time

df['original_response'] = 'missing'

for row in tqdm.tqdm(df.itertuples()):
    response = request_tm_source(row.source)
    if response != None:
        df.loc[row.Index, 'original_response'] = response
        if parse_result(response) != None:
            result_dict = parse_result(response)
            for key, value in result_dict.items():
                if key not in df.columns:
                    df[key] = 'missing'
                else:
                    df.loc[row.Index, key] = value
            time.sleep(1)

# %%
df.to_csv('negative_data/tm_linked_sources_non_authors.csv')

# %%



