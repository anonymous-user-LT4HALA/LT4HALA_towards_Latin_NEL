import os
import pandas as pd
import json
import re

from pathlib import Path

real_dict_path = Path("Paulys_kb")

base_path = Path(__file__).resolve().parents[2]
base_real_dict_path = base_path / real_dict_path

def to_real_csv(df, real_version):
    file_name = f"blink_real_dict_{real_version}.csv"
    file_path = os.path.join(base_real_dict_path, real_version, file_name)

    real_df = pd.read_csv(file_path, dtype=str, index_col=0)

    # Work on a copy to avoid side effects
    df = df.copy()

    # Standardize column name
    df.rename(columns={'TM_Real': 'RE_id'}, inplace=True)

    before_len = len(df)

    # Merge
    df_merged = df.merge(real_df, how='left', on='RE_id')

    after_merge_len = len(df_merged)

    # Drop rows that failed to match anything in real_df
    # Assumes RE_id must exist and match
        # Keep only rows whose RE_id exists in real_df
    valid_real_ids = set(real_df["RE_id"].dropna())
    df_clean = df_merged[df_merged["RE_id"].isin(valid_real_ids)]

    after_filter_len = len(df_clean)

    print(f"Rows before merge: {before_len}")
    print(f"Rows after merge: {after_merge_len}")
    print(f"Rows after dropping non-merged: {after_filter_len}")
    print(f"Dropped rows: {after_merge_len - after_filter_len}")

    return df_clean

###
# add to the entity if the entity is a god y
# or whether the entity is a TM Author
###

def to_jsonl_blink_from_df(df, real_version, real_candidates='Voll', language='greek', world='subset'):
    dataset = []
    file_name = f"blink_real_dict_{real_version}_{real_candidates}.jsonl"
    file_path = os.path.join(base_real_dict_path, real_version, file_name)
    print(file_path)

    # Load the JSONL with all fields as strings
    real_df = pd.read_json(file_path, lines=True, dtype=str)

    # Align key and merge text/context in
    df = df.copy()
    # df.rename(columns={'TM_Real': 'RE_id'}, inplace=True)
    df_with_text = df.merge(real_df, on="RE_id", suffixes=("", "_dup"))

    # Drop duplicate columns if they are exactly identical to the original
    for col in list(df_with_text.columns):
        dup = f"{col}_dup"
        if dup in df_with_text.columns:
            if df_with_text[col].equals(df_with_text[dup]):
                df_with_text.drop(columns=dup, inplace=True)

    # Define the canonical keys for the example
    core_keys = {
        "context_left",
        "context_right",
        "mention",
        "text",
        "RE_id",
        "label_id",
        "label_title",
        "serial",
        "world"
    }

    # Iterate rows and build examples
    for row in df_with_text.itertuples(index=False):
        # Build the main example fields (use getattr with fallback to None)
        example = {}
        example["context_left"] = getattr(row, "left_context")
        example["context_right"] = getattr(row, "right_context")
        example["mention"] = getattr(row, "mention")
        example["text"] = getattr(row, "text")
        example["RE_id"] = getattr(row, "RE_id")
        example["label_id"] = getattr(row, "label_id")
        example["label_title"] = getattr(row, "title")

        # Choose serial from priority list (first available)
        for key in ("token_id", "glaux_id", "serial"):
            if hasattr(row, key):
                example["serial"] = getattr(row, key)
                break
        if "serial" not in example:
            example["serial"] = None

        # World can be a column name provided as a parameter
        example["world"] = getattr(row, world, None)

        # Build metadata: include all columns not already in core example keys
        # Grab a dict view of the row with column names mapped to values
        row_dict = row._asdict() if hasattr(row, "_asdict") else {c: getattr(row, c, None) for c in df_with_text.columns}
        metadata = {}
        for col_name, value in row_dict.items():
            if col_name not in core_keys:
                # Exclude columns already mapped into the core fields by different names
                # (e.g., 'title' went to 'label_title', 'left_context' went to 'context_left', etc.)
                if col_name in ("title", "left_context", "right_context", "token_id", "glaux_id", "serial"):
                    continue
                # Also skip the world column used to map into 'world'
                if col_name == world:
                    continue
                metadata[col_name] = value

        example["metadata"] = metadata

        dataset.append(example)
        
    return dataset

   


def df_write_jsonl(dataset, output_file):
    with open(output_file, 'w', encoding='UTF-8') as f:
        for entry in dataset:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')


def clean_lemma(lemma):
    # Lowercase
    lemma = lemma.lower()
    
    # Remove Arabic numbers
    lemma = re.sub(r'\d', '', lemma)
    
    # Replace 'v' with 'u', 'j' with 'i'
    lemma = lemma.replace('v', 'u').replace('j', 'i')
    
    return lemma