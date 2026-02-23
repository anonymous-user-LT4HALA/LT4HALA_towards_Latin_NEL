import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def shuffle_and_split_by_group(df, group_column, split_percentage=0.8, random_state=None):
    unique_groups = df[group_column].drop_duplicates()
    groups_train, groups_test = train_test_split(
        unique_groups,
        test_size=(1 - split_percentage),
        random_state=random_state,
        shuffle=True
    )

    # Filter the DataFrame based on the split groups
    train = df[df[group_column].isin(groups_train)]
    df_test = df[df[group_column].isin(groups_test)]

    return train, df_test

from sklearn.model_selection import train_test_split


import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

def split_jsonl(
    input_path_voll,
    input_path_kurz,
    train_path,
    valid_path,
    test_path,
    train_ratio=0.95,
    seed=42,
    report_path=None
):
    """
    Splits input JSONL files (Kurz/Voll variants), writes train/dev/test JSONL,
    and writes a report file with:
      1) lengths of initial files,
      2) lengths of resulting train/dev/test,
      3) value_counts of the `in_training` column for dev and test.
    """

    # --- Load ---
    df_kurz = pd.read_json(input_path_kurz, lines=True)
    df_voll = pd.read_json(input_path_voll, lines=True)

    # --- Basic sanity checks / alignment ---
    # 1) Ensure both files have the same number of rows

    assert df_kurz.drop(columns=['text']).equals(df_voll.drop(columns=['text'])), \
        "df_kurz and df_voll (without 'label') differ."


    # 2) Create a stable id we can carry through splits
    df_kurz = df_kurz.copy()
    df_voll = df_voll.copy()
    df_kurz['mention_id'] = df_kurz.index
    df_voll['mention_id'] = df_voll.index

    label_ids = list(df_kurz.label_id.unique())
    mid_point = int(len(label_ids)/2)
    label_ids_seen = label_ids[:mid_point]
    label_ids_unseen = label_ids[mid_point:]
    df_unseen = df_kurz.loc[df_kurz.label_id.isin(label_ids_unseen)]
    df_seen = df_kurz.loc[df_kurz.label_id.isin(label_ids_seen)]

    # Assumes a helper exists:
    #   shuffle_and_split_by_group(df, group_col, split_percentage, random_state)
    train_unseen, test_unseen = shuffle_and_split_by_group(
        df_unseen, 'label_id', split_percentage=0.85, random_state=seed
    )
    test_unseen, dev_unseen = shuffle_and_split_by_group(
        test_unseen, 'label_id', split_percentage=0.5, random_state=seed
    )

    # Seen part: simple random splits
    train_seen, test_seen = train_test_split(
        df_seen, train_size=train_ratio, random_state=seed
    )
    dev_seen, test_seen = train_test_split(
        test_seen, train_size=0.5, random_state=seed
    )

    # Merge splits
    train = pd.concat([train_seen, train_unseen]).reset_index(drop=True)
    dev = pd.concat([dev_seen, dev_unseen]).reset_index(drop=True)
    test = pd.concat([test_seen, test_unseen]).reset_index(drop=True)

    # Mark whether dev/test labels occur in training
    if 'label_id' not in train.columns:
        raise KeyError("Expected column 'label_id' not found in the data.")
    dev['in_training'] = dev['label_id'].isin(set(train['label_id'].values))
    test['in_training'] = test['label_id'].isin(set(train['label_id'].values))

    # Map back to VOLL using mention_id (correct use of .isin)
    train_voll = df_voll[df_voll['mention_id'].isin(set(train['mention_id'].values))]
    test_voll = df_voll[df_voll['mention_id'].isin(set(test['mention_id'].values))]
    dev_voll = df_voll[df_voll['mention_id'].isin(set(dev['mention_id'].values))]

    # --- Save splits (Kurz) ---
    train.to_json(train_path, lines=True, orient='records', force_ascii=False)
    dev.to_json(valid_path, lines=True, orient='records', force_ascii=False)
    test.to_json(test_path, lines=True, orient='records', force_ascii=False)

    # --- Save splits (Voll) ---
    train_voll.to_json(train_path.replace('Kurz', 'Voll'), lines=True, orient='records', force_ascii=False)
    dev_voll.to_json(valid_path.replace('Kurz', 'Voll'), lines=True, orient='records', force_ascii=False)
    test_voll.to_json(test_path.replace('Kurz', 'Voll'), lines=True, orient='records', force_ascii=False)

    # --- Build the report ---
    # Initial lengths
    init_len_kurz = len(df_kurz)
    init_len_voll = len(df_voll)

    # Resulting lengths (Kurz)
    len_train = len(train)
    len_dev = len(dev)
    len_test = len(test)

    # value_counts for in_training
    dev_in_training = dev['in_training'].value_counts(dropna=False)
    test_in_training = test['in_training'].value_counts(dropna=False)

    # Ensure deterministic ordering of True/False in report
    dev_true = int(dev_in_training.get(True, 0))
    dev_false = int(dev_in_training.get(False, 0))
    test_true = int(test_in_training.get(True, 0))
    test_false = int(test_in_training.get(False, 0))

    # Default report path if not provided
    if report_path is None:
        base_dir = os.path.dirname(os.path.abspath(train_path))
        report_path = os.path.join(base_dir, "split_report.txt")

    # Write report (human-readable text)
    timestamp = datetime.now().isoformat(timespec='seconds')
    lines = [
        f"Split Report ({timestamp})",
        "-" * 60,
        "Input files:",
        f"  - input_path_kurz: {input_path_kurz}",
        f"  - input_path_voll: {input_path_voll}",
        "",
        "Initial lengths:",
        f"  - kurz: {init_len_kurz}",
        f"  - voll: {init_len_voll}",
        "",
        "Resulting split lengths (Kurz):",
        f"  - train: {len_train}",
        f"  - dev:   {len_dev}",
        f"  - test:  {len_test}",
        "",
        "Value counts for 'in_training':",
        "  - dev:",
        f"      True : {dev_true}",
        f"      False: {dev_false}",
        "  - test:",
        f"      True : {test_true}",
        f"      False: {test_false}",
        ""
    ]

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {
        "report_path": report_path,
        "init_lengths": {"kurz": init_len_kurz, "voll": init_len_voll},
        "split_lengths_kurz": {"train": len_train, "dev": len_dev, "test": len_test},
        "dev_in_training_counts": {"True": dev_true, "False": dev_false},
        "test_in_training_counts": {"True": test_true, "False": test_false},
    }



if __name__ == "__main__":
    input_jsonl_kurz = "/home/nobackup/wikipedia/la/lang_titles_Kurz_blink.jsonl"  # Change to your input file path
    input_jsonl_voll = "/home/nobackup/wikipedia/la/lang_titles_Voll_blink.jsonl"  # Change to your input file path    
    split_jsonl(
        input_path_kurz=input_jsonl_kurz,
        input_path_voll=input_jsonl_voll,
        train_path="/home/nobackup/wikipedia/la/v4/Kurz/train.jsonl",
        valid_path="/home/nobackup/wikipedia/la/v4/Kurz/valid.jsonl",
        test_path ="/home/nobackup/wikipedia/la/v4/Kurz/test.jsonl"
    )