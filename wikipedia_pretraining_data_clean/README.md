# MGENRE summary
Summary of the scripts:
* `download_wiki.sh`: donwloads all Wikipedia
* `preprocess_anchors.py`: solves hyperlinks to Wikidata IDs and saves the Wikipedia file again
* `preprocess_extract.py`: extract Wikipedia files from wikiextractor and it constructs a dictionary (later saved into a pickle file)
* `preprocess_mention_dicts.py`: generates mention tables
* `preprocess_mgenre_adapted.py`: takes Wikipedia a pickle file (with solved hyperlinks) and it generates training data for the linking objective
* `preprocess_wikidata.py`: preprocess Wikidata 1) reducing the size from 1TB to 25GB removing unused items 2) generates useful dictionaries title -> ID, ID -> title or alias tables

# Wikipedia Entity Linking Dataset
Overview

This dataset is a Wikipedia-based resource for entity linking / entity disambiguation, constructed using modified versions of the scripts from the GENRE and mGENRE repositories. Meant as pre-training data for a BLINK model that should link Latin texts to a German knowledge base.

The dataset provides:

1) A BLINK jsonl-style dataset containing 30 000 mention-entity pairs, derived from hyperlinks in a Latin wikipedia page text to a German wikipedia page text. These have not been manually checked, save from a brief, informal look. All extracted entities are those present in the knowledge base detailed below.

Split Report (2026-01-20T16:18:07)
------------------------------------------------------------

Initial lengths:
  - kurz: 29421
  - voll: 29421

Resulting split lengths (Kurz):
  - train: 26665
  - dev:   1218
  - test:  1538

Value counts for 'in_training':
  - dev:
      True : 1096
      False: 122
  - test:
      True : 1420
      False: 118



2) A knowledge base, based the discussion paper 
"Wikidata as a Knowledge Base for People of the Graeco-Roman World" by
Margherita Fantoli, Valeria Irene Boano, Evelien de Graaf, and Camillo Carlo Pellizzari di San Girolamo. The entities come from the file /wikidata_kb_margherita/full_json_retrieved_entities.json which can be found on their repository [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/42QMWG].


Based on this, a BLINK jsonl-style version is made, containing only those entities that have German wikipedia page. These knowledge bases contains 11 652 entities and can be found blink_kb_de.jsonl.

Example:
{"wikidata_id": "Q999504", 
"label_id": 
11651, 
"text": "griechischer Historiker",
"title": "Ephoros der Jüngere", 
"metadata": {"pageweight": 1773.0}}



3) scripts to generate both, based on mGENRE and GENRE script. The preprocess_mgenre_adapted.py script underwent some changes to be adapted to our scenario and to support explicit cross-lingual linking.

