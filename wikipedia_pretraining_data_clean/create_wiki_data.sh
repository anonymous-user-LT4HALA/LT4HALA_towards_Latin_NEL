conda activate wikienv


# wget https://dumps.wikimedia.org/lawiki/latest/lawiki-latest-redirect.sql.gz  -O ../wikidata_kb/lawiki-latest-redirect.sql.gz
# wget https://dumps.wikimedia.org/lawiki/latest/lawiki-latest-page.sql.gz  -O ../wikidata_kb/lawiki-latest-page.sql.gz
# python ../wikidata_kb/extract_redirects.py

# python preprocess_wikidata.py compress --base_wikidata path/to/wikidata_kb 
# python preprocess_wikidata.py dicts  --base_wikidata path/to/wikidata_kb 
# python preprocess_wikidata.py redirects --base_wikidata path/to/wikidata_kb 

# python preprocess_extract.py --base_wikipedia /home/nobackup/wikipedia --lang de --rank 0
# python preprocess_extract.py --base_wikipedia /home/nobackup/wikipedia --lang de --rank 1 
# python preprocess_extract.py --base_wikipedia /home/nobackup/wikipedia --lang de --rank 2 
# python preprocess_extract.py --base_wikipedia /home/nobackup/wikipedia --lang de --rank 3 

# python preprocess_extract.py --base_wikipedia /home/nobackup/wikipedia --lang la --rank 0
# python preprocess_extract.py --base_wikipedia /home/nobackup/wikipedia --lang la --rank 1
# python preprocess_extract.py --base_wikipedia /home/nobackup/wikipedia --lang la --rank 2
# python preprocess_extract.py --base_wikipedia /home/nobackup/wikipedia --lang la --rank 3

# echo "Starting preprocessing of anchors"
# python preprocess_anchors.py prepare --base_wikipedia /home/nobackup/wikipedia --base_wikidata path/to/wikidata_kb --langs "la|de" -v --debug
# echo "Prepared anchors"
# python preprocess_anchors.py solve --base_wikipedia /home/nobackup/wikipedia --base_wikidata path/to/wikidata_kb --langs "la|de" -v --debug
# echo "Solved anchors"
# python preprocess_anchors.py fill --base_wikipedia /home/nobackup/wikipedia --base_wikidata path/to/wikidata_kb --langs "la|de" -v --debug
# echo "Filled anchors"
# python preprocess_anchors.py paragraphs --base_wikipedia /home/nobackup/wikipedia --base_wikidata path/to/wikidata_kb --langs "la|de" -v --debug
# echo "made paragraphs"
python preprocess_mgenre_adapted.py lang_titles --create_blink_kb de --base_wikipedia /home/nobackup/wikipedia --base_wikidata path/to/wikidata_kb  -v --langs la --target_switch_x_lingual "la|de" --description --RE_candidates Kurz --output_version LT4HALA
python preprocess_mgenre_adapted.py lang_titles --create_blink_kb de --base_wikipedia /home/nobackup/wikipedia --base_wikidata path/to/wikidata_kb  -v --langs la --target_switch_x_lingual "la|de" --description --RE_candidates Voll --output_version LT4HALA
