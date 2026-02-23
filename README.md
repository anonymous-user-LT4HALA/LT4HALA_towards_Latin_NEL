----
"CODE FOR THE PAPER *I, RE:Claudius 256*: Towards Linking Classical Latin Person Mentions to a Domain-specific Knowledge Base (WIP)"
----
This repo accompanies the paper, in which we explore using the BLINK architecture to link Latin texts to a domain-specific knowledge base. It provides a minimal codebase to reproduce the paper's results, the full training data for the Wikipedia training phase, along with additional scripts included for transparency. At present, this repository is a work in progress.

**Note**: Re-training is not possible because only the test set of the manually annotated (gold) data is included, per the annotators’ request. The full annotations will be released separately in a more complete version on their own pages.


## Directory Overview
- **blink_files**: Contains code based on the BLINK repository with adjustments.
- **training_scripts**: Contains the training scripts.
- **data**:
  - **silver_data**: Contains the scripts to create the silver data. These cannot be reproduced since the TM algorithm is not available to the public, but the scripts and intermediate steps (with this information removed) are included for transparency.
  - **blink_data**: Contains the test sets the models were tested on.
  - **Wikipedia_data**: TO BE ADDED
- **Paulys_kb**: Contains the version of the RE used for these experiments.
- **results**: Contains CSV files of the results and TXT files of the scores.


## How to use:
### Create the environment
```bash
conda env create -f environment.yml
```

### Download the models (to be added)
```bash
bash dowload_models.py
```

### Testing (to be added)
```bash
bash blink_training/test_models.sh
```

## Credits
We thank the following projects and people for making this research possible:

### Gold data annotators:
[TO BE ADDED UPON ACCEPTANCE]

### Silver data preliminaries:
**LASLA corpus**: [link](https://www.lasla.uliege.be/cms/c_11821932/fr/lasla-lasla-dataverse)


**TM database** [link](https://www.trismegistos.org/)

```bibtex
@inproceedings{depauw_trismegistos_2014,
	address = {Cham},
	series = {Communications in {Computer} and {Information} {Science}},
	title = {Trismegistos: {An} {Interdisciplinary} {Platform} for {Ancient} {World} {Texts} and {Related} {Information}},
	isbn = {978-3-319-08425-1},
	shorttitle = {Trismegistos},
	doi = {10.1007/978-3-319-08425-1_5},
	language = {en},
	booktitle = {Theory and {Practice} of {Digital} {Libraries} -- {TPDL} 2013 {Selected} {Workshops}},
	publisher = {Springer International Publishing},
	author = {Depauw, Mark and Gheldof, Tom},
	editor = {Bolikowski, Łukasz and Casarosa, Vittore and Goodale, Paula and Houssos, Nikos and Manghi, Paolo and Schirrwagen, Jochen},
	year = {2014},
	pages = {40--52},
}
```

### BLINK
```bibtex
@inproceedings{wu2019zero,
 title={Zero-shot Entity Linking with Dense Entity Retrieval},
 author={Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian Riedel, Luke Zettlemoyer},
 booktitle={EMNLP},
 year={2020}
}
```
[Repo](https://github.com/facebookresearch/BLINK.git)

### Wikisource RE
[Wikisource](https://de.wikisource.org/wiki/Paulys_Realencyclop%C3%A4die_der_classischen_Altertumswissenschaft)



Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
