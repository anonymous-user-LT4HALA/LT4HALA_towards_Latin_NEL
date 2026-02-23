#!/bin/bash
which python
conda activate wikienv
which python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



# mkdir ../wikipedia
# cd ../wikipedia


mkdir -p ../wikipedia
cd ../wikipedia || exit 1

for LANG in la de
do
    wget https://mirror.accum.se/mirror/wikimedia.org/dumps/${LANG}wiki/latest/${LANG}wiki-latest-pages-articles-multistream.xml.bz2
done

for LANG in de la
do
    python -m wikiextractor.WikiExtractor \
        ${LANG}wiki-latest-pages-articles-multistream.xml.bz2 \
        -o ${LANG} \
        --links
done
