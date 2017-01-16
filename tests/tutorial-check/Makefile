# This Makefile extracts config file from each tutorial in ../../docs/source/
SHELL=/bin/bash

TUTORIALS=tutorial machine_translation

all: $(TUTORIALS:%=%.check)

all-extracts: $(TUTORIALS:%=%.extract) $(TUTORIALS:%=%.dataset)

# This generic goal extracts inis from any tutorial:
.PRECIOUS: %.ini
%.extract: ../../docs/source/%.rst
	./extract_inis.pl < $<
	touch $@

# Prepare data for tutorial
tutorial.dataset: 
	touch $@

machine_translation.dataset:
	wget http://ufallab.ms.mff.cuni.cz/~popel/batch1and2.zip
	unzip -u batch1and2.zip -d exp-nm-mt/data/
	rm batch1and2.zip
	wget http://ufallab.ms.mff.cuni.cz/~popel/batch3.zip
	unzip -u batch3.zip -d exp-nm-mt/data/
	rm batch3.zip
	mkdir -p exp-nm-mt/data/train
	mkdir -p exp-nm-mt/data/dev
	mkdir -p exp-nm-mt/data/test
	cat exp-nm-mt/data/Batch1a_cs.txt | gzip > exp-nm-mt/data/train/Batch1a_cs.txt.gz
	cat exp-nm-mt/data/Batch1a_en.txt | gzip > exp-nm-mt/data/train/Batch1a_en.txt.gz
	cat exp-nm-mt/data/Batch2a_cs.txt | gzip > exp-nm-mt/data/dev/Batch2a_cs.txt.gz
	cat exp-nm-mt/data/Batch2a_en.txt | gzip > exp-nm-mt/data/dev/Batch2a_en.txt.gz
	cat exp-nm-mt/data/batch3 | gzip > exp-nm-mt/data/test/batch3.gz
	paste exp-nm-mt/data/Batch1a_en.txt exp-nm-mt/data/Batch1a_cs.txt | ../../lib/subword_nmt/learn_bpe.py -s 8000 > exp-nm-mt/data/merge_file.bpe
	touch $@

# Once the inis are ready, test them

tutorial.check: all-extracts
	../../bin/neuralmonkey-train post-edit.ini
	../../bin/neuralmonkey-train exp-nm-mt/translation.ini
