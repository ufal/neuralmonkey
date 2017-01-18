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
	mkdir -p exp-nm-ape/data/test
	cp -r /net/data/WMT16/WMT16_APE_Shared_Task_Data/ exp-nm-ape/data/
	chmod -R u+w exp-nm-ape/data/WMT16_APE_Shared_Task_Data
	mv exp-nm-ape/data/WMT16_APE_Shared_Task_Data/TrainDev/* exp-nm-ape/data/
	mv exp-nm-ape/data/WMT16_APE_Shared_Task_Data/Test/* exp-nm-ape/data/test/
	mv exp-nm-ape/data/WMT16_APE_Shared_Task_Data/test_pe/* exp-nm-ape/data/test/
	../../scripts/postedit_prepare_data.py --translated-sentences=exp-nm-ape/data/train/train.mt --target-sentences=exp-nm-ape/data/train/train.pe > exp-nm-ape/data/train/train.edits
	../../scripts/postedit_prepare_data.py --translated-sentences=exp-nm-ape/data/dev/dev.mt --target-sentences=exp-nm-ape/data/dev/dev.pe > exp-nm-ape/data/dev/dev.edits
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
	gzip < exp-nm-mt/data/Batch1a_cs.txt > exp-nm-mt/data/train/Batch1a_cs.txt.gz
	gzip < exp-nm-mt/data/Batch1a_en.txt > exp-nm-mt/data/train/Batch1a_en.txt.gz
	gzip < exp-nm-mt/data/Batch2a_cs.txt > exp-nm-mt/data/dev/Batch2a_cs.txt.gz
	gzip < exp-nm-mt/data/Batch2a_en.txt > exp-nm-mt/data/dev/Batch2a_en.txt.gz
	gzip < exp-nm-mt/data/batch3 > exp-nm-mt/data/test/batch3.gz
	paste exp-nm-mt/data/Batch1a_en.txt exp-nm-mt/data/Batch1a_cs.txt \
    | ../../lib/subword_nmt/learn_bpe.py -s 8000 \
    > exp-nm-mt/data/merge_file.bpe
	touch $@

# Once the inis are ready, test them

tutorial.check: all-extracts
	../../bin/neuralmonkey-train post-edit.ini
	../../bin/neuralmonkey-train exp-nm-mt/translation.ini
