# This Makefile extracts config file from each tutorial in ../../docs/source/
SHELL=/bin/bash

TUTORIALS=tutorial machine_translation # machine_translation

all: $(TUTORIALS:%=%.check)

all-extracts: $(TUTORIALS:%=%.extract)

# This generic goal extracts inis from any tutorial:
.PRECIOUS: %.ini
%.extract: ../../docs/source/%.rst
	./extract_inis.pl < $<
	touch $@

# Once the inis are ready, test them

tutorial.check:
	../../bin/neuralmonkey-train post-edit.ini
	../../bin/neuralmonkey-train translation.ini
