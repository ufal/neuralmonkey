# This Makefile extracts config file from each tutorial in ../../docs/source/
SHELL=/bin/bash

TUTORIALS=tutorial # machine_translation

all-checks: $(TUTORIALS:%=%.check)

all-extracts: $(TUTORIALS:%=%.extract)

.PRECIOUS: %.ini
%.extract: ../../docs/source/%.rst
	./extract_inis.pl < $<
	touch $@

%.check: %.ini
	../../bin/neuralmonkey-train $<
