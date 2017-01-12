# This Makefile extracts config file from each tutorial in ../../docs/source/
SHELL=/bin/bash

TUTORIALS=tutorial # machine_translation

all-checks: $(TUTORIALS:%=%.check)

all-inis: $(TUTORIALS:%=%.ini)

.PRECIOUS: %.ini
%.ini: ../../docs/source/%.rst
	cat $< \
	| sed -n '/code-block::/,/TUTCHECKEND/p' \
	| grep -v 'code-block::\|TUTCHECKEND' \
	> $@

%.check: %.ini
	../../bin/neuralmonkey-train $<
