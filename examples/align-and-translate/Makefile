.PHONY
all: bitexts.en bitexts.fr

bitexts.en.gz: bitexts.selected
	cat bitexts.selected/{ccb2_pc30,crawl,dev08_11,ep7_pc45,nc9,un2000_pc34}.en.gz > $@

bitexts.fr.gz: bitexts.selected
	cat bitexts.selected/{ccb2_pc30,crawl,dev08_11,ep7_pc45,nc9,un2000_pc34}.fr.gz > $@

bitexts.selected: bitexts.tgz
	tar xpzf bitexts.tgz

bitexts.tgz:
	wget "http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/data/bitexts.tgz"

.PHONY
clean:
	$(RM) bitexts.tgz
	$(RM) bitexts.selected
