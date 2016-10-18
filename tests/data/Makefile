.PHONY: all
all: train.tc.en train.tc.de val.tc.en val.tc.de

.PHONY: clean
clean:
	$(RM) *.en *.de

%:
	wget "http://ufallab.ms.mff.cuni.cz/~musil/$@"
