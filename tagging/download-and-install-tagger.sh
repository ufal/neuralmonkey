#!/bin/bash

INSTALL_DIR=$HOME/treeTagger


# download data
parameter_file_url=http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/german-par-linux-3.2-utf8.bin.gz
tagger_url=http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.tar.gz
tagging_scripts_url=http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz
install_script_url=http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/install-tagger.sh

if [ -d $INSTALL_DIR ]; then
    echo "Tree tagger directory $INSTALL_DIR exists. Why?"
    exit 1
fi

mkdir $INSTALL_DIR

pushd $INSTALL_DIR

echo "Dwnld"

wget "$tagger_url"
wget "$tagging_scripts_url"
wget "$parameter_file_url"
wget "$install_script_url"

echo "installing"

sh install-tagger.sh

echo "testing"

echo 'Das ist ein Test. Wie geht es dir, mein Führer?' | cmd/tree-tagger-german

if [ $? == 0 ]; then
    echo "test looks ok, now it's up to you."

    echo "==== DONE"
    echo "consider adding $INSTAL_DIR/cmd and $INSTALL_DIR/bin folders into your \$PATH variable"
else
    echo "test returned nonzero status. time to think what went wrong"
fi

