#!/usr/bin/perl

# This is a script for post-processing final result of the translation
# post-editing taks. It fixes some errors almost exclusively regarding
# punctuation which was particualarly difficult in the WMT 16 dataset.

use strict;
use warnings;
use utf8;

while (<>) {
    chomp;

    # multiple parentheses
    s/" " >/" > "/g;
    s/> " "/" > "/g;
    s/" "$/"/;
    s/ ([^,]) " "/ $1/g;

    # missing full-stop at the end
    s/([^.:")])$/$1 ./;

    # errors from true-casing
    s/Http/http/g;
    s/3d/3D/g;

    # if there is only one parenthesis, remove it
    my @par_count = (/"/g);
    if (@par_count == 1) {
        s/" //;
        s/"$//;
    }

    print "$_\n";
}
