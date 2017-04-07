#!/bin/bash

set -o pipefail

function die() { echo "$@" >&2; exit 1; }

mkdir yesno || die "Cannot create data directory"
cd yesno || exit 1

wget http://www.openslr.org/resources/1/waves_yesno.tar.gz -O- | tar xz \
  || die "Cannot download 'yesno' data"

function prep() {
  name=$1
  exp_count=$2
  shift; shift

  printf "%s\n" "$@" | LC_ALL=C sort >${name}.wavlist

  count=`wc -l <${name}.wavlist`
  [ "$count" -eq $exp_count ] \
    || die "Expected $exp_count training WAVs, got $count"

  grep -Eo '([01]_)+[01]' ${name}.wavlist | tr _ ' ' \
    | sed 's/0/NO/g; s/1/YES/g;' >${name}.txt \
    || die "Cannot create transcriptions"
}

prep train 31 waves_yesno/0_*.wav
prep test 29 waves_yesno/1_*.wav

echo >&2 "Data prepared."
