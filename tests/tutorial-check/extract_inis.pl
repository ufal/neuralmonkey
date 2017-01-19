#!/usr/bin/env perl
# process one .rst file and extract all .ini it contains.
# The contents are identified by starting with code-block::
# and ending with TUTCHECK <name>
# Blocks with code-block:: to be ignored should end with TUTCHECK IGNORE
## Now limited to code-block:: ini

use strict;
use File::Basename;
use File::Path qw/make_path/;

my %data;

my $blockstart = undef;
my $in_block = 0;
my $thisblock;
my $nr = 0;
while (<>) {
  $nr++;
  if (/code-block::\s*ini/) {
    $in_block = 1;
    $blockstart = $nr;
    next;
  }
  if (/TUTCHECK (.*)/) {
    my $fname = $1;
    $data{$fname} .= $thisblock if $fname ne "IGNORE";
    $thisblock = "";
    $in_block = 0;
  }
  $thisblock .= $_ if $in_block;
}
die "Missed TUTCHECK end after code-block at line $blockstart"
  if $in_block;

foreach my $fn (keys %data) {
  my $dir = dirname($fn);
  make_path($dir);
  open OUTF, ">$fn" or die "Can't write $fn";
  print OUTF $data{$fn};
  close OUTF;
  print STDERR "Saved $fn\n";
}

