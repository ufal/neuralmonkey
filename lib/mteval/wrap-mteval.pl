#!/usr/bin/env perl
use strict;
use warnings;
use autodie;
use Getopt::Long;
use File::Basename;
use File::Temp;

my $mteval_path = dirname(__FILE__); # expect mteval-13a.pl in the current directory by default
my $mteval_command = 'mteval-v13a.pl --international-tokenization -b';
my $help=0;


GetOptions(
  'mteval_path=s' => \$mteval_path,
  'mteval_command=s' => \$mteval_command,
  'h|help' => \$help,
) or exit 1;


my $USAGE = 'wrap-mteval.pl reference.txt < translation.txt';
if ($help){
    print "$USAGE\n";
    exit;
}

my $ref_tmp = File::Temp->new( UNLINK => 1, SUFFIX => '.ref');
my $tst_tmp = File::Temp->new( UNLINK => 1, SUFFIX => '.tst');
my $src_tmp = File::Temp->new( UNLINK => 1, SUFFIX => '.src');

my $reference_file = shift;
die "no reference file specified\n$USAGE\n" if !defined $reference_file;
open my $ref_txt, '<', $reference_file;
print {$ref_tmp} '<refset setid="sample-set" srclang="any" trglang="cs" sysid="manual">
<doc docid="sample-doc-1" origlang="cs" sysid="manual">
';
print {$src_tmp} '<srcset setid="sample-set" srclang="any">
<doc docid="sample-doc-1" origlang="en">
';

my $line_no=1;
while (<$ref_txt>){
    chomp;
    print {$ref_tmp} "<seg id=\"$line_no\">$_</seg>\n";
    print {$src_tmp} "<seg id=\"$line_no\">not needed</seg>\n";
    $line_no++;
}
print {$ref_tmp} '</doc>
</refset>
';
print {$src_tmp} '</doc>
</srcset>
';

print {$tst_tmp} '<tstset trglang="cs" setid="sample-set" srclang="any">
<doc sysid="sample-system" docid="sample-doc-1" origlang="en">
';
$line_no=1;
while(<>){
    chomp;
    print {$tst_tmp} "<seg id=\"$line_no\">$_</seg>\n";
    $line_no++;
}
print {$tst_tmp} '</doc>
</tstset>
';

my $command = "$mteval_path/$mteval_command -r $ref_tmp -s $src_tmp -t $tst_tmp";
warn "$command\n";
my $scores =`$command`;
#my ($nist, $bleu) = $scores =~ /NIST score = ([\d.]+)  BLEU score = ([\d.]+)/;
my ($bleu) = $scores =~ /BLEU score = ([\d.]+)/;
warn "$scores\n";
$bleu = $bleu * 100;
#print "$bleu\n$nist\n";

print "$bleu\n_\n";
