#!/usr/bin/perl

use warnings;
use strict;

binmode(STDIN,":utf8");
binmode(STDOUT,":utf8");

while(<STDIN>) {
  $_ = " $_ ";

  # remove punctuation except apostrophe
  s/<space>/spacemark/g;  # for scoring
  s/'/apostrophe/g;
  s/[[:punct:]]/ /g;
  s/apostrophe/'/g;
  s/spacemark/<space>/g;  # for scoring

  # remove non-speech tokens for scoring
  s/\[cough\]//g;
  s/\[laugh\]//g;
  s/\[breath\]//g;
  s/\[mouthnoise\]//g;
  s/\[tone\]//g;
  s/\[hesitation\]//g;

  # remove whitespace
  s/\s+/ /g;
  s/^\s+//;
  s/\s+$//;

  print "$_\n";
}
